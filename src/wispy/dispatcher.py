"""Async JSON-RPC dispatcher for WSP requests.

The dispatcher is the single point that turns a parsed JSON-RPC message
into one or more JSON-RPC responses (or ``None`` for notifications).
Aside from invoking handlers and consulting the lifecycle FSM, it is
purely functional from the caller's perspective: the same parsed
message, registry, and lifecycle state always produce the same result.

Top-level guarantees:

* The dispatcher MUST NOT raise. Every code path either returns a
  :class:`JsonRpcResponse`, a ``list[JsonRpcResponse]``, an
  :class:`ExitDispatch` sentinel, or ``None``. Any unexpected exception
  is converted to a ``-32603`` internal-error response (or ``None`` for
  notifications) and its traceback is logged via the supplied ``log``
  callable.
* Notifications never produce a response, even on error.
* Single-request error classification follows the
  priority order: parse error -> invalid request -> method not found
  -> invalid params.
* Batch dispatch is sequential. Sequential ordering keeps the
  lifecycle FSM well-defined when a batch contains methods that
  mutate it (``initialize``, ``shutdown``, ``exit``); any throughput
  loss is negligible since handler concurrency is achieved at the
  transport level by spawning one task per *message*, not per
  *batch entry*. Per-entry responses are emitted in input order with
  notifications elided.

The transport receives an :class:`ExitDispatch` whenever an ``exit``
notification is processed; the carried ``status`` is the value the
:class:`~wispy.lifecycle.LifecycleManager` wants the process to exit
with, and ``response`` carries any per-entry responses that should be
flushed before the runtime tears down.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wispy.endpoints import WSP_METHODS, Capabilities
from wispy.errors import (
    JsonRpcErrorCode,
    ProtocolError,
    WspError,
    WspErrorCode,
)
from wispy.lifecycle import Allow, Exit, LifecycleManager, RejectWith
from wispy.protocol import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    ParseFailure,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from wispy.registry import HandlerRegistry

__all__ = ["ExitDispatch", "dispatch"]


# --------------------------------------------------------------------- #
# Public sentinel for transport-level exit signalling.
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExitDispatch:
    """Sentinel returned by :func:`dispatch` for ``exit`` notifications.

    Attributes:
        status: The desired process exit status (``0`` if ``shutdown``
            preceded ``exit``, ``1`` otherwise).
        response: Any responses produced earlier in the same batch that
            the transport should flush before terminating, or ``None``.
    """

    status: int
    response: JsonRpcResponse | list[JsonRpcResponse] | None


# --------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------- #


_DispatchResult = (
    JsonRpcResponse
    | list[JsonRpcResponse]
    | ExitDispatch
    | None
)


async def dispatch(
    msg: JsonRpcRequest | list[JsonRpcRequest | ParseFailure] | ParseFailure,
    *,
    registry: HandlerRegistry,
    lifecycle: LifecycleManager,
    log: Callable[[str], None],
) -> _DispatchResult:
    """Translate one parsed JSON-RPC message into responses.

    See module docstring for guarantees. The function never raises;
    unexpected exceptions are caught at this top level, logged, and
    mapped to a ``-32603`` response (or ``None`` for notifications).
    """
    try:
        if isinstance(msg, ParseFailure):
            return _response_from_parse_failure(msg)
        if isinstance(msg, list):
            return await _dispatch_batch(
                msg,
                registry=registry,
                lifecycle=lifecycle,
                log=log,
            )
        if isinstance(msg, JsonRpcRequest):
            return await _dispatch_single(
                msg,
                registry=registry,
                lifecycle=lifecycle,
                log=log,
            )
        # Defensive: an unknown shape -- emit a generic internal error
        # with id null. The codec contract should prevent this branch.
        _safe_log(log, traceback.format_stack())
        return _internal_error_response(None)
    except Exception:  # noqa: BLE001 - see module docstring
        # The dispatcher MUST NOT raise. Catch *anything* and turn it
        # into a wire-emittable response.
        _safe_log(log, traceback.format_exc())
        request_id = msg.id if isinstance(msg, JsonRpcRequest) else None
        is_notification = isinstance(msg, JsonRpcRequest) and msg.is_notification
        if is_notification:
            return None
        return _internal_error_response(request_id)


# --------------------------------------------------------------------- #
# Single-request dispatch.
# --------------------------------------------------------------------- #


async def _dispatch_single(
    request: JsonRpcRequest,
    *,
    registry: HandlerRegistry,
    lifecycle: LifecycleManager,
    log: Callable[[str], None],
) -> JsonRpcResponse | ExitDispatch | None:
    """Dispatch one parsed :class:`JsonRpcRequest`.

    Returns either a single :class:`JsonRpcResponse`, an
    :class:`ExitDispatch` sentinel (only for ``exit`` notifications),
    or ``None`` (notifications and ``exit`` have no per-request
    response).
    """
    # ----- Lifecycle gate -------------------------------------------- #
    decision = lifecycle.admit(request.method, request.is_notification)

    if isinstance(decision, Exit):
        # ``exit`` notifications never produce a per-request response;
        # the transport learns the status from the ExitDispatch.
        status = lifecycle.on_exit()
        return ExitDispatch(status=status, response=None)

    if isinstance(decision, RejectWith):
        # Notifications never get a response, even on error.
        if request.is_notification:
            return None
        return JsonRpcResponse(id=request.id, error=decision.error)

    # decision is Allow.
    if not isinstance(decision, Allow):
        # Defensive: unknown decision variant -- treat as internal error.
        _safe_log(
            log,
            f"unrecognized lifecycle admit decision: {decision!r}",
        )
        if request.is_notification:
            return None
        return _internal_error_response(request.id)

    # ----- Method lookup --------------------------------------------- #
    handler = registry.lookup(request.method)
    if handler is None:
        if request.is_notification:
            return None
        return JsonRpcResponse(
            id=request.id,
            error=JsonRpcError(
                code=int(JsonRpcErrorCode.METHOD_NOT_FOUND),
                message=f"method not found: {request.method}",
            ),
        )

    # ----- Param validation ------------------------------------------ #
    method_def = WSP_METHODS.get(request.method)
    if method_def is not None:
        try:
            validated = method_def.validate_params(request.params)
        except Exception:  # noqa: BLE001 - validators should not raise
            _safe_log(log, traceback.format_exc())
            if request.is_notification:
                return None
            return _internal_error_response(request.id)
        if isinstance(validated, list):
            # The WspMethod.validate_params contract uses ``list`` to
            # signal "violations"; any non-list (including ``None`` and
            # normalized dataclasses) means success.
            if request.is_notification:
                return None
            return JsonRpcResponse(
                id=request.id,
                error=JsonRpcError(
                    code=int(JsonRpcErrorCode.INVALID_PARAMS),
                    message="invalid params",
                    data={"violations": list(validated)},
                ),
            )
        params_for_handler: Any = validated
    else:
        # Method registered but not in WSP_METHODS: skip validation and
        # pass params through unchanged. This supports custom non-WSP
        # methods registered programmatically.
        params_for_handler = request.params

    # ----- Handler invocation ---------------------------------------- #
    try:
        result = await _invoke_handler(handler, params_for_handler)
    except WspError as err:
        if request.is_notification:
            return None
        return _wsp_error_response(request.id, err)
    except ProtocolError as err:
        # ProtocolError without a WSP code maps to execution-failed.
        if request.is_notification:
            return None
        return JsonRpcResponse(
            id=request.id,
            error=JsonRpcError(
                code=int(WspErrorCode.EXECUTION_FAILED),
                message=err.message or "execution failed",
            ),
        )
    except Exception:  # noqa: BLE001 - blanket catch for handler errors
        _safe_log(log, traceback.format_exc())
        if request.is_notification:
            return None
        return _internal_error_response(request.id)

    # ----- Lifecycle edge transitions on success --------------------- #
    if request.method == "initialize":
        try:
            # Handlers may return Capabilities or its documented
            # ``to_jsonable()`` dict; either way the FSM caches a real
            # Capabilities. Anything else is logged but still cached so
            # downstream observers see a consistent FSM transition.
            caps = _as_capabilities(result)
            if caps is None:
                _safe_log(
                    log,
                    f"warning: initialize handler did not return Capabilities (got {type(result).__name__})",
                )
            lifecycle.on_initialize_success(caps if caps is not None else result)
        except Exception:  # noqa: BLE001 - lifecycle edge must not abort
            _safe_log(log, traceback.format_exc())
    elif request.method == "shutdown":
        try:
            lifecycle.on_shutdown_success()
        except Exception:  # noqa: BLE001
            _safe_log(log, traceback.format_exc())

    # ----- Response -------------------------------------------------- #
    if request.is_notification:
        return None
    return JsonRpcResponse(id=request.id, result=result)


# --------------------------------------------------------------------- #
# Batch dispatch.
# --------------------------------------------------------------------- #


async def _dispatch_batch(
    entries: list[JsonRpcRequest | ParseFailure],
    *,
    registry: HandlerRegistry,
    lifecycle: LifecycleManager,
    log: Callable[[str], None],
) -> list[JsonRpcResponse] | ExitDispatch:
    """Dispatch a batch of parsed entries sequentially.

    The response array length equals the count of
    non-notification entries that produced a response, and order
    matches the input. Sequential dispatch keeps lifecycle FSM
    transitions deterministic.

    Empty batches never reach this function: the codec converts them to
    a top-level ``INVALID_REQUEST`` :class:`ParseFailure`.
    """
    responses: list[JsonRpcResponse] = []
    for entry in entries:
        if isinstance(entry, ParseFailure):
            # Per-entry parse failures get a per-entry response so the
            # client can correlate (when the id was recoverable). The
            # codec sets ``entry.id = None`` when the id was malformed
            # or absent, so the response carries ``"id": null``.
            responses.append(_response_from_parse_failure(entry))
            continue
        if not isinstance(entry, JsonRpcRequest):
            # Defensive: an unrecognized batch entry shape. Emit a
            # generic invalid-request response with id null.
            responses.append(
                JsonRpcResponse(
                    id=None,
                    error=JsonRpcError(
                        code=int(JsonRpcErrorCode.INVALID_REQUEST),
                        message="invalid batch entry",
                    ),
                )
            )
            continue

        result = await _dispatch_single(
            entry,
            registry=registry,
            lifecycle=lifecycle,
            log=log,
        )
        if isinstance(result, ExitDispatch):
            # An ``exit`` notification inside a batch: stop processing
            # further entries and surface the exit, attaching whatever
            # responses were produced so the transport flushes them.
            attached: list[JsonRpcResponse] | None
            attached = responses if responses else None
            return ExitDispatch(status=result.status, response=attached)
        if result is None:
            continue
        responses.append(result)
    return responses


# --------------------------------------------------------------------- #
# Handler invocation helper.
# --------------------------------------------------------------------- #


async def _invoke_handler(handler: Callable[..., Any], params: Any) -> Any:
    """Invoke ``handler`` with ``params`` and return its (awaited) result.

    Async handlers (detected by :func:`inspect.iscoroutinefunction`)
    are awaited directly. Sync handlers are off-loaded to the default
    executor so they cannot block the event loop. As a final
    accommodation, if the executor's return value happens to be
    awaitable (e.g. a sync wrapper that returns a coroutine), it is
    awaited too.

    ``asyncio.iscoroutinefunction`` is deliberately not consulted: it
    is deprecated since Python 3.14 and warns on every call.
    """
    if inspect.iscoroutinefunction(handler):
        return await handler(params)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, handler, params)
    if inspect.isawaitable(result):
        result = await result
    return result


def _as_capabilities(result: Any) -> Capabilities | None:
    """Return ``result`` as :class:`Capabilities`, or ``None`` if it is not one.

    Accepts the dataclass itself or its ``to_jsonable()`` dict form.
    """
    if isinstance(result, Capabilities):
        return result
    if isinstance(result, dict):
        try:
            return Capabilities.from_jsonable(result)
        except (TypeError, ValueError):
            return None
    return None


# --------------------------------------------------------------------- #
# Response constructors.
# --------------------------------------------------------------------- #


def _response_from_parse_failure(failure: ParseFailure) -> JsonRpcResponse:
    """Render a :class:`ParseFailure` as a JSON-RPC error response."""
    return JsonRpcResponse(
        id=failure.id,
        error=JsonRpcError(code=failure.code, message=failure.message),
    )


def _internal_error_response(
    request_id: str | int | None,
) -> JsonRpcResponse:
    """``-32603`` internal error response."""
    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(
            code=int(JsonRpcErrorCode.INTERNAL_ERROR),
            message="internal error",
        ),
    )


def _wsp_error_response(
    request_id: str | int | None,
    err: WspError,
) -> JsonRpcResponse:
    """Render a :class:`WspError` into a wire-emittable response.

    ``data`` is omitted from the wire when
    the error carries no structured context (``data is None``).
    """
    if err.data is None:
        json_err = JsonRpcError(code=err.code, message=err.message)
    else:
        json_err = JsonRpcError(
            code=err.code,
            message=err.message,
            data=err.data,
        )
    return JsonRpcResponse(id=request_id, error=json_err)


# --------------------------------------------------------------------- #
# Internal logging helper.
# --------------------------------------------------------------------- #


def _safe_log(log: Callable[[str], None], message: object) -> None:
    """Invoke the user-supplied log callable, swallowing any failure.

    The dispatcher's "MUST NOT raise" guarantee extends to misbehaving
    log callables: if logging itself raises, the dispatcher silently
    drops the diagnostic rather than propagating.
    """
    with contextlib.suppress(Exception):
        log(str(message))
