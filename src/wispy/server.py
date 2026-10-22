"""WSP server runtime: ``run_stdio`` and friends.

The runtime ties together the pure :mod:`wispy.protocol`,
:mod:`wispy.dispatcher`, :mod:`wispy.lifecycle`, and
:mod:`wispy.registry` modules with the asyncio
:class:`~wispy.transport.StdioTransport` to produce a working WSP
server. It is intentionally thin: every interesting decision lives in
one of the underlying modules. The runtime's responsibilities are:

* Construct a :class:`~wispy.lifecycle.LifecycleManager` for the run.
* Iterate the transport's framed messages, parsing each one and
  spawning a per-message asyncio task that calls
  :func:`~wispy.dispatcher.dispatch`.
* Dispatch messages that touch the lifecycle (``initialize``,
  ``shutdown``, ``exit``) inline in the read loop rather than as a
  task, so their state transitions are applied in arrival order:
  a message read after ``shutdown`` is always admitted against the
  ``SHUTTING_DOWN`` state, and ``exit`` takes effect as soon as it is
  read.
* Admit every other message against a snapshot of the lifecycle state
  taken when it was read. A task does not run until the read loop
  suspends, and the loop reads already-buffered messages without
  suspending, so without the snapshot a request pipelined ahead of
  ``exit`` would be rejected as "server has exited".
* Surface :class:`~wispy.framing.DecodeError` records to stderr and
  continue; no JSON-RPC response is emitted because
  no id is recoverable from a malformed frame.
* Drain in-flight handler tasks on stdin EOF, cancelling any that take
  longer than ``drain_timeout``, and return exit
  status ``0``.
* Surface the :class:`~wispy.dispatcher.ExitDispatch` sentinel produced
  for an ``exit`` notification by flushing any attached responses,
  draining outstanding tasks, and returning the carried status.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from wispy.dispatcher import ExitDispatch, _internal_error_response, dispatch
from wispy.framing import DecodeError
from wispy.lifecycle import LifecycleManager
from wispy.protocol import (
    JsonRpcRequest,
    JsonRpcResponse,
    ParseFailure,
    parse_message,
    serialize_response,
)
from wispy.transport import StdioTransport

if TYPE_CHECKING:
    from wispy.registry import HandlerRegistry

__all__ = ["run_stdio"]


# Methods whose handling advances the lifecycle FSM. Messages carrying
# any of them are dispatched inline by the read loop (see module
# docstring).
_LIFECYCLE_METHODS = frozenset({"initialize", "shutdown", "exit"})


async def run_stdio(
    registry: HandlerRegistry,
    *,
    transport: StdioTransport | None = None,
    drain_timeout: float = 5.0,
) -> int:
    """Run the WSP server over a stdio transport.

    The function returns when stdin reaches EOF (and any in-flight
    handler tasks have either completed or been cancelled after
    ``drain_timeout`` seconds), or when a dispatched ``exit``
    notification produces an :class:`ExitDispatch` sentinel and the
    in-flight tasks have drained. ``exit`` takes effect immediately,
    even if the client keeps stdin open.

    Args:
        registry: The :class:`HandlerRegistry` shared by every dispatched
            message. The runtime does not register any handlers itself;
            callers are responsible for binding ``initialize``,
            ``shutdown``, and any custom WSP methods before invoking
            ``run_stdio``.
        transport: An optional pre-built :class:`StdioTransport`. When
            ``None`` (the common case), a default transport bound to
            the real ``sys.stdin``/``sys.stdout``/``sys.stderr`` is
            constructed. Tests inject in-memory pipes here.
        drain_timeout: How long to wait, in seconds, for in-flight
            handler tasks to finish after stdin EOF (or after an
            ``exit`` notification) before cancelling them. Defaults to
            five seconds, which comfortably covers the typical
            sub-second handler while still bounding teardown.

    Returns:
        The desired process exit status: ``0`` for the EOF-drain path
        and for ``shutdown`` -> ``exit``, and
        whatever the lifecycle FSM chose otherwise (typically ``1``
        for ``exit`` without a preceding ``shutdown``.
    """
    if transport is None:
        transport = StdioTransport()
    lifecycle = LifecycleManager()
    tasks: set[asyncio.Task[None]] = set()
    # Set by ``handle_message`` once an ``exit`` notification has been
    # processed. ``exit`` is always dispatched inline, so the read loop
    # sees the status as soon as the message has been handled.
    exit_status: int | None = None

    async def handle_message(
        parsed: JsonRpcRequest | list[JsonRpcRequest | ParseFailure] | ParseFailure,
        admission: LifecycleManager,
    ) -> None:
        """Dispatch one parsed message and emit its responses.

        Runs either inline in the read loop (lifecycle messages, against
        the live ``lifecycle``) or as a per-message task (everything
        else, against a snapshot taken at read time). Never raises:
        failures are logged so a task cannot die with an unretrieved
        exception.
        """
        nonlocal exit_status
        try:
            result = await dispatch(
                parsed,
                registry=registry,
                lifecycle=admission,
                log=transport.log,
            )
        except Exception as exc:  # noqa: BLE001 - dispatcher MUST NOT raise; log and continue
            # The dispatcher contract guarantees no exceptions, but a
            # truly unexpected failure (e.g. a programming error in the
            # registry) should still be logged rather than propagated.
            transport.log(f"unexpected error in dispatch: {exc!r}")
            return

        response = result.response if isinstance(result, ExitDispatch) else result
        if response is not None:
            # Notifications (or batches consisting only of
            # notifications) produce ``None``: nothing to emit.
            try:
                await _emit(response, transport)
            except Exception as exc:  # noqa: BLE001 - a failed write must not kill the server
                transport.log(f"failed to write response: {exc!r}")

        if isinstance(result, ExitDispatch):
            # Set after any attached responses are on the wire so the
            # read loop cannot tear down before they are written.
            exit_status = result.status

    try:
        async with contextlib.aclosing(transport.messages()) as messages:
            async for item in messages:
                if isinstance(item, DecodeError):
                    # A malformed frame is reported on stderr and the
                    # read loop continues. No JSON-RPC response is
                    # produced because the request id, if any, is
                    # unrecoverable from a malformed frame.
                    transport.log(f"frame decode error: {item.reason} (discarded {item.discarded} bytes)")
                    continue

                try:
                    parsed = parse_message(item)
                except Exception as exc:  # noqa: BLE001 - one bad message must not stop the read loop
                    # ``parse_message`` maps malformed input to a
                    # ``ParseFailure``; this only guards against an
                    # unexpected failure, which leaves no id to answer.
                    transport.log(f"unexpected error parsing message: {exc!r}")
                    continue
                if _touches_lifecycle(parsed):
                    # Lifecycle transitions are only applied once the
                    # handler has succeeded. Waiting for them here keeps
                    # later messages from being admitted against stale
                    # state (e.g. a pipelined ``exit`` racing ahead of
                    # ``shutdown`` and exiting with status 1).
                    await handle_message(parsed, lifecycle)
                    if exit_status is not None:
                        break
                    continue

                task = asyncio.create_task(handle_message(parsed, lifecycle.snapshot()))
                tasks.add(task)
                # Self-prune the set so completed tasks can be garbage
                # collected without us scanning the set on every iteration.
                task.add_done_callback(tasks.discard)

    except Exception as exc:  # noqa: BLE001 - transport teardown must not propagate
        transport.log(f"unexpected error in transport: {exc!r}")

    # Drain in-flight handler tasks. ``asyncio.wait`` returns
    # ``(done, pending)`` after at most ``drain_timeout`` seconds; any
    # tasks still running are cancelled and gathered so their
    # cancellation propagates before the function returns.
    if tasks:
        # Snapshot the live tasks; ``tasks`` may be mutated by the
        # done-callback during the wait.
        in_flight = list(tasks)
        _, pending = await asyncio.wait(in_flight, timeout=drain_timeout)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # Flush any buffered stdout bytes one last time so the final
    # response is on the wire before we return, then release stdin.
    await transport.drain()
    transport.close()

    return exit_status if exit_status is not None else 0


def _touches_lifecycle(
    parsed: JsonRpcRequest | list[JsonRpcRequest | ParseFailure] | ParseFailure,
) -> bool:
    """``True`` iff ``parsed`` (or any batch entry) is a lifecycle method."""
    if isinstance(parsed, JsonRpcRequest):
        return parsed.method in _LIFECYCLE_METHODS
    if isinstance(parsed, list):
        return any(isinstance(e, JsonRpcRequest) and e.method in _LIFECYCLE_METHODS for e in parsed)
    return False


async def _emit(
    response: JsonRpcResponse | list[JsonRpcResponse],
    transport: StdioTransport,
) -> None:
    """Serialise ``response`` and write it through ``transport``.

    An empty batch response (``[]``) is silently dropped: per JSON-RPC
    2.0 a batch of only notifications produces no response object on
    the wire. Non-empty batches and single responses
    are framed and written via :meth:`StdioTransport.write`, which
    flushes stdout after each call so clients see responses
    immediately.

    A handler result that cannot be serialized (e.g. an arbitrary
    object, or ``NaN``) is replaced by a ``-32603`` response for the
    same id, so the client still gets an answer.
    """
    if isinstance(response, list) and not response:
        return
    try:
        payload = serialize_response(response)
    except (TypeError, ValueError) as exc:
        transport.log(f"failed to serialize response: {exc!r}")
        if isinstance(response, list):
            payload = serialize_response([_serializable_or_internal_error(r) for r in response])
        else:
            payload = serialize_response(_internal_error_response(response.id))
    await transport.write(payload)


def _serializable_or_internal_error(response: JsonRpcResponse) -> JsonRpcResponse:
    """Return ``response`` if it serializes, else a ``-32603`` for its id."""
    try:
        serialize_response(response)
    except (TypeError, ValueError):
        return _internal_error_response(response.id)
    return response
