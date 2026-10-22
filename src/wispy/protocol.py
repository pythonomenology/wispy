"""JSON-RPC 2.0 types and codec for the WSP server.

Pure data structures, parsing, and serialization. This module performs no
I/O so it can be exhaustively unit- and property-tested in isolation.

The codec implements only what the WSP spec requires:

* Single requests and batch requests (JSON-RPC 2.0 sections 4 and 6).
* Notifications (requests without an ``id`` member).
* Standard error codes ``-32700`` (parse error) and ``-32600``
  (invalid request).

Public surface:

* :class:`JsonRpcRequest`, :class:`JsonRpcResponse`, :class:`JsonRpcError`
  -- frozen dataclass representations of the wire types.
* :class:`ParseFailure` and :class:`ParseFailureKind` -- a recoverable
  parse failure suitable for rendering as a ``-32700`` or ``-32600``
  response.
* :func:`parse_message` -- bytes -> request(s) or :class:`ParseFailure`.
* :func:`serialize_response` -- response(s) -> UTF-8 bytes.

The ``id`` field on requests and responses is preserved as the original
JSON type (``str``, ``int``, or ``None``); the codec rejects floats,
booleans, non-finite numbers, and non-scalar JSON values as request ids
per the JSON-RPC 2.0 spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from wispy.errors import JsonRpcErrorCode

__all__ = [
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "ParseFailure",
    "ParseFailureKind",
    "parse_message",
    "serialize_response",
]


# --------------------------------------------------------------------- #
# Sentinel for "field absent on the wire".
# --------------------------------------------------------------------- #


class _UnsetType(Enum):
    """Type of the :data:`_UNSET` sentinel.

    Modeled as a single-member enum so that ``_UNSET`` is a singleton with
    a stable identity (``is _UNSET``) and a useful ``repr`` for debugging.
    """

    UNSET = "UNSET"

    def __bool__(self) -> bool:  # pragma: no cover - cosmetic
        return False


_UNSET: Final = _UnsetType.UNSET


# --------------------------------------------------------------------- #
# Wire-shape dataclasses.
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class JsonRpcRequest:
    """A parsed JSON-RPC 2.0 request.

    Attributes:
        method: The method name from the request (always a string).
        params: The ``params`` value -- a list (JSON array) or a dict
            (JSON object) -- or ``None`` if the request had no
            ``params`` member.
        id: The request id preserved as its original JSON type
            (``str``, ``int``, or ``None`` for JSON null or for
            notifications).
        is_notification: ``True`` iff the original request had no ``id``
            member at all. ``False`` when ``id`` was present, including
            when it was explicitly set to JSON null.
    """

    method: str
    params: Any | None
    id: str | int | None
    is_notification: bool


@dataclass(frozen=True)
class JsonRpcError:
    """A JSON-RPC error object.

    The ``data`` field defaults to the :data:`_UNSET` sentinel, in which
    case :func:`serialize_response` omits ``data`` from the emitted JSON
    object. Setting ``data`` to ``None`` (or any other value) causes it
    to be included on the wire.
    """

    code: int
    message: str
    data: Any = _UNSET


@dataclass(frozen=True)
class JsonRpcResponse:
    """A JSON-RPC response.

    Exactly one of :attr:`result` and :attr:`error` must be provided;
    both default to the :data:`_UNSET` sentinel. The constructor raises
    ``ValueError`` if neither or both are set.
    """

    id: str | int | None
    result: Any = _UNSET
    error: Any = _UNSET

    def __post_init__(self) -> None:
        result_set = self.result is not _UNSET
        error_set = self.error is not _UNSET
        if result_set and error_set:
            msg = "JsonRpcResponse must have exactly one of 'result' or 'error'"
            raise ValueError(msg)
        if not result_set and not error_set:
            msg = "JsonRpcResponse must have exactly one of 'result' or 'error'"
            raise ValueError(msg)
        if error_set and not isinstance(self.error, JsonRpcError):
            msg = f"JsonRpcResponse.error must be a JsonRpcError, got {type(self.error).__name__}"
            raise TypeError(msg)


# --------------------------------------------------------------------- #
# Parse-failure record.
# --------------------------------------------------------------------- #


class ParseFailureKind(Enum):
    """Why :func:`parse_message` rejected a message.

    * :attr:`PARSE_ERROR` -- bytes were not valid UTF-8 JSON. Maps to
      JSON-RPC error code ``-32700``.
    * :attr:`INVALID_REQUEST` -- bytes parsed as JSON but the resulting
      value is not a valid JSON-RPC 2.0 request. Maps to ``-32600``.
    """

    PARSE_ERROR = "PARSE_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"


@dataclass(frozen=True)
class ParseFailure:
    """A failure record produced by :func:`parse_message`.

    Carries enough information to render either a ``-32700`` (parse
    error) or ``-32600`` (invalid request) response.

    For top-level failures, :attr:`id` is always ``None`` so the
    dispatcher emits ``"id": null`` per the JSON-RPC 2.0 spec. For
    per-batch-entry failures, :attr:`id` may carry the recovered id of
    the offending entry (when the id itself was a valid ``str`` or
    ``int``) so the dispatcher can echo it in the per-entry response.
    """

    kind: ParseFailureKind
    message: str
    id: str | int | None = None

    @property
    def code(self) -> int:
        """JSON-RPC error code corresponding to :attr:`kind`."""
        if self.kind is ParseFailureKind.PARSE_ERROR:
            return int(JsonRpcErrorCode.PARSE_ERROR)
        return int(JsonRpcErrorCode.INVALID_REQUEST)


# --------------------------------------------------------------------- #
# Public API: parse_message / serialize_response.
# --------------------------------------------------------------------- #


# ``parse_message`` returns one of:
#
# * :class:`JsonRpcRequest` -- a single, well-formed request.
# * :class:`ParseFailure` -- a top-level failure (the whole message is
#   not JSON or is not a valid JSON-RPC 2.0 request).
# * ``list[JsonRpcRequest | ParseFailure]`` -- a non-empty batch.
#   Per JSON-RPC 2.0 individual entries in a batch may be invalid; we
#   surface them per-entry so the dispatcher can emit per-entry
#   responses (and skip notification-shaped invalid entries). The
#   declared return type uses ``list[Any]`` to keep the signature
#   compact; consumers should treat the list as
#   ``list[JsonRpcRequest | ParseFailure]``.
ParsedMessage = JsonRpcRequest | list["JsonRpcRequest | ParseFailure"] | ParseFailure


def parse_message(raw: bytes) -> JsonRpcRequest | list[JsonRpcRequest | ParseFailure] | ParseFailure:
    """Parse a single JSON-RPC 2.0 message from raw bytes.

    Returns either a :class:`JsonRpcRequest` (single request), a list of
    ``JsonRpcRequest | ParseFailure`` (non-empty batch), or a
    :class:`ParseFailure` describing why the bytes could not be
    interpreted as a valid JSON-RPC 2.0 message.

    Distinguishes "not JSON" (``PARSE_ERROR``) from "JSON but not a
    valid JSON-RPC 2.0 request" (``INVALID_REQUEST``) so the dispatcher
    can apply the priority order.
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        msg = f"raw must be a bytes-like object, got {type(raw).__name__}"
        raise TypeError(msg)

    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError as exc:
        return ParseFailure(
            ParseFailureKind.PARSE_ERROR,
            f"message is not valid UTF-8: {exc.reason}",
        )

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParseFailure(
            ParseFailureKind.PARSE_ERROR,
            f"message is not valid JSON: {exc.msg}",
        )
    except RecursionError:
        # ``json.loads`` recurses once per nesting level, so a payload
        # like ``[[[[...`` exhausts the stack before it can be rejected.
        return ParseFailure(
            ParseFailureKind.PARSE_ERROR,
            "message is nested too deeply",
        )

    if isinstance(value, list):
        if not value:
            return ParseFailure(
                ParseFailureKind.INVALID_REQUEST,
                "batch request must be a non-empty array",
            )
        # Per the design: batch with at least one entry returns a list of
        # per-entry results so the dispatcher can produce per-entry
        # responses.
        return [_parse_request_object(entry) for entry in value]

    return _parse_request_object(value)


def serialize_response(
    resp: JsonRpcResponse | list[JsonRpcResponse],
) -> bytes:
    """Serialize a response (or batch) into a single UTF-8 JSON document.

    Always emits ``"jsonrpc": "2.0"``. For success responses, emits
    ``result``. For error responses, emits ``error: {code, message[,
    data]}`` and omits ``data`` when it is unset. Ids are serialized
    back to their original JSON type, including JSON null.

    Objects exposing a ``to_jsonable()`` method (the wire data models in
    :mod:`wispy.endpoints`) are converted through it, so handlers may
    return :class:`~wispy.endpoints.Capabilities`,
    :class:`~wispy.endpoints.Environment`, etc. directly. Any other
    non-JSON value raises :class:`TypeError`.
    """
    if isinstance(resp, list):
        body: Any = [_response_to_jsonable(r) for r in resp]
    else:
        body = _response_to_jsonable(resp)
    return json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


# --------------------------------------------------------------------- #
# Internal helpers.
# --------------------------------------------------------------------- #


def _parse_request_object(value: Any) -> JsonRpcRequest | ParseFailure:
    """Validate a single decoded JSON value as a JSON-RPC 2.0 request."""
    if not isinstance(value, dict):
        return ParseFailure(
            ParseFailureKind.INVALID_REQUEST,
            f"request must be a JSON object, got {_jsontype(value)}",
        )

    id_present, id_ok, recovered_id = _extract_id(value)

    if value.get("jsonrpc") != "2.0":
        return ParseFailure(
            ParseFailureKind.INVALID_REQUEST,
            "missing or invalid 'jsonrpc' field; expected \"2.0\"",
            id=recovered_id,
        )

    if "method" not in value:
        return ParseFailure(
            ParseFailureKind.INVALID_REQUEST,
            "missing 'method' field",
            id=recovered_id,
        )
    method = value["method"]
    if not isinstance(method, str):
        return ParseFailure(
            ParseFailureKind.INVALID_REQUEST,
            f"'method' must be a string, got {_jsontype(method)}",
            id=recovered_id,
        )

    if not id_ok:
        # Per spec: when the id cannot be detected, the response id is
        # null. We surface that here by deliberately not echoing the
        # malformed id.
        return ParseFailure(
            ParseFailureKind.INVALID_REQUEST,
            "'id' must be a string, integer, or null",
            id=None,
        )

    if "params" in value:
        params = value["params"]
        if not isinstance(params, (list, dict)):
            return ParseFailure(
                ParseFailureKind.INVALID_REQUEST,
                f"'params' must be an array or object, got {_jsontype(params)}",
                id=recovered_id,
            )
    else:
        params = None

    is_notification = not id_present
    return JsonRpcRequest(
        method=method,
        params=params,
        id=recovered_id,
        is_notification=is_notification,
    )


def _extract_id(value: dict[str, Any]) -> tuple[bool, bool, str | int | None]:
    """Inspect the ``id`` member of a request object.

    Returns ``(id_present, id_ok, recovered_id)``:

    * ``id_present``: ``True`` iff the dict has an ``"id"`` key
      (independent of its value). Used to decide notification status.
    * ``id_ok``: ``False`` if the id is present but of an unsupported
      JSON type (boolean, float, list, dict, etc.). Booleans are int
      subclasses in Python so we reject them explicitly. Floats and
      non-finite numbers (``NaN``, ``Infinity``, ``-Infinity`` -- which
      Python's ``json`` module decodes to floats) are rejected per the
      WSP design notes.
    * ``recovered_id``: a usable id for echoing in error responses --
      ``None`` whenever ``id_ok`` is ``False`` or when the id is
      explicitly JSON null or absent.
    """
    if "id" not in value:
        return False, True, None
    raw_id = value["id"]
    if isinstance(raw_id, bool):
        return True, False, None
    if raw_id is None:
        return True, True, None
    if isinstance(raw_id, str):
        return True, True, raw_id
    if isinstance(raw_id, int):
        return True, True, raw_id
    return True, False, None


def _response_to_jsonable(resp: JsonRpcResponse) -> dict[str, Any]:
    """Convert a :class:`JsonRpcResponse` to a plain ``dict`` for JSON."""
    if not isinstance(resp, JsonRpcResponse):
        msg = f"expected JsonRpcResponse, got {type(resp).__name__}"
        raise TypeError(msg)
    out: dict[str, Any] = {"jsonrpc": "2.0"}
    if resp.result is not _UNSET:
        out["result"] = resp.result
    else:
        err = resp.error
        # JsonRpcResponse.__post_init__ guarantees this is a JsonRpcError.
        if not isinstance(err, JsonRpcError):
            msg = f"JsonRpcResponse.error must be a JsonRpcError, got {type(err).__name__}"
            raise TypeError(msg)
        err_obj: dict[str, Any] = {"code": err.code, "message": err.message}
        if err.data is not _UNSET:
            err_obj["data"] = err.data
        out["error"] = err_obj
    out["id"] = resp.id
    return out


def _json_default(value: Any) -> Any:
    """``json.dumps`` fallback: encode wire data models via ``to_jsonable``."""
    to_jsonable = getattr(value, "to_jsonable", None)
    if callable(to_jsonable):
        return to_jsonable()
    msg = f"Object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def _jsontype(value: Any) -> str:
    """Return a human-readable name for the JSON type of ``value``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__
