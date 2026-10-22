"""Content-Length framing for the WSP stdio transport.

This module is pure (no I/O) so it can be exhaustively property-tested.

The wire format follows the LSP base protocol: each message is preceded by a
``Content-Length: <N>\\r\\n\\r\\n`` header block, after which the next ``N``
bytes are the UTF-8 encoded JSON payload.

Public surface:

* :func:`encode_frame` -- encode a payload into a complete frame.
* :class:`DecodeError` -- record describing bytes the decoder discarded.
* :class:`FrameDecoder` -- streaming decoder that tolerates arbitrary chunking
  and recovers from malformed framing or non-UTF-8 payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["DEFAULT_MAX_CONTENT_LENGTH", "DecodeError", "FrameDecoder", "encode_frame"]


# Largest payload :class:`FrameDecoder` accepts by default. A frame that
# declares more is skipped (and reported) as its bytes arrive instead of
# being buffered, so a hostile or buggy peer cannot exhaust memory.
DEFAULT_MAX_CONTENT_LENGTH = 64 * 1024 * 1024

# A legitimate header block is a few dozen bytes. Once this many bytes
# are buffered without a ``\r\n\r\n`` the stream is garbage, and is
# discarded rather than buffered forever.
_MAX_HEADER_BYTES = 8 * 1024

_HEADER_TERMINATOR = b"\r\n\r\n"
_LINE_SEPARATOR = b"\r\n"
_CONTENT_LENGTH_NAME = "content-length"


def encode_frame(payload: bytes) -> bytes:
    """Encode ``payload`` as a Content-Length framed message.

    The returned bytes are ``b"Content-Length: <N>\\r\\n\\r\\n"`` followed by
    ``payload`` verbatim, where ``<N>`` is ``len(payload)``.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        msg = f"payload must be a bytes-like object, got {type(payload).__name__}"
        raise TypeError(msg)
    payload_bytes = bytes(payload)
    return b"Content-Length: " + str(len(payload_bytes)).encode("ascii") + _HEADER_TERMINATOR + payload_bytes


@dataclass(frozen=True)
class DecodeError:
    """Diagnostic record emitted when the decoder skips malformed bytes.

    Attributes:
        discarded: Number of bytes that were dropped from the stream while
            recovering to the next plausible frame boundary.
        reason: Human-readable description of why the bytes were discarded.
    """

    discarded: int
    reason: str


class FrameDecoder:
    """Streaming Content-Length frame decoder.

    Feed arbitrary byte chunks via :meth:`feed`. Each call returns an iterator
    that yields complete frame payloads (as ``bytes``) and :class:`DecodeError`
    records for malformed input. The decoder maintains an internal buffer so
    it is resilient to arbitrary chunk boundaries: a payload need only be
    yielded once both the header terminator ``\\r\\n\\r\\n`` and the full
    ``Content-Length`` worth of bytes have been observed.

    On a malformed header block or a payload that is not valid UTF-8, the
    decoder emits a :class:`DecodeError` describing the discarded bytes and
    advances past the offending region so subsequent valid frames are still
    decoded.

    Memory is bounded: a frame whose ``Content-Length`` exceeds
    ``max_content_length`` is reported immediately and its payload bytes
    are dropped as they arrive, and more than ``_MAX_HEADER_BYTES`` of
    input without a header terminator is discarded as garbage. Pass
    ``max_content_length=None`` to disable the payload limit.
    """

    __slots__ = ("_buffer", "_max_content_length", "_skip")

    def __init__(self, max_content_length: int | None = DEFAULT_MAX_CONTENT_LENGTH) -> None:
        self._buffer: bytearray = bytearray()
        self._max_content_length = max_content_length
        # Payload bytes of an oversized frame still to be dropped.
        self._skip = 0

    def feed(self, chunk: bytes) -> Iterator[bytes | DecodeError]:
        """Append ``chunk`` to the internal buffer and yield available items.

        Yields each complete payload (as ``bytes``) or a :class:`DecodeError`
        for any region of the stream that could not be parsed as a valid
        frame. Returns when the buffer is exhausted of complete items.
        """
        if chunk:
            self._buffer.extend(chunk)
        while True:
            item = self._extract_next()
            if item is None:
                return
            yield item

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _extract_next(self) -> bytes | DecodeError | None:
        """Attempt to extract a single frame or DecodeError from the buffer.

        Returns ``None`` when more bytes are needed before any complete item
        can be produced. Successful extraction (either a payload or a recovery
        DecodeError) drops the corresponding bytes from the front of the
        buffer.
        """
        if self._skip:
            dropped = min(self._skip, len(self._buffer))
            del self._buffer[:dropped]
            self._skip -= dropped
            if self._skip:
                return None

        terminator_index = self._buffer.find(_HEADER_TERMINATOR)
        if terminator_index == -1:
            if len(self._buffer) > _MAX_HEADER_BYTES:
                # Keep the last few bytes: they may be the start of a
                # terminator split across chunks.
                discarded = len(self._buffer) - (len(_HEADER_TERMINATOR) - 1)
                del self._buffer[:discarded]
                return DecodeError(
                    discarded=discarded,
                    reason=f"no header terminator within {_MAX_HEADER_BYTES} bytes",
                )
            # No complete header block yet. Wait for more bytes.
            return None

        header_block = bytes(self._buffer[:terminator_index])
        header_total = terminator_index + len(_HEADER_TERMINATOR)

        parsed = _parse_headers(header_block)
        if isinstance(parsed, str):
            # Malformed header block: drop everything up through the
            # terminator and report the discarded count.
            del self._buffer[:header_total]
            return DecodeError(discarded=header_total, reason=parsed)

        content_length = parsed

        if self._max_content_length is not None and content_length > self._max_content_length:
            # Drop the header now and the payload as it arrives, rather
            # than buffering it only to throw it away.
            del self._buffer[:header_total]
            self._skip = content_length
            return DecodeError(
                discarded=header_total + content_length,
                reason=f"Content-Length {content_length} exceeds the maximum of {self._max_content_length} bytes",
            )

        # Payload may not be fully buffered yet.
        if len(self._buffer) - header_total < content_length:
            return None

        payload_end = header_total + content_length
        payload = bytes(self._buffer[header_total:payload_end])

        # Validate UTF-8.
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            del self._buffer[:payload_end]
            return DecodeError(
                discarded=payload_end,
                reason=f"payload is not valid UTF-8: {exc.reason}",
            )

        del self._buffer[:payload_end]
        return payload


def _parse_headers(header_block: bytes) -> int | str:
    """Parse an LSP-style header block.

    Returns the integer ``Content-Length`` value on success, or a
    human-readable reason string on failure. Header names are matched
    case-insensitively; unknown header names (e.g. ``Content-Type``) are
    accepted and ignored.
    """
    if not header_block:
        return "empty header block"

    try:
        text = header_block.decode("ascii")
    except UnicodeDecodeError:
        return "header block contains non-ASCII bytes"

    content_length: int | None = None
    for line in text.split("\r\n"):
        if line == "":
            # An empty line inside the header block is malformed; the only
            # legitimate empty separator is the trailing \r\n\r\n which is
            # not part of header_block.
            return "empty header line within header block"
        if ":" not in line:
            return f"header line missing ':' separator: {line!r}"
        name, _, value = line.partition(":")
        name = name.strip()
        value = value.strip()
        if not name:
            return f"header line has empty name: {line!r}"
        if name.lower() != _CONTENT_LENGTH_NAME:
            # Unknown header (e.g. Content-Type) -- explicitly ignored.
            continue
        if not value:
            return "Content-Length header has empty value"
        # ``str.isdigit`` accepts only ASCII digits 0-9, which is exactly
        # what we want: no leading sign, no whitespace, no Unicode digits.
        if not value.isdigit():
            return f"Content-Length is not a non-negative integer: {value!r}"
        content_length = int(value)

    if content_length is None:
        return "missing Content-Length header"
    return content_length
