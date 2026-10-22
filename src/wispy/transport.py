"""Asyncio stdio transport for the WSP server.

Implements :class:`StdioTransport`, the default transport that reads
Content-Length framed JSON-RPC messages from ``stdin`` and writes them to
``stdout``. All diagnostic logging is routed to ``stderr`` so the stdout
channel remains a pure framing stream.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import TYPE_CHECKING, BinaryIO, TextIO

from wispy.framing import DEFAULT_MAX_CONTENT_LENGTH, DecodeError, FrameDecoder, encode_frame

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["StdioTransport"]

# Most chunks the stdin reader thread may read ahead of the read loop.
# Bounds its memory the way the event loop's pipe flow control does.
_MAX_QUEUED_CHUNKS = 4


class StdioTransport:
    """Async stdio transport for Content-Length framed messages.

    The transport owns a :class:`FrameDecoder` and surfaces decoded payloads
    (or :class:`DecodeError` records for malformed regions) via the
    :meth:`messages` async iterator. Outbound writes go through
    :meth:`write`, which frames the payload and flushes ``stdout`` after each
    response so the client sees responses immediately.

    Constructor arguments allow injection of arbitrary binary file-like
    objects in place of the real ``sys.stdin``/``sys.stdout``/``sys.stderr``
    streams. This is the seam that integration tests use to drive the
    transport over an in-memory pipe.

    ``stdin`` is read through the event loop when it is a pipe, socket
    or character device. Anything else (a regular file redirected with
    ``<``, or a handle the Windows proactor loop cannot watch) is read
    by a daemon thread instead, which reads at most
    ``_MAX_QUEUED_CHUNKS`` chunks ahead of the consumer.

    ``max_frame_size`` bounds the payload size the decoder will accept;
    larger frames are skipped and reported as :class:`DecodeError`.
    """

    def __init__(
        self,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
        stderr: TextIO | None = None,
        *,
        max_frame_size: int | None = DEFAULT_MAX_CONTENT_LENGTH,
    ) -> None:
        self._stdin = stdin if stdin is not None else sys.stdin.buffer
        self._stdout = stdout if stdout is not None else sys.stdout.buffer
        self._stderr = stderr if stderr is not None else sys.stderr
        self._reader: asyncio.StreamReader | None = None
        self._pipe_transport: asyncio.ReadTransport | None = None
        # Fed by the reader thread when stdin cannot be watched by the
        # event loop (see ``_attach``).
        self._chunks: asyncio.Queue[bytes] | None = None
        # Free queue slots; the reader thread takes one per chunk and
        # ``_read_chunk`` returns it once the chunk is consumed.
        self._chunk_slots = threading.Semaphore(_MAX_QUEUED_CHUNKS)
        self._read_chunk_size = 65536
        self._decoder = FrameDecoder(max_content_length=max_frame_size)
        self._stdout_lock = asyncio.Lock()

    async def _attach(self) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        try:
            pipe_transport, _ = await loop.connect_read_pipe(lambda: protocol, self._stdin)
        except (ValueError, OSError, NotImplementedError):
            # ``connect_read_pipe`` only accepts pipes, sockets and
            # character devices. The thread is a daemon so a read
            # blocked on an open stdin never holds up interpreter exit.
            self._chunks = asyncio.Queue()
            threading.Thread(
                target=self._read_blocking,
                args=(loop, self._chunks, self._chunk_slots),
                name="wispy-stdin-reader",
                daemon=True,
            ).start()
            return
        self._reader = reader
        self._pipe_transport = pipe_transport

    def _read_blocking(
        self,
        loop: asyncio.AbstractEventLoop,
        chunks: asyncio.Queue[bytes],
        slots: threading.Semaphore,
    ) -> None:
        """Reader-thread body: forward stdin chunks to ``chunks`` until EOF.

        Waits for a free slot before each read, so a consumer that falls
        behind pauses the thread instead of letting the queue grow.
        """
        read = getattr(self._stdin, "read1", self._stdin.read)
        while True:
            slots.acquire()
            try:
                chunk = read(self._read_chunk_size) or b""
            except Exception:  # noqa: BLE001 - a failed read is reported as EOF
                chunk = b""
            try:
                loop.call_soon_threadsafe(chunks.put_nowait, chunk)
            except RuntimeError:
                # The event loop has closed; nobody is listening.
                return
            if not chunk:
                return

    async def _read_chunk(self) -> bytes:
        """Return the next chunk of stdin, or ``b""`` at EOF."""
        if self._reader is None and self._chunks is None:
            await self._attach()
        if self._reader is not None:
            return await self._reader.read(self._read_chunk_size)
        assert self._chunks is not None  # noqa: S101 - set by _attach
        chunk = await self._chunks.get()
        self._chunk_slots.release()
        return chunk

    async def messages(self) -> AsyncIterator[bytes | DecodeError]:
        """Yield decoded payloads (or :class:`DecodeError`) until stdin EOF.

        The iterator terminates when ``stdin`` reaches EOF, after surfacing
        any final items still buffered inside the decoder. The server uses
        this termination as its signal to begin draining in-flight handler
        tasks.
        """
        while True:
            chunk = await self._read_chunk()
            if not chunk:
                # EOF: surface any final items still in the decoder's
                # buffer (e.g. trailing garbage), then stop.
                for item in self._decoder.feed(b""):
                    yield item
                return
            for item in self._decoder.feed(chunk):
                yield item

    async def write(self, payload: bytes) -> None:
        """Frame ``payload`` and write it to stdout, flushing after each call.

        Concurrent dispatcher tasks may call :meth:`write` simultaneously, so
        the implementation serialises writes behind a lock to guarantee
        frame bytes are not interleaved mid-frame.
        """
        framed = encode_frame(payload)
        async with self._stdout_lock:
            self._stdout.write(framed)
            self._stdout.flush()

    async def drain(self) -> None:
        """Flush any buffered stdout bytes."""
        async with self._stdout_lock:
            self._stdout.flush()

    def close(self) -> None:
        """Stop watching stdin. Safe to call more than once."""
        if self._pipe_transport is not None:
            self._pipe_transport.close()
            self._pipe_transport = None

    def log(self, msg: str) -> None:
        """Write a log line to stderr only.

        A trailing newline is appended if not already present so individual
        log lines stay separated when the host captures stderr line-by-line.
        """
        self._stderr.write(msg)
        if not msg.endswith("\n"):
            self._stderr.write("\n")
        self._stderr.flush()
