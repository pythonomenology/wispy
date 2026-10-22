"""Integration tests for the WSP server lifecycle (shutdown/exit) runtime.

These tests exercise the full ``run_stdio`` pipeline -- transport, framing,
codec, dispatcher, and lifecycle FSM -- by spawning a real Python child
process configured with a tiny driver that registers ``initialize`` and
``shutdown`` handlers and then runs ``run_stdio(registry)``.

Each test sends framed JSON-RPC messages on the child's stdin and reads
framed responses from the child's stdout, then asserts the child's process
exit status. Every ``proc.wait`` is time-bounded so a stuck child fails the
test rather than hanging the suite.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from wispy.framing import DecodeError, FrameDecoder, encode_frame

# Repo root resolved from this file; the driver child needs ``src/`` on
# its PYTHONPATH so it can ``import wispy``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"

# Empty file-descriptor lists for ``select.select``. Hoisted to module
# scope so they carry an explicit element type rather than the
# ambiguous ``list[Any]`` an inline ``[]`` would produce.
_NO_FDS: list[int] = []


# Driver script run by ``python -c``. Registers ``initialize`` and
# ``shutdown`` handlers and drives ``run_stdio`` until the child exits.
#
# The ``initialize`` handler returns the ``to_jsonable()`` dict form of
# :class:`~wispy.endpoints.Capabilities`; the dispatcher accepts either
# that or the dataclass itself.
_DRIVER_SCRIPT = """
import asyncio
import sys

from wispy.endpoints import Capabilities
from wispy.registry import HandlerRegistry
from wispy.server import run_stdio

registry = HandlerRegistry()


def initialize(params):
    return Capabilities(
        methods=tuple(registry.methods()),
        protocol_version="0.1.0",
    ).to_jsonable()


def shutdown(params):
    return None


registry.register("initialize", initialize)
registry.register("shutdown", shutdown)
sys.exit(asyncio.run(run_stdio(registry)))
"""


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _spawn_driver() -> subprocess.Popen[bytes]:
    """Spawn the WSP driver child process with PYTHONPATH=src/."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    return subprocess.Popen(
        [sys.executable, "-c", _DRIVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _send(proc: subprocess.Popen[bytes], message: dict[str, Any]) -> None:
    """Frame ``message`` as JSON and write it to the child's stdin."""
    assert proc.stdin is not None
    payload = json.dumps(message).encode("utf-8")
    proc.stdin.write(encode_frame(payload))
    proc.stdin.flush()


def _read_one_frame(
    proc: subprocess.Popen[bytes],
    decoder: FrameDecoder,
    pending: list[bytes | DecodeError],
    timeout: float = 5.0,
) -> bytes:
    """Read exactly one decoded frame payload from the child's stdout.

    The decoder may produce :class:`DecodeError` records for malformed
    regions; tests in this module expect well-formed traffic from the
    server, so any :class:`DecodeError` raises ``AssertionError``.
    """
    assert proc.stdout is not None
    if pending:
        item = pending.pop(0)
        if isinstance(item, DecodeError):
            msg = f"unexpected decode error: {item}"
            raise AssertionError(msg)
        return item

    deadline = time.monotonic() + timeout
    fd = proc.stdout.fileno()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            msg = f"timed out waiting for response after {timeout}s"
            raise TimeoutError(msg)
        rlist, _, _ = select.select([fd], _NO_FDS, _NO_FDS, remaining)
        if not rlist:
            msg = f"timed out waiting for response after {timeout}s"
            raise TimeoutError(msg)
        chunk = os.read(fd, 65536)
        if not chunk:
            msg = "child closed stdout before producing a frame"
            raise EOFError(msg)
        pending.extend(decoder.feed(chunk))
        if pending:
            item = pending.pop(0)
            if isinstance(item, DecodeError):
                msg = f"unexpected decode error: {item}"
                raise AssertionError(msg)
            return item


def _close_stdin(proc: subprocess.Popen[bytes]) -> None:
    """Close the child's stdin to signal EOF."""
    assert proc.stdin is not None
    # Child may have already exited; that's fine.
    with contextlib.suppress(BrokenPipeError):
        proc.stdin.close()


def _wait_with_timeout(proc: subprocess.Popen[bytes], timeout: float = 5.0) -> int:
    """Wait for ``proc`` to exit within ``timeout`` seconds, or fail."""
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait(timeout=2.0)
        msg = f"child did not exit within {timeout}s"
        raise AssertionError(msg) from exc


def _drain_stderr(proc: subprocess.Popen[bytes]) -> str:
    """Best-effort drain of the child's stderr for diagnostic context."""
    assert proc.stderr is not None
    try:
        return proc.stderr.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best-effort drain may fail in many ways
        return ""


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_initialize_shutdown_exit_returns_zero() -> None:
    """initialize -> shutdown -> exit (notification) yields exit status 0."""
    proc = _spawn_driver()
    try:
        decoder = FrameDecoder()
        pending: list[bytes | DecodeError] = []

        # initialize
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1,
                "params": {
                    "client_name": "lifecycle-test",
                    "client_protocol_version": "0.1.0",
                },
            },
        )
        init_resp = json.loads(_read_one_frame(proc, decoder, pending))
        assert init_resp["jsonrpc"] == "2.0"
        assert init_resp["id"] == 1
        assert "result" in init_resp, init_resp
        assert init_resp["result"]["protocol_version"] == "0.1.0"

        # shutdown
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "shutdown",
                "id": 2,
            },
        )
        shutdown_resp = json.loads(_read_one_frame(proc, decoder, pending))
        assert shutdown_resp["jsonrpc"] == "2.0"
        assert shutdown_resp["id"] == 2
        assert shutdown_resp.get("result", "absent") is None

        # exit notification (no id) and EOF
        _send(proc, {"jsonrpc": "2.0", "method": "exit"})
        _close_stdin(proc)

        status = _wait_with_timeout(proc, timeout=5.0)
        stderr = _drain_stderr(proc)
        assert status == 0, f"expected exit 0 after shutdown+exit, got {status}; stderr={stderr!r}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)


def test_exit_without_shutdown_returns_one() -> None:
    """initialize -> exit (no shutdown) yields exit status 1."""
    proc = _spawn_driver()
    try:
        decoder = FrameDecoder()
        pending: list[bytes | DecodeError] = []

        # initialize
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1,
                "params": {
                    "client_name": "lifecycle-test",
                    "client_protocol_version": "0.1.0",
                },
            },
        )
        init_resp = json.loads(_read_one_frame(proc, decoder, pending))
        assert init_resp["id"] == 1
        assert "result" in init_resp, init_resp

        # exit notification with no preceding shutdown -> status 1
        _send(proc, {"jsonrpc": "2.0", "method": "exit"})
        _close_stdin(proc)

        status = _wait_with_timeout(proc, timeout=5.0)
        stderr = _drain_stderr(proc)
        assert status == 1, f"expected exit 1 for exit without shutdown, got {status}; stderr={stderr!r}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)


def test_method_before_initialize_returns_minus_32002() -> None:
    """A non-``initialize`` method before initialize yields ``-32002``.

    The lifecycle FSM rejects every method except ``initialize`` from the
    UNINITIALIZED state with JSON-RPC error code ``-32002`` (server not
    initialized). After receiving the error, closing stdin should let the
    child exit cleanly via the EOF drain path with status 0.
    """
    proc = _spawn_driver()
    try:
        decoder = FrameDecoder()
        pending: list[bytes | DecodeError] = []

        # Send any non-initialize method as a request (with id) so an
        # error response is produced.
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "method": "environment/list",
                "id": 1,
                "params": {},
            },
        )
        resp = json.loads(_read_one_frame(proc, decoder, pending))
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "error" in resp, resp
        assert resp["error"]["code"] == -32002, resp["error"]

        # EOF drain path: no shutdown/exit, just close stdin.
        _close_stdin(proc)

        status = _wait_with_timeout(proc, timeout=5.0)
        stderr = _drain_stderr(proc)
        assert status == 0, f"expected exit 0 from EOF drain, got {status}; stderr={stderr!r}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)
