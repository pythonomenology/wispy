"""Regression tests: messages are admitted in the state they arrived in.

The TLA+ model ``tla/ServerAdmission.tla`` (property
``ArrivalOrderAdmission``) found that a task's admission ran when the
task first ran, not when its message was read. The read loop takes
already-buffered messages without suspending, so a pipelined ``exit``
(or a lifecycle handler that never suspends) changed the state first,
and the earlier request was rejected with ``server has exited`` or
``server is shutting down``.

Every test writes the whole client sequence into a pipe in one write,
so the read loop sees it as a single stdin chunk.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
from typing import Any

import pytest

from wispy.endpoints import PROTOCOL_VERSION, Capabilities
from wispy.framing import FrameDecoder, encode_frame
from wispy.registry import HandlerRegistry
from wispy.server import run_stdio
from wispy.transport import StdioTransport

_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"client_name": "test", "client_protocol_version": "0.1.0"},
}
_REQ = {"jsonrpc": "2.0", "id": 2, "method": "req"}
_NOTE = {"jsonrpc": "2.0", "method": "note"}
_SHUTDOWN = {"jsonrpc": "2.0", "id": 3, "method": "shutdown"}
_EXIT = {"jsonrpc": "2.0", "method": "exit"}


def _registry(*, async_lifecycle: bool, notes: list[Any]) -> HandlerRegistry:
    """Built-in-shaped handlers; ``async_lifecycle`` makes them never suspend."""
    registry = HandlerRegistry()

    def caps() -> Capabilities:
        return Capabilities(methods=tuple(registry.methods()), protocol_version=PROTOCOL_VERSION)

    if async_lifecycle:

        async def initialize(_params: Any) -> Capabilities:
            return caps()

        async def shutdown(_params: Any) -> None:
            return None

    else:

        def initialize(_params: Any) -> Capabilities:
            return caps()

        def shutdown(_params: Any) -> None:
            return None

    registry.register("initialize", initialize)
    registry.register("shutdown", shutdown)
    registry.register("req", lambda _params: "ok")
    registry.register("note", notes.append)
    return registry


def _run(messages: list[Any], registry: HandlerRegistry) -> tuple[int, list[Any]]:
    """Run the server on ``messages`` and return its status and responses."""
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"".join(encode_frame(json.dumps(m).encode()) for m in messages))
    os.close(write_fd)
    out = io.BytesIO()
    with os.fdopen(read_fd, "rb") as stdin:
        transport = StdioTransport(stdin=stdin, stdout=out, stderr=io.StringIO())
        status = asyncio.run(run_stdio(registry, transport=transport))
    return status, [json.loads(p) for p in FrameDecoder().feed(out.getvalue())]


def _by_id(responses: list[Any]) -> dict[Any, Any]:
    return {r["id"]: r for r in responses if isinstance(r, dict)}


@pytest.mark.parametrize("async_lifecycle", [False, True])
def test_request_pipelined_before_exit_is_handled(*, async_lifecycle: bool) -> None:
    """``initialize, req, exit``: ``req`` was read while initialized."""
    status, responses = _run([_INIT, _REQ, _EXIT], _registry(async_lifecycle=async_lifecycle, notes=[]))
    assert status == 1
    assert _by_id(responses)[2] == {"jsonrpc": "2.0", "id": 2, "result": "ok"}


def test_request_pipelined_before_shutdown_is_handled() -> None:
    """``initialize, req, shutdown, exit`` with a shutdown that never suspends."""
    status, responses = _run([_INIT, _REQ, _SHUTDOWN, _EXIT], _registry(async_lifecycle=True, notes=[]))
    assert status == 0
    by_id = _by_id(responses)
    assert by_id[2] == {"jsonrpc": "2.0", "id": 2, "result": "ok"}
    assert by_id[3] == {"jsonrpc": "2.0", "id": 3, "result": None}


def test_request_pipelined_before_initialize_is_rejected() -> None:
    """``req, initialize``: ``req`` was read before initialize succeeded."""
    _, responses = _run([_REQ, _INIT, _SHUTDOWN, _EXIT], _registry(async_lifecycle=True, notes=[]))
    assert _by_id(responses)[2]["error"]["code"] == -32002


def test_notification_pipelined_before_exit_runs() -> None:
    """``initialize, note, exit``: the notification's handler still runs."""
    notes: list[Any] = []
    _run([_INIT, _NOTE, _EXIT], _registry(async_lifecycle=False, notes=notes))
    assert notes == [None]


def test_batch_pipelined_before_exit_is_handled() -> None:
    """A batch of ordinary requests is admitted as a whole at read time."""
    batch = [{**_REQ, "id": 10}, {**_REQ, "id": 11}]
    _, responses = _run([_INIT, batch, _EXIT], _registry(async_lifecycle=False, notes=[]))
    batch_response = next(r for r in responses if isinstance(r, list))
    assert [r.get("result") for r in batch_response] == ["ok", "ok"]
