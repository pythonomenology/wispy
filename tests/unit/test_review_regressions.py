"""Regression tests for the second code review of the bugfix changes.

Each section names the finding it pins down: a crash or hang the
review found in the earlier fixes, and the behaviour that replaced it.
"""

from __future__ import annotations

import asyncio
import errno
import gc
import io
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

from wispy.cli import fallback as fb
from wispy.cli import main as cli_main
from wispy.cli._exit_codes import ExitCode
from wispy.cli.client import WspClient
from wispy.cli.state import FallbackState, _retry_until_locked
from wispy.config import _spawn_and_capture, configured_timeout
from wispy.endpoints import (
    Capabilities,
    CreateEnvironmentParams,
    DeleteEnvironmentParams,
    Environment,
    ExecuteParams,
    GetEnvironmentParams,
)
from wispy.errors import JsonRpcErrorCode, WspError, WspErrorCode
from wispy.framing import encode_frame
from wispy.protocol import ParseFailure, ParseFailureKind, parse_message
from wispy.registry import HandlerRegistry
from wispy.server import run_stdio
from wispy.transport import _MAX_QUEUED_CHUNKS, StdioTransport

_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX flock and sh scripts")


def _frame(obj: Any) -> bytes:
    return encode_frame(json.dumps(obj).encode())


def _init_frame() -> bytes:
    return _frame(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"client_name": "t", "client_protocol_version": "0.1.0"},
        }
    )


def _decode_responses(raw: bytes) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while raw:
        header, _, rest = raw.partition(b"\r\n\r\n")
        length = int(header.split(b":")[1])
        out.append(json.loads(rest[:length]))
        raw = rest[length:]
    return out


def _run_server(tmp_path: Path, stdin_bytes: bytes, registry: HandlerRegistry) -> tuple[int, list[dict[str, Any]], str]:
    """Run ``run_stdio`` over a regular-file stdin; return status, responses, stderr."""
    stdin_path = tmp_path / "stdin.bin"
    stdin_path.write_bytes(stdin_bytes)
    out = io.BytesIO()
    err = io.StringIO()
    with open(stdin_path, "rb") as stdin:
        transport = StdioTransport(stdin=stdin, stdout=out, stderr=err)
        status = asyncio.run(run_stdio(registry, transport=transport))
    return status, _decode_responses(out.getvalue()), err.getvalue()


def _ping_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.register("initialize", lambda _p: Capabilities())
    registry.register("ping", lambda _p: "pong")
    return registry


# --------------------------------------------------------------------- #
# A deeply nested JSON frame stopped the server's read loop.
# --------------------------------------------------------------------- #


def test_deeply_nested_json_is_a_parse_error() -> None:
    parsed = parse_message(b"[" * 100_000)
    assert isinstance(parsed, ParseFailure)
    assert parsed.kind is ParseFailureKind.PARSE_ERROR


def test_server_keeps_reading_after_a_deeply_nested_frame(tmp_path: Path) -> None:
    stdin = _init_frame() + encode_frame(b"[" * 100_000) + _frame({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    status, responses, _ = _run_server(tmp_path, stdin, _ping_registry())
    assert status == 0
    assert responses[1]["error"]["code"] == int(JsonRpcErrorCode.PARSE_ERROR)
    assert responses[2] == {"jsonrpc": "2.0", "result": "pong", "id": 2}


def test_server_survives_an_unexpected_parse_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_parse = parse_message
    calls: list[bytes] = []

    def parse_raising_once(raw: bytes) -> Any:
        calls.append(raw)
        if len(calls) == 2:
            msg = "boom"
            raise RuntimeError(msg)
        return real_parse(raw)

    monkeypatch.setattr("wispy.server.parse_message", parse_raising_once)
    stdin = _init_frame() + _frame({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    stdin += _frame({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    _, responses, stderr = _run_server(tmp_path, stdin, _ping_registry())
    assert [r["id"] for r in responses] == [1, 3]
    assert "unexpected error parsing message" in stderr


# --------------------------------------------------------------------- #
# Windows lock waits gave up after ten seconds.
# --------------------------------------------------------------------- #


def test_lock_retry_waits_through_contention() -> None:
    attempts: list[int] = []

    def try_lock() -> None:
        attempts.append(1)
        if len(attempts) < 25:
            raise OSError(errno.EACCES, "locked")

    _retry_until_locked(try_lock, poll=0)
    assert len(attempts) == 25


def test_lock_retry_propagates_other_errors() -> None:
    def try_lock() -> None:
        raise OSError(errno.EBADF, "bad file descriptor")

    with pytest.raises(OSError) as excinfo:
        _retry_until_locked(try_lock, poll=0)
    assert excinfo.value.errno == errno.EBADF


def test_env_lock_path_stays_inside_locks_dir(tmp_path: Path) -> None:
    state = FallbackState(tmp_path)
    for env_id in ("env-0123456789ab", "../../escape", "a/b\\c", "\ud800"):
        assert state.env_lock_path(env_id).parent == state.env_locks_dir


# --------------------------------------------------------------------- #
# Config_File ``timeout`` above 30 s was cut off by the CLI client.
# --------------------------------------------------------------------- #


def test_configured_timeout_reads_only_the_named_entry(tmp_path: Path) -> None:
    config = tmp_path / "c.toml"
    config.write_text(
        '[handlers."environment/create"]\n'
        'command = ["true"]\n'
        "timeout = 120\n"
        '[handlers."environment/list"]\n'
        'import = "no_such_module_wispy_test:handler"\n'
    )
    assert configured_timeout(config, "environment/create") == 120.0
    # No timeout, an ``import`` entry the CLI must not import, or no entry.
    assert configured_timeout(config, "environment/list") is None
    assert configured_timeout(config, "environment/get") is None
    assert "no_such_module_wispy_test" not in sys.modules


@pytest.mark.parametrize("content", ["not toml [", '[handlers."environment/create"]\ntimeout = -1\n'])
def test_configured_timeout_is_none_for_bad_files(tmp_path: Path, content: str) -> None:
    config = tmp_path / "c.toml"
    config.write_text(content)
    assert configured_timeout(config, "environment/create") is None


def test_config_mode_waits_for_the_entry_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "c.toml"
    config.write_text('[handlers."environment/list"]\ncommand = ["true"]\nresult = "none"\ntimeout = 120\n')
    seen: dict[str, Any] = {}

    async def fake_run_with_client(argv: list[str], method: str, params: dict[str, Any], **kwargs: Any) -> int:
        seen.update(kwargs)
        return ExitCode.SUCCESS

    monkeypatch.setattr("wispy.cli.main._run_with_client", fake_run_with_client)
    assert cli_main.main(["--config", str(config), "environment/list"]) == ExitCode.SUCCESS
    assert seen["response_timeout"] == 125.0


_SLOW_LIST_SERVER = """
import asyncio, sys, time
from wispy.endpoints import Capabilities
from wispy.registry import HandlerRegistry
from wispy.server import run_stdio

registry = HandlerRegistry()
registry.register("initialize", lambda _p: Capabilities())
registry.register("shutdown", lambda _p: None)

def slow_list(_p):
    time.sleep(1.0)
    return []

registry.register("environment/list", slow_list)
sys.exit(asyncio.run(run_stdio(registry)))
"""


@pytest.mark.parametrize(("response_timeout", "expected"), [(0.3, ExitCode.GENERIC_ERROR), (10.0, ExitCode.SUCCESS)])
def test_client_response_timeout_applies_to_the_method(
    response_timeout: float,
    expected: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run() -> int:
        async with WspClient([sys.executable, "-c", _SLOW_LIST_SERVER], response_timeout=response_timeout) as client:
            result = await client.call("environment/list", None)
        return result.exit_code

    assert asyncio.run(run()) == expected
    if expected == ExitCode.GENERIC_ERROR:
        assert "response timeout after 0.3s for environment/list" in capsys.readouterr().err


# --------------------------------------------------------------------- #
# The client capped responses at 64 MiB and timed out on larger ones.
# --------------------------------------------------------------------- #


def test_client_accepts_responses_of_any_size() -> None:
    client = WspClient(["unused"])
    assert client._decoder._max_content_length is None


# --------------------------------------------------------------------- #
# A child that exited before reading made the client raise BrokenPipe.
# --------------------------------------------------------------------- #


def test_child_that_exits_before_reading_has_its_stderr_shown(capsys: pytest.CaptureFixture[str]) -> None:
    script = "import sys; sys.stderr.write('mytool: unknown flag --bad\\n'); sys.exit(2)"

    async def run() -> int:
        async with WspClient([sys.executable, "-c", script]) as client:
            assert client.proc is not None
            await client.proc.wait()
            result = await client.call("environment/list", None)
        return result.exit_code

    assert asyncio.run(run()) == ExitCode.GENERIC_ERROR
    err = capsys.readouterr().err
    assert "server exited before reading initialize" in err
    assert "mytool: unknown flag --bad" in err


# --------------------------------------------------------------------- #
# Fallback: silent sweep failures, nested use of an env, double reads.
# --------------------------------------------------------------------- #


def _fake_venv(_interpreter: Any, target: Any) -> None:
    bin_dir = Path(target) / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    (bin_dir / ("python.exe" if os.name == "nt" else "python")).write_text("")


@pytest.fixture
def fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[HandlerRegistry, str, FallbackState]:
    monkeypatch.delenv("WISPY_FALLBACK_ENVS_IN_USE", raising=False)
    monkeypatch.setattr("wispy.cli.fallback.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("wispy.cli.fallback._create_venv", _fake_venv)
    registry = fb.make_fallback_registry(tmp_path / "state")
    created = registry.lookup("environment/create")(CreateEnvironmentParams(name="demo", python_version="3.12"))
    return registry, created["id"], FallbackState(tmp_path / "state")


def _call(registry: HandlerRegistry, method: str, params: Any) -> Any:
    handler = registry.lookup(method)
    assert handler is not None
    return handler(params)


def test_failed_sweep_is_reported_on_stderr(
    fallback: tuple[HandlerRegistry, str, FallbackState],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry, env_id, state = fallback

    def failing_rmtree(_path: Any, *_args: Any, **_kwargs: Any) -> None:
        msg = "directory busy"
        raise OSError(msg)

    monkeypatch.setattr("wispy.cli.fallback.shutil.rmtree", failing_rmtree)
    assert _call(registry, "environment/delete", DeleteEnvironmentParams(id=env_id)) == {"id": env_id}
    err = capsys.readouterr().err
    assert "could not remove leftover environment directory" in err
    assert env_id in err
    assert (state.envs_dir / env_id).is_dir()


def test_sweep_removes_stray_files(fallback: tuple[HandlerRegistry, str, FallbackState]) -> None:
    registry, _, state = fallback
    stray = state.envs_dir / "stray.txt"
    stray.write_text("x")
    _call(registry, "environment/create", CreateEnvironmentParams(name="other", python_version="3.12"))
    assert not stray.exists()


def test_execute_marks_its_env_in_use_for_the_command(
    fallback: tuple[HandlerRegistry, str, FallbackState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, env_id, _ = fallback
    seen_env: dict[str, str] = {}

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen_env.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setenv("WISPY_FALLBACK_ENVS_IN_USE", "env-outer000000")
    monkeypatch.setattr("wispy.cli.fallback.subprocess.run", fake_run)
    _call(registry, "environment/execute", ExecuteParams(id=env_id, argv=("tool",)))
    assert seen_env["WISPY_FALLBACK_ENVS_IN_USE"].split(os.pathsep) == ["env-outer000000", env_id]


@_POSIX_ONLY
def test_nested_delete_of_the_running_env_fails_instead_of_deadlocking(
    fallback: tuple[HandlerRegistry, str, FallbackState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, env_id, state = fallback
    monkeypatch.setenv("WISPY_FALLBACK_ENVS_IN_USE", env_id)
    # The ancestor execute holds the env lock while its command runs.
    with state.env_lock(env_id, shared=True):
        started = time.monotonic()
        with pytest.raises(WspError) as excinfo:
            _call(registry, "environment/delete", DeleteEnvironmentParams(id=env_id))
        assert time.monotonic() - started < 5
    assert excinfo.value.code == int(WspErrorCode.EXECUTION_FAILED)
    assert excinfo.value.data == {"reason": "env-in-use", "id": env_id}
    assert [e["id"] for e in _call(registry, "environment/list", None)] == [env_id]


@_POSIX_ONLY
def test_nested_get_does_not_wait_on_its_ancestors_lock(
    fallback: tuple[HandlerRegistry, str, FallbackState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, env_id, state = fallback
    monkeypatch.setenv("WISPY_FALLBACK_ENVS_IN_USE", env_id)
    results: list[Any] = []
    # Exclusive, as a Windows ancestor's "shared" lock is.
    with state.env_lock(env_id, shared=False):
        worker = threading.Thread(
            target=lambda: results.append(_call(registry, "environment/get", GetEnvironmentParams(id=env_id))),
        )
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
    assert results[0]["id"] == env_id


def test_get_reads_the_index_once(
    fallback: tuple[HandlerRegistry, str, FallbackState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, env_id, _ = fallback
    reads: list[int] = []
    real_read = FallbackState.read_index

    def counting_read(self: FallbackState) -> list[Environment]:
        reads.append(1)
        return real_read(self)

    monkeypatch.setattr(FallbackState, "read_index", counting_read)
    _call(registry, "environment/get", GetEnvironmentParams(id=env_id))
    assert len(reads) == 1


def test_unknown_id_with_path_separators_is_not_found(fallback: tuple[HandlerRegistry, str, FallbackState]) -> None:
    registry, _, state = fallback
    with pytest.raises(WspError) as excinfo:
        _call(registry, "environment/get", GetEnvironmentParams(id="../../outside"))
    assert excinfo.value.code == int(WspErrorCode.ENVIRONMENT_NOT_FOUND)
    assert not (state.root.parent / "outside.lock").exists()


# --------------------------------------------------------------------- #
# A cancelled subprocess handler leaked its transport.
# --------------------------------------------------------------------- #


def test_cancelled_subprocess_handler_closes_its_transport(tmp_path: Path) -> None:
    started = tmp_path / "started"
    script = f"import time; open({str(started)!r}, 'w').close(); time.sleep(60)"

    async def run_then_cancel() -> None:
        task = asyncio.create_task(_spawn_and_capture([sys.executable, "-c", script]))
        for _ in range(200):
            if started.exists():
                break
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(run_then_cancel())
        gc.collect()
    assert not [w for w in caught if "unclosed transport" in str(w.message)]


# --------------------------------------------------------------------- #
# The stdin reader thread queued chunks without bound.
# --------------------------------------------------------------------- #


def test_reader_thread_reads_a_bounded_distance_ahead(tmp_path: Path) -> None:
    stdin_path = tmp_path / "stdin.bin"
    stdin_path.write_bytes(b"x" * (64 * 1024 * 64))

    async def run() -> int:
        with open(stdin_path, "rb") as stdin:
            transport = StdioTransport(stdin=stdin, stdout=io.BytesIO(), stderr=io.StringIO())
            await transport._read_chunk()
            await asyncio.sleep(0.3)
            assert transport._chunks is not None
            return transport._chunks.qsize()

    assert asyncio.run(run()) <= _MAX_QUEUED_CHUNKS


# --------------------------------------------------------------------- #
# The in-process CLI could not print wire models.
# --------------------------------------------------------------------- #


def test_inprocess_cli_prints_wire_models(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    registry = HandlerRegistry()
    registry.register("environment/list", lambda _p: [Environment(id="env-1", name="a", python_version="3.12")])
    monkeypatch.setattr("wispy.cli.main.make_fallback_registry", lambda: registry)
    assert cli_main.main(["environment/list"]) == ExitCode.SUCCESS
    assert json.loads(capsys.readouterr().out) == [{"id": "env-1", "name": "a", "python_version": "3.12"}]
