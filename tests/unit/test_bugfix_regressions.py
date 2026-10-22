"""Regression tests for bugs found in code review (in-process half).

Each test names the bug it pins down. The subprocess-driven half (exit
handling, lifecycle ordering, stdin from a file, CLI ``--tool`` and
child stderr) lives in ``tests/integration/test_bugfix_regressions_e2e.py``.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any

import pytest

from wispy.cli.fallback import make_fallback_registry
from wispy.cli.main import build_parser, main
from wispy.config import (
    ConfigEntry,
    ResultMode,
    SubprocessHandlerSpec,
    _spawn_and_capture,
    load_config,
    make_handler,
)
from wispy.dispatcher import _invoke_handler, dispatch
from wispy.endpoints import (
    Capabilities,
    CreateEnvironmentParams,
    Environment,
    ExecuteParams,
    GetEnvironmentParams,
)
from wispy.errors import JsonRpcErrorCode, WspError, WspErrorCode
from wispy.framing import DecodeError, FrameDecoder, encode_frame
from wispy.lifecycle import LifecycleManager
from wispy.protocol import JsonRpcResponse, parse_message, serialize_response
from wispy.registry import HandlerRegistry
from wispy.server import _emit

# Prints its argv (minus the script name) as a JSON array.
_ECHO_ARGV = [sys.executable, "-c", "import json, sys; print(json.dumps(sys.argv[1:]))"]


def _request(method: str, params: Any, request_id: int = 1) -> bytes:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg).encode()


async def _dispatch_initialized(registry: HandlerRegistry, method: str, params: Any) -> Any:
    lifecycle = LifecycleManager()
    lifecycle.on_initialize_success(Capabilities())
    return await dispatch(
        parse_message(_request(method, params)),
        registry=registry,
        lifecycle=lifecycle,
        log=lambda _msg: None,
    )


# --------------------------------------------------------------------- #
# Bug 1: Config_File command handlers received params dataclasses.
# --------------------------------------------------------------------- #


def test_bug1_subprocess_handler_substitutes_keys_through_dispatcher() -> None:
    """``{id}`` resolves for a validated ``environment/get`` request."""
    registry = HandlerRegistry()
    spec = SubprocessHandlerSpec((*_ECHO_ARGV, "get", "{id}"), ResultMode.JSON)
    registry.register("environment/get", make_handler(ConfigEntry("environment/get", spec)))

    response = asyncio.run(_dispatch_initialized(registry, "environment/get", {"id": "env-1"}))

    assert response == JsonRpcResponse(id=1, result=["get", "env-1"])


def test_bug1_subprocess_handler_splats_argv_through_dispatcher() -> None:
    """``{argv}`` splats the validated ``environment/execute`` argv tuple."""
    registry = HandlerRegistry()
    spec = SubprocessHandlerSpec((*_ECHO_ARGV, "{id}", "{argv}"), ResultMode.EXEC)
    registry.register("environment/execute", make_handler(ConfigEntry("environment/execute", spec)))

    response = asyncio.run(
        _dispatch_initialized(registry, "environment/execute", {"id": "env-1", "argv": ["python", "-V"]}),
    )

    assert isinstance(response, JsonRpcResponse)
    assert response.result["exit_code"] == 0
    assert json.loads(response.result["stdout"]) == ["env-1", "python", "-V"]


def test_bug1_unset_optional_param_reports_missing() -> None:
    """An unset optional field (``cwd``) is "missing", not the string "None"."""
    handler = make_handler(
        ConfigEntry("environment/execute", SubprocessHandlerSpec((*_ECHO_ARGV, "{cwd}"), ResultMode.EXEC)),
    )
    with pytest.raises(WspError) as excinfo:
        asyncio.run(handler(ExecuteParams(id="e", argv=("x",))))
    assert excinfo.value.data["reason"] == "missing-template-key"


def test_bug1_template_result_renders_from_dataclass_params() -> None:
    spec = SubprocessHandlerSpec(
        (sys.executable, "-c", "pass"),
        ResultMode.TEMPLATE,
        result_template={"id": "{name}", "python_version": "{python_version}"},
    )
    handler = make_handler(ConfigEntry("environment/create", spec))
    result = asyncio.run(handler(CreateEnvironmentParams(name="scratch", python_version="3.12")))
    assert result == {"id": "scratch", "python_version": "3.12"}


# --------------------------------------------------------------------- #
# Bug 2: dataclass results were unserializable and dropped silently.
# --------------------------------------------------------------------- #


def test_bug2_wire_models_serialize_via_to_jsonable() -> None:
    env = Environment(id="e", name="n", python_version="3.12")
    payload = serialize_response(JsonRpcResponse(id=1, result=[env, Capabilities(methods=("initialize",))]))
    assert json.loads(payload)["result"] == [env.to_jsonable(), Capabilities(methods=("initialize",)).to_jsonable()]


def test_bug2_initialize_dict_result_caches_capabilities_without_warning() -> None:
    caps = Capabilities(methods=("initialize",))
    registry = HandlerRegistry()
    registry.register("initialize", lambda _p: caps.to_jsonable())
    lifecycle = LifecycleManager()
    logs: list[str] = []

    asyncio.run(
        dispatch(
            parse_message(_request("initialize", {"client_name": "t", "client_protocol_version": "0.1.0"})),
            registry=registry,
            lifecycle=lifecycle,
            log=logs.append,
        ),
    )

    assert lifecycle.capabilities == caps
    assert logs == []


class _RecordingTransport:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []
        self.logs: list[str] = []

    async def write(self, payload: bytes) -> None:
        self.payloads.append(payload)

    def log(self, msg: str) -> None:
        self.logs.append(msg)


def test_bug2_unserializable_result_becomes_internal_error() -> None:
    transport = _RecordingTransport()
    asyncio.run(_emit(JsonRpcResponse(id=7, result=object()), transport))  # type: ignore[arg-type]

    (payload,) = transport.payloads
    wire = json.loads(payload)
    assert wire["id"] == 7
    assert wire["error"]["code"] == int(JsonRpcErrorCode.INTERNAL_ERROR)
    assert transport.logs


def test_bug2_unserializable_batch_entry_is_replaced_individually() -> None:
    transport = _RecordingTransport()
    batch = [JsonRpcResponse(id=1, result="ok"), JsonRpcResponse(id=2, result=float("nan"))]
    asyncio.run(_emit(batch, transport))  # type: ignore[arg-type]

    wire = json.loads(transport.payloads[0])
    assert wire[0] == {"jsonrpc": "2.0", "result": "ok", "id": 1}
    assert wire[1]["error"]["code"] == int(JsonRpcErrorCode.INTERNAL_ERROR)


# --------------------------------------------------------------------- #
# Bugs 5-7: CLI argument parsing.
# --------------------------------------------------------------------- #


def test_bug5_tool_argv_is_split_at_the_method(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_run_with_client(argv: list[str], method: str, params: dict[str, Any]) -> int:
        captured.update(argv=argv, method=method, params=params)
        return 0

    monkeypatch.setattr("wispy.cli.main._run_with_client", _fake_run_with_client)

    code = main(["--tool", "mytool", "serve", "--verbose", "environment/get", "--id", "env-1"])

    assert code == 0
    assert captured == {"argv": ["mytool", "serve", "--verbose"], "method": "environment/get", "params": {"id": "env-1"}}


def test_bug5_tool_without_server_argv_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--tool", "environment/list"])
    assert excinfo.value.code == 2


def test_bug6_top_level_params_json_survives_the_subcommand() -> None:
    args = build_parser().parse_args(["--params-json", "-", "environment/create"])
    assert args.params_json == "-"


def test_bug7_argv_and_env_flags_produce_list_and_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_fallback(method: str, params: dict[str, Any]) -> int:
        captured.update(method=method, params=params)
        return 0

    monkeypatch.setattr("wispy.cli.main._run_fallback_inprocess", _fake_fallback)

    main(["environment/execute", "--id", "e", "--env", "A=1", "--env", "B=x=y", "--argv", "python", "-c", "pass"])

    assert captured["params"] == {"id": "e", "env": {"A": "1", "B": "x=y"}, "argv": ["python", "-c", "pass"]}


def test_bug7_malformed_env_flag_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["environment/execute", "--env", "NOEQUALS"])
    assert excinfo.value.code == 2


# --------------------------------------------------------------------- #
# Bug 9: fallback create ignored the requested Python version.
# --------------------------------------------------------------------- #


def test_bug9_fallback_create_builds_venv_on_requested_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("wispy.cli.fallback.shutil.which", lambda name: f"/opt/py/bin/{name}")
    calls: list[list[str]] = []

    def _fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr("wispy.cli.fallback.subprocess.run", _fake_run)
    registry = make_fallback_registry(tmp_path)

    handler = registry.lookup("environment/create")
    assert handler is not None
    created = handler(CreateEnvironmentParams(name="old", python_version="3.9"))

    assert calls == [["/opt/py/bin/python3.9", "-m", "venv", str(tmp_path / "envs" / created["id"])]]


def test_bug9_failed_venv_command_is_execution_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wispy.cli.fallback.shutil.which", lambda name: f"/opt/py/bin/{name}")
    monkeypatch.setattr(
        "wispy.cli.fallback.subprocess.run",
        lambda argv, **_kw: subprocess.CompletedProcess(argv, 1, b"", b"no ensurepip"),
    )
    handler = make_fallback_registry(tmp_path).lookup("environment/create")
    assert handler is not None

    with pytest.raises(WspError) as excinfo:
        handler(CreateEnvironmentParams(name="x", python_version="3.9"))

    assert excinfo.value.data == {"reason": "venv-create-failed"}
    assert "no ensurepip" in excinfo.value.message


# --------------------------------------------------------------------- #
# Bug 11: subprocess handler orphaned children; timeout not configurable.
# --------------------------------------------------------------------- #


def test_bug11_cancelled_subprocess_handler_kills_child(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    script = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"

    async def _run_then_cancel() -> int:
        task = asyncio.create_task(_spawn_and_capture([sys.executable, "-c", script]))
        for _ in range(200):
            if pid_file.exists() and pid_file.read_text():
                break
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return int(pid_file.read_text())

    pid = asyncio.run(_run_then_cancel())

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.05)
    assert not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    # ``ps`` rather than ``os.kill(pid, 0)``: an unreaped zombie still
    # "exists" for kill(), but ps reports it as state Z.
    proc = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], check=False, capture_output=True, text=True)
    state = proc.stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_bug11_timeout_is_configurable(tmp_path: Path) -> None:
    config = tmp_path / "c.toml"
    config.write_text(
        textwrap.dedent(
            f"""\
            [handlers."environment/list"]
            command = [{json.dumps(sys.executable)}, "-c", "import time; time.sleep(30)"]
            timeout = 0.5
            """
        )
    )
    (entry,) = load_config(config)
    assert isinstance(entry.spec, SubprocessHandlerSpec)
    assert entry.spec.timeout == 0.5

    started = time.monotonic()
    with pytest.raises(WspError) as excinfo:
        asyncio.run(make_handler(entry)(None))
    assert time.monotonic() - started < 10
    assert excinfo.value.data["reason"] == "timeout"


@pytest.mark.parametrize("bad", ["0", "-1", "true", '"30"'])
def test_bug11_invalid_timeout_is_a_config_error(tmp_path: Path, bad: str) -> None:
    from wispy.config import ConfigError

    config = tmp_path / "c.toml"
    config.write_text(f'[handlers."environment/list"]\ncommand = ["echo"]\ntimeout = {bad}\n')
    with pytest.raises(ConfigError, match="timeout"):
        load_config(config)


# --------------------------------------------------------------------- #
# Bug 12: asyncio.iscoroutinefunction deprecation warning (3.14).
# --------------------------------------------------------------------- #


def test_bug12_invoking_handlers_emits_no_deprecation_warning() -> None:
    async def _async_handler(params: Any) -> Any:
        return params

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert asyncio.run(_invoke_handler(lambda p: p, 1)) == 1
        assert asyncio.run(_invoke_handler(_async_handler, 2)) == 2


# --------------------------------------------------------------------- #
# Bug 13: unbounded frame buffering.
# --------------------------------------------------------------------- #


def test_bug13_oversized_frame_is_skipped_without_buffering() -> None:
    decoder = FrameDecoder(max_content_length=10)
    items = list(decoder.feed(b"Content-Length: 1000\r\n\r\n" + b"x" * 400))

    assert len(items) == 1
    assert isinstance(items[0], DecodeError)
    assert "exceeds" in items[0].reason
    assert len(decoder._buffer) == 0  # noqa: SLF001 - asserting the memory bound

    # The rest of the oversized payload is dropped; the next frame decodes.
    items = list(decoder.feed(b"x" * 600 + encode_frame(b"{}")))
    assert items == [b"{}"]


def test_bug13_header_garbage_is_bounded() -> None:
    decoder = FrameDecoder()
    items = list(decoder.feed(b"g" * 100_000))

    assert len(items) == 1
    assert isinstance(items[0], DecodeError)
    assert len(decoder._buffer) < 16  # noqa: SLF001 - asserting the memory bound


# --------------------------------------------------------------------- #
# Bug 14: fallback error reporting.
# --------------------------------------------------------------------- #


def test_bug14_execute_timeout_is_reported_as_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wispy.cli.fallback.shutil.which", lambda name: f"/opt/py/bin/{name}")
    registry = make_fallback_registry(tmp_path)
    monkeypatch.setattr(
        "wispy.cli.fallback.subprocess.run",
        lambda argv, **_kw: subprocess.CompletedProcess(argv, 0, b"", b""),
    )
    create = registry.lookup("environment/create")
    assert create is not None
    env_id = create(CreateEnvironmentParams(name="x", python_version="3.9"))["id"]

    def _timeout(argv: list[str], **kw: Any) -> Any:
        raise subprocess.TimeoutExpired(argv, kw.get("timeout", 0))

    monkeypatch.setattr("wispy.cli.fallback.subprocess.run", _timeout)
    execute = registry.lookup("environment/execute")
    assert execute is not None

    with pytest.raises(WspError) as excinfo:
        execute(ExecuteParams(id=env_id, argv=("sleep", "1000")))
    assert excinfo.value.data == {"reason": "timeout"}


@pytest.mark.parametrize("content", ["not json", '{"an": "object"}', '[{"id": 1}]'])
def test_bug14_corrupt_index_is_execution_failed(tmp_path: Path, content: str) -> None:
    registry = make_fallback_registry(tmp_path)
    (tmp_path / "index.json").write_text(content)
    get = registry.lookup("environment/get")
    assert get is not None

    with pytest.raises(WspError) as excinfo:
        get(GetEnvironmentParams(id="env-1"))

    assert excinfo.value.code == int(WspErrorCode.EXECUTION_FAILED)
    assert excinfo.value.data["reason"] == "index-corrupt"
