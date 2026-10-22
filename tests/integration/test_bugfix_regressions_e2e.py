"""Regression tests for bugs found in code review (subprocess half).

Covers the bugs that only show up with a real child process: ``exit``
handling, lifecycle ordering, stdin from a regular file, the CLI's
``--tool`` mode, child stderr handling, and the Config_File template
flow end to end. The in-process half lives in
``tests/unit/test_bugfix_regressions.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from wispy.framing import FrameDecoder, encode_frame

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"client_name": "test", "client_protocol_version": "0.1.0"},
}
_SHUTDOWN = {"jsonrpc": "2.0", "id": 2, "method": "shutdown"}
_EXIT = {"jsonrpc": "2.0", "method": "exit"}

# A server whose ``shutdown`` handler takes a moment, as a real tool
# flushing state would. ``initialize`` returns the Capabilities
# dataclass itself, which must serialize (bug 2).
_SERVER_SCRIPT = """
import asyncio, sys, time
from wispy import Capabilities, HandlerRegistry, run_stdio

registry = HandlerRegistry()
registry.register("initialize", lambda _p: Capabilities(methods=tuple(registry.methods())))

def shutdown(_params):
    time.sleep(0.3)
    return None

registry.register("shutdown", shutdown)
sys.exit(asyncio.run(run_stdio(registry)))
"""


def _env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    return env


def _frame(msg: dict[str, Any]) -> bytes:
    return encode_frame(json.dumps(msg).encode())


def _responses(stdout: bytes) -> list[dict[str, Any]]:
    return [json.loads(item) for item in FrameDecoder().feed(stdout) if isinstance(item, bytes)]


def _spawn_server() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", _SERVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_env(),
    )


def _send(proc: subprocess.Popen[bytes], *msgs: dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(b"".join(_frame(m) for m in msgs))
    proc.stdin.flush()


def _wait_for_exit(proc: subprocess.Popen[bytes], timeout: float) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(0.05)
    return None


def _finish(proc: subprocess.Popen[bytes]) -> bytes:
    if proc.poll() is None:
        proc.kill()
    out, _err = proc.communicate(timeout=10)
    return out


def _run_cli(argv: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[bytes]:
    cmd = [sys.executable, "-c", "import sys; from wispy.cli.main import main; sys.exit(main(sys.argv[1:]))", *argv]
    return subprocess.run(cmd, check=False, capture_output=True, env=_env(), timeout=timeout)


# --------------------------------------------------------------------- #
# Bugs 2-4: runtime exit handling and lifecycle ordering.
# --------------------------------------------------------------------- #


def test_bug3_exit_takes_effect_while_stdin_stays_open() -> None:
    proc = _spawn_server()
    try:
        _send(proc, _INITIALIZE)
        time.sleep(0.5)
        _send(proc, _SHUTDOWN)
        time.sleep(0.8)
        _send(proc, _EXIT)
        # stdin is deliberately left open.
        status = _wait_for_exit(proc, timeout=5.0)
    finally:
        out = _finish(proc)

    assert status == 0
    # Bug 2: the dataclass ``initialize`` result reached the wire.
    assert [r["id"] for r in _responses(out)] == [1, 2]
    assert _responses(out)[0]["result"]["methods"] == ["initialize", "shutdown"]


def test_bug4_pipelined_shutdown_then_exit_exits_zero() -> None:
    """``exit`` sent before the slow ``shutdown`` completes still sees it."""
    proc = _spawn_server()
    try:
        _send(proc, _INITIALIZE)
        time.sleep(0.5)
        _send(proc, _SHUTDOWN, _EXIT)
        status = _wait_for_exit(proc, timeout=5.0)
    finally:
        out = _finish(proc)

    assert status == 0
    assert [r["id"] for r in _responses(out)] == [1, 2]


# --------------------------------------------------------------------- #
# Bug 10: stdin redirected from a regular file.
# --------------------------------------------------------------------- #


def test_bug10_server_reads_requests_from_a_regular_file(tmp_path: Path) -> None:
    requests = tmp_path / "requests.bin"
    requests.write_bytes(_frame(_INITIALIZE) + _frame(_SHUTDOWN) + _frame(_EXIT))

    with requests.open("rb") as stdin:
        proc = subprocess.run(
            [sys.executable, "-c", _SERVER_SCRIPT],
            stdin=stdin,
            capture_output=True,
            env=_env(),
            timeout=30,
            check=False,
        )

    assert proc.returncode == 0, proc.stderr.decode()
    assert [r["id"] for r in _responses(proc.stdout)] == [1, 2]


# --------------------------------------------------------------------- #
# Bugs 1 and 5: Config_File templates and --tool through the real CLI.
# --------------------------------------------------------------------- #


def _write_template_config(path: Path) -> None:
    # No literal braces: argv elements are templates.
    script = "import json, sys; print(json.dumps(dict(id=sys.argv[1], name='n', python_version='3.12')))"
    path.write_text(
        textwrap.dedent(
            f"""\
            [handlers."environment/get"]
            command = [{json.dumps(sys.executable)}, "-c", {json.dumps(script)}, "{{id}}"]
            """
        )
    )


def test_bug1_config_mode_substitutes_template_keys(tmp_path: Path) -> None:
    config = tmp_path / "tool.toml"
    _write_template_config(config)

    proc = _run_cli(["--config", str(config), "environment/get", "--id", "env-42"])

    assert proc.returncode == 0, proc.stderr.decode()
    assert json.loads(proc.stdout)["id"] == "env-42"


def test_bug5_tool_mode_through_the_parser(tmp_path: Path) -> None:
    config = tmp_path / "tool.toml"
    _write_template_config(config)

    proc = _run_cli(
        ["--tool", sys.executable, "-m", "wispy", "--config", str(config), "environment/get", "--id", "env-7"],
    )

    assert proc.returncode == 0, proc.stderr.decode()
    assert json.loads(proc.stdout)["id"] == "env-7"


# --------------------------------------------------------------------- #
# Bug 8: the child's stderr was piped and never read.
# --------------------------------------------------------------------- #


def test_bug8_child_startup_error_is_shown_to_the_user(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text('[handlers."environment/list"]\ncommand = ["definitely-not-a-real-binary-wispy"]\n')

    proc = _run_cli(["--config", str(config), "environment/list"])

    assert proc.returncode == 1
    stderr = proc.stderr.decode()
    assert "config error" in stderr
    assert "definitely-not-a-real-binary-wispy" in stderr


def test_bug8_chatty_child_stderr_does_not_block(tmp_path: Path) -> None:
    server = tmp_path / "chatty.py"
    server.write_text(
        textwrap.dedent(
            """\
            import asyncio, sys
            from wispy import Capabilities, HandlerRegistry, run_stdio

            registry = HandlerRegistry()
            registry.register("initialize", lambda _p: Capabilities(methods=tuple(registry.methods())))
            registry.register("shutdown", lambda _p: None)

            def list_envs(_params):
                # Far more than a pipe buffer holds.
                sys.stderr.write("x" * 1024 * 1024)
                sys.stderr.flush()
                return []

            registry.register("environment/list", list_envs)
            sys.exit(asyncio.run(run_stdio(registry)))
            """
        )
    )

    started = time.monotonic()
    proc = _run_cli(["--tool", sys.executable, str(server), "environment/list"], timeout=60)

    assert proc.returncode == 0, proc.stderr.decode()[-500:]
    assert json.loads(proc.stdout) == []
    # Without draining, this blocks until the client's 30 s response timeout.
    assert time.monotonic() - started < 20
