"""Regression tests for the fallback's per-environment lock and sweep.

Each test forces an interleaving or failure that the TLA+ model
``tla/FallbackStore.tla`` found against the earlier handlers:

* ``ExecuteOnComplete`` / ``GetSeesInterpreter``: a delete ran between
  ``environment/execute`` (or ``get``) reading the index and using the
  venv, so the command ran on the caller's PATH instead.
* ``IndexedComplete`` / ``NoOrphanDir``: an interrupted delete or create
  left a listed env without a venv, or an unlisted directory forever.

The handlers are called directly, as in ``test_fallback_edge_cases``.
A thread stands in for the second CLI process; ``flock`` excludes it
the same way because every lock call opens the lock file anew.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from wispy.cli import fallback as fb
from wispy.cli.state import FallbackState
from wispy.endpoints import (
    CreateEnvironmentParams,
    DeleteEnvironmentParams,
    ExecuteParams,
    GetEnvironmentParams,
)
from wispy.errors import WspError, WspErrorCode

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX flock and sh scripts")

# Long enough for a delete that is not blocked to finish.
_BLOCKED_S = 0.5


def _script(path: Path, text: str) -> None:
    path.write_text(f"#!/bin/sh\necho {text}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _fake_venv(_interpreter: Any, target: Any) -> None:
    """A venv whose ``bin/tool`` prints ``from-venv``."""
    bin_dir = Path(target) / "bin"
    bin_dir.mkdir(parents=True)
    _script(bin_dir / "tool", "from-venv")
    (bin_dir / "python").write_text("")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, str, Path]:
    """A fallback registry holding one env, and a PATH dir with its own ``tool``."""
    monkeypatch.setattr("wispy.cli.fallback.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("wispy.cli.fallback._create_venv", _fake_venv)
    registry = fb.make_fallback_registry(tmp_path / "state")
    created = registry.lookup("environment/create")(CreateEnvironmentParams(name="demo", python_version="3.12"))
    system_bin = tmp_path / "system-bin"
    system_bin.mkdir()
    _script(system_bin / "tool", "from-system-PATH")
    return registry, created["id"], system_bin


def _call(registry: Any, method: str, params: Any) -> Any:
    return registry.lookup(method)(params)


@pytest.mark.parametrize("method", ["environment/execute", "environment/get"])
def test_delete_waits_for_a_running_execute_or_get(
    env: tuple[Any, str, Path],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """A delete issued while the env is in use waits until it is released.

    ``subprocess.run`` is the point where ``execute`` runs the command
    and ``get`` runs ``pip list``; the delete starts right there.
    """
    registry, env_id, system_bin = env
    real_run = subprocess.run
    deleted: list[Any] = []
    delete_thread = threading.Thread(
        target=lambda: deleted.append(_call(registry, "environment/delete", DeleteEnvironmentParams(id=env_id))),
    )
    still_blocked: list[bool] = []

    def run_with_delete_racing(*args: Any, **kwargs: Any) -> Any:
        delete_thread.start()
        delete_thread.join(_BLOCKED_S)
        still_blocked.append(delete_thread.is_alive())
        return real_run(*args, **kwargs)

    monkeypatch.setattr("wispy.cli.fallback.subprocess.run", run_with_delete_racing)

    if method == "environment/execute":
        params = ExecuteParams(id=env_id, argv=("tool",), env={"PATH": str(system_bin)})
        result = _call(registry, method, params)
        assert result["stdout"].strip() == "from-venv"
    else:
        result = _call(registry, method, GetEnvironmentParams(id=env_id))
        assert Path(result["interpreter_path"]).exists()

    assert still_blocked == [True]
    delete_thread.join(10)
    assert deleted == [{"id": env_id}]
    assert _call(registry, "environment/list", None) == []


def test_execute_waiting_on_a_delete_reports_not_found(
    env: tuple[Any, str, Path],
    tmp_path: Path,
) -> None:
    """An execute that queued behind a delete re-checks the index.

    The test holds the env lock exclusively, as a delete does, while an
    execute starts; it then unlists the env and releases the lock.
    """
    registry, env_id, system_bin = env
    state = FallbackState(tmp_path / "state")
    errors: list[WspError] = []

    def execute() -> None:
        try:
            _call(
                registry,
                "environment/execute",
                ExecuteParams(id=env_id, argv=("tool",), env={"PATH": str(system_bin)}),
            )
        except WspError as exc:
            errors.append(exc)

    with state.env_lock(env_id, shared=False):
        worker = threading.Thread(target=execute)
        worker.start()
        worker.join(_BLOCKED_S)
        assert worker.is_alive()
        with state.lock():
            state.atomic_write_index([])
    worker.join(10)
    assert [e.code for e in errors] == [int(WspErrorCode.ENVIRONMENT_NOT_FOUND)]


def test_delete_unlists_before_removing_the_directory(
    env: tuple[Any, str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process killed during removal leaves only an unlisted directory."""
    registry, env_id, _ = env
    state = FallbackState(tmp_path / "state")
    listed_during_removal: list[list[str]] = []

    def killed_during_rmtree(_path: Any, *_args: Any, **_kwargs: Any) -> None:
        listed_during_removal.append([e.id for e in state.read_index()])
        msg = "process killed"
        raise KeyboardInterrupt(msg)

    monkeypatch.setattr("wispy.cli.fallback.shutil.rmtree", killed_during_rmtree)
    with pytest.raises(KeyboardInterrupt):
        _call(registry, "environment/delete", DeleteEnvironmentParams(id=env_id))
    assert listed_during_removal == [[]]
    assert (state.envs_dir / env_id).is_dir()


def test_create_sweeps_directories_left_by_an_interrupted_create(
    env: tuple[Any, str, Path],
    tmp_path: Path,
) -> None:
    """A directory no create or delete finished is removed by the next create."""
    registry, env_id, _ = env
    state = FallbackState(tmp_path / "state")
    orphan = state.envs_dir / "env-crashed0000"
    _fake_venv(None, orphan)
    state.env_lock_path("env-crashed0000").parent.mkdir(exist_ok=True)
    state.env_lock_path("env-crashed0000").write_text("")

    created = _call(
        registry,
        "environment/create",
        CreateEnvironmentParams(name="other", python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"),
    )
    assert sorted(p.name for p in state.envs_dir.iterdir()) == sorted([env_id, created["id"]])
    assert not state.env_lock_path("env-crashed0000").exists()
