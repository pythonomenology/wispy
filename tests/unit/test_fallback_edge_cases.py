"""Unit tests for fallback Workflow_Tool edge cases.

These tests pin down the failure-mode contract of the built-in
fallback handlers in :mod:`wispy.cli.fallback` -- the cases that are
inconvenient to express in the rule-based property tests because they
require a one-shot monkey-patch of ``wispy.cli.fallback._create_venv``,
``shutil.rmtree``, or ``subprocess.run`` and exercise paths the
state machine model does not enumerate.

Because the fallback handlers receive *already-validated* params
dataclasses from the dispatcher, these tests bypass the dispatcher
and call the registered handler directly via
:meth:`HandlerRegistry.lookup` with pre-built normalized params. The
behaviour under test is the handler-level rule, not param parsing
(that surface is covered by ``tests/unit/test_endpoints.py``).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from wispy.cli.fallback import make_fallback_registry
from wispy.cli.state import FallbackState
from wispy.endpoints import (
    CreateEnvironmentParams,
    DeleteEnvironmentParams,
    Environment,
    ExecuteParams,
    GetEnvironmentParams,
)
from wispy.errors import WspError, WspErrorCode

# ---------------------------------------------------------------------------
# Helpers and fixtures.
# ---------------------------------------------------------------------------


def _which_python(name: str) -> str | None:
    """Return a fake interpreter path for any ``pythonMAJOR.MINOR`` query.

    The fallback's ``environment/create`` handler calls
    ``shutil.which(f"python{major}.{minor}")`` to satisfy the
    ``python-version-available`` rule. These unit tests do not care
    whether a real interpreter exists on PATH; they monkey-patch
    ``shutil.which`` so the rule passes deterministically and the
    test focuses on the failure path under exercise.
    """
    if name.startswith("python"):
        return f"/usr/bin/{name}"
    return None


def _make_dir_create(_interpreter: Any, target: Any) -> None:
    """Stand-in for ``wispy.cli.fallback._create_venv`` that just makes the dir.

    A real venv creation is too slow for unit tests and
    pulls in a real interpreter. The fallback only cares that the
    target directory exists after a successful create; this stub
    satisfies that contract without invoking ``ensurepip``.
    """
    Path(target).mkdir(parents=True, exist_ok=True)


def _call(registry: Any, method: str, params: Any) -> Any:
    """Invoke a registered handler directly, awaiting if needed.

    The dispatcher ordinarily runs sync handlers on a thread-pool
    executor; for unit tests we just call them inline. Coroutine
    handlers (none currently exist in the fallback, but be future
    proof) are driven via ``asyncio.run``.
    """
    handler = registry.lookup(method)
    assert handler is not None, f"handler not registered: {method}"
    result = handler(params)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


@pytest.fixture
def fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build a fallback registry rooted at a fresh tmp dir.

    Patches ``shutil.which`` at the fallback module level so the
    ``python-version-available`` rule reliably passes regardless of
    which interpreters are installed on the test host.
    """
    monkeypatch.setattr("wispy.cli.fallback.shutil.which", _which_python)
    state_root = tmp_path / "state"
    return make_fallback_registry(state_root)


# ---------------------------------------------------------------------------
# environment/create failure path.
# ---------------------------------------------------------------------------


def test_create_failure_removes_partial_dir_and_does_not_update_index(
    fallback: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``venv`` failure cleans up the partial dir and skips the index.

    When ``wispy.cli.fallback._create_venv`` raises during
    ``environment/create``, the handler must remove the partial
    ``envs/<id>/`` directory and leave the persisted index unchanged
    so a retry behaves like a fresh request.
    """

    def _failing_create(_interpreter: Any, target: Any) -> None:
        # Simulate a venv that started laying down files before
        # blowing up: the directory exists when the exception is
        # raised, exercising the cleanup branch.
        Path(target).mkdir(parents=True, exist_ok=True)
        msg = "simulated venv failure"
        raise RuntimeError(msg)

    monkeypatch.setattr("wispy.cli.fallback._create_venv", _failing_create)

    with pytest.raises(WspError) as excinfo:
        _call(
            fallback,
            "environment/create",
            CreateEnvironmentParams(name="foo", python_version="3.12"),
        )
    assert excinfo.value.code == int(WspErrorCode.EXECUTION_FAILED)

    # The index must reflect zero environments because the failed
    # create never ran ``atomic_write_index``.
    listing = _call(fallback, "environment/list", None)
    assert listing == []

    # And the partial directory must have been cleaned up. Look at
    # ``envs_dir`` directly: it should be empty.
    state = FallbackState(tmp_path / "state")
    assert list(state.envs_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# environment/delete failure path.
# ---------------------------------------------------------------------------


def test_delete_rmtree_failure_unlists_env_and_next_create_sweeps_it(
    fallback: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``shutil.rmtree`` OSError still deletes the env.

    The index is rewritten before the directory is removed, so a
    removal that fails part-way cannot leave a listed env whose venv
    is half gone (``tla/FallbackStore`` trace ``IndexedComplete``).
    The leftover directory is unlisted and the next create removes it.
    """
    # Create an env first using a stub create so rmtree has something
    # to operate on.
    monkeypatch.setattr("wispy.cli.fallback._create_venv", _make_dir_create)
    created = _call(
        fallback,
        "environment/create",
        CreateEnvironmentParams(name="foo", python_version="3.12"),
    )
    env_id = created["id"]

    # Now sabotage shutil.rmtree so the delete handler hits the
    # OSError branch.
    def _failing_rmtree(_path: Any, *_args: Any, **_kwargs: Any) -> None:
        msg = "simulated rmtree failure"
        raise OSError(msg)

    real_rmtree = shutil.rmtree
    monkeypatch.setattr("wispy.cli.fallback.shutil.rmtree", _failing_rmtree)

    ack = _call(fallback, "environment/delete", DeleteEnvironmentParams(id=env_id))
    assert ack == {"id": env_id}
    assert _call(fallback, "environment/list", None) == []
    with pytest.raises(WspError) as excinfo:
        _call(fallback, "environment/get", GetEnvironmentParams(id=env_id))
    assert excinfo.value.code == int(WspErrorCode.ENVIRONMENT_NOT_FOUND)

    state = FallbackState(tmp_path / "state")
    assert (state.envs_dir / env_id).is_dir()

    monkeypatch.setattr("wispy.cli.fallback.shutil.rmtree", real_rmtree)
    created = _call(
        fallback,
        "environment/create",
        CreateEnvironmentParams(name="bar", python_version="3.12"),
    )
    assert [p.name for p in state.envs_dir.iterdir()] == [created["id"]]


def test_delete_unknown_id_raises_environment_not_found(fallback: Any) -> None:
    """Deleting an unknown id surfaces ``environment-not-found``."""
    with pytest.raises(WspError) as excinfo:
        _call(
            fallback,
            "environment/delete",
            DeleteEnvironmentParams(id="env-does-not-exist"),
        )
    assert excinfo.value.code == int(WspErrorCode.ENVIRONMENT_NOT_FOUND)
    assert excinfo.value.data == {"id": "env-does-not-exist"}


# ---------------------------------------------------------------------------
# environment/execute failure path.
# ---------------------------------------------------------------------------


def test_execute_spawn_failure_raises_execution_failed(
    fallback: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subprocess spawn failure surfaces as ``execution-failed``.

    When the requested command cannot be launched
    (executable missing, cwd missing, etc.) the handler must raise
    ``WspError(EXECUTION_FAILED, ...)``.
    """
    monkeypatch.setattr("wispy.cli.fallback._create_venv", _make_dir_create)
    created = _call(
        fallback,
        "environment/create",
        CreateEnvironmentParams(name="foo", python_version="3.12"),
    )
    env_id = created["id"]

    def _failing_run(*_args: Any, **_kwargs: Any) -> None:
        msg = "simulated executable not found"
        raise FileNotFoundError(msg)

    monkeypatch.setattr("wispy.cli.fallback.subprocess.run", _failing_run)

    with pytest.raises(WspError) as excinfo:
        _call(
            fallback,
            "environment/execute",
            ExecuteParams(
                id=env_id,
                argv=("nonexistent-binary",),
                cwd=None,
                env=None,
            ),
        )
    assert excinfo.value.code == int(WspErrorCode.EXECUTION_FAILED)


def test_execute_unknown_id_raises_environment_not_found(
    fallback: Any,
) -> None:
    """Executing in an unknown env surfaces ``environment-not-found``."""
    with pytest.raises(WspError) as excinfo:
        _call(
            fallback,
            "environment/execute",
            ExecuteParams(
                id="env-does-not-exist",
                argv=("echo",),
                cwd=None,
                env=None,
            ),
        )
    assert excinfo.value.code == int(WspErrorCode.ENVIRONMENT_NOT_FOUND)
    assert excinfo.value.data == {"id": "env-does-not-exist"}


# ---------------------------------------------------------------------------
# environment/get edge cases.
# ---------------------------------------------------------------------------


def test_get_unknown_id_includes_id_in_data(fallback: Any) -> None:
    """``environment/get`` for an unknown id includes the id in data.

    The handler must respond with
    ``ENVIRONMENT_NOT_FOUND`` and include the requested ``id`` in
    the error's ``data`` field.
    """
    with pytest.raises(WspError) as excinfo:
        _call(
            fallback,
            "environment/get",
            GetEnvironmentParams(id="env-missing"),
        )
    assert excinfo.value.code == int(WspErrorCode.ENVIRONMENT_NOT_FOUND)
    assert excinfo.value.data == {"id": "env-missing"}


def test_get_returns_full_details_with_extra_always_present(
    fallback: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``environment/get`` returns the details form with ``extra`` always set.

    The details payload must include
    ``interpreter_path``, ``installed_packages``, and ``extra`` --
    even when ``extra`` is empty. This test also stubs
    ``subprocess.run`` so the pip-list invocation does not actually
    spawn anything.
    """
    monkeypatch.setattr("wispy.cli.fallback._create_venv", _make_dir_create)
    created = _call(
        fallback,
        "environment/create",
        CreateEnvironmentParams(name="foo", python_version="3.12"),
    )
    env_id = created["id"]

    # Stub subprocess.run so the lazy pip-list call inside
    # environment/get returns no packages without spawning anything.
    class _FakeProc:
        returncode = 0
        stdout = b"[]"
        stderr = b""

    def _fake_run(*_args: Any, **_kwargs: Any) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr("wispy.cli.fallback.subprocess.run", _fake_run)

    details = _call(
        fallback,
        "environment/get",
        GetEnvironmentParams(id=env_id),
    )

    # All three detail fields must be present, with ``extra`` an
    # empty mapping rather than absent.
    assert "interpreter_path" in details
    assert "installed_packages" in details
    assert "extra" in details
    assert details["extra"] == {}
    assert details["installed_packages"] == []
    # And the round-trip through ``Environment.from_jsonable`` reports
    # the value as a details-form Environment.
    env = Environment.from_jsonable(details)
    assert env.is_details


# ---------------------------------------------------------------------------
# environment/list edge cases (ordering and emptiness).
# ---------------------------------------------------------------------------


def test_list_returns_empty_when_index_is_empty(fallback: Any) -> None:
    """``environment/list`` returns ``[]`` against a fresh state dir."""
    assert _call(fallback, "environment/list", None) == []


def test_list_ordering_is_stable_and_ascending_by_id(
    fallback: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consecutive ``environment/list`` calls yield equal, sorted output."""
    monkeypatch.setattr("wispy.cli.fallback._create_venv", _make_dir_create)

    # Create several envs; ids are uuid-prefixed so insertion order is
    # not the same as id order in general.
    for n in range(5):
        _call(
            fallback,
            "environment/create",
            CreateEnvironmentParams(name=f"env-{n}", python_version="3.12"),
        )

    first = _call(fallback, "environment/list", None)
    second = _call(fallback, "environment/list", None)
    third = _call(fallback, "environment/list", None)

    # Successive calls return identical output (stability).
    assert first == second == third

    # And the ids are sorted ascending (deterministic ordering
    # contract).
    ids = [entry["id"] for entry in first]
    assert ids == sorted(ids)
    assert len(ids) == 5


# ---------------------------------------------------------------------------
# State directory honors the WISPY_STATE_DIR environment override.
# ---------------------------------------------------------------------------


def test_state_dir_respects_wispy_state_dir_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``WISPY_STATE_DIR`` overrides the default state location.

    The fallback delegates state-directory resolution to
    :class:`FallbackState`, which reads ``WISPY_STATE_DIR`` first.
    Constructing the registry with no explicit ``state_root`` and the
    env var set must produce a state directory at the override path.
    """
    custom = tmp_path / "custom-state"
    monkeypatch.setenv("WISPY_STATE_DIR", str(custom))
    monkeypatch.setattr("wispy.cli.fallback.shutil.which", _which_python)

    registry = make_fallback_registry()  # No state_root override.

    # ``make_fallback_registry`` calls ``ensure_layout`` already, so
    # the directory exists by now. Verify that the override path was
    # honored, and that a list call works against it.
    assert custom.exists()
    assert (custom / "envs").exists()
    assert _call(registry, "environment/list", None) == []
