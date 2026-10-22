"""Built-in fallback Workflow_Tool backed by ``venv``.

This module wires the WSP method surface (``initialize``, ``shutdown``,
``environment/list``, ``environment/create``, ``environment/get``,
``environment/delete``, ``environment/execute``) to a tiny on-disk
Workflow_Tool whose implementation lives entirely in the Python
standard library: it persists Environments via :class:`FallbackState`
and creates them with ``pythonX.Y -m venv``. The fallback exists so
the WSP_CLI is useful out of the box -- a user with no Workflow_Tool
selected still has working ``environment/*`` calls.

Design highlights:

* The dispatcher runs WSP method validators *before* reaching a
  handler, so each handler receives a normalized params dataclass
  (e.g. :class:`CreateEnvironmentParams`) and only has to enforce the
  handler-level rules: ``name-conflict`` and
  ``python-version-available`` for ``environment/create``,
  unknown-id for ``environment/get`` / ``environment/delete`` /
  ``environment/execute``.
* All index reads and writes are wrapped in :meth:`FallbackState.lock`
  so concurrent WSP_CLI invocations against the same state directory
  cannot corrupt the index (the lock primitive is
  implemented in :mod:`wispy.cli.state`).
* ``environment/get`` and ``environment/execute`` hold the
  environment's own lock shared while they use its directory, and
  ``environment/delete`` holds it exclusive, so a delete cannot remove
  the venv from under a running command. A command run by
  ``environment/execute`` inherits ``WISPY_FALLBACK_ENVS_IN_USE``
  naming that env; a nested ``wsp`` call uses the env without locking
  it again (its ancestor already holds it), and a nested delete of it
  fails instead of waiting forever on its own ancestor.
* ``environment/delete`` drops the env from the index before removing
  its directory, so a removal that fails or is interrupted leaves an
  unlisted directory rather than a listed env with a broken venv.
  Create and delete sweep such leftovers, warning on stderr about any
  they cannot remove.
* ``subprocess`` is imported at module scope, and venv creation goes
  through :func:`_create_venv`, so tests can monkey-patch
  :data:`wispy.cli.fallback.subprocess.run` and
  :func:`wispy.cli.fallback._create_venv` to drive the rule-based
  property tests without ever spawning a real interpreter.
* ``environment/*`` handlers return JSON-serializable dicts (via
  ``Environment``, ``DeleteAck``, and ``ExecuteResult``'s
  ``to_jsonable`` helpers). Returning the models themselves would
  work too: both the server and the in-process CLI path encode them
  through ``to_jsonable``.

See ``design.md`` -> ``cli.fallback``.
"""

from __future__ import annotations

import contextlib
import json as _json
import os
import re
import shutil
import subprocess
import sys
import uuid
from typing import TYPE_CHECKING, Any

from wispy.cli.state import FallbackState
from wispy.endpoints import (
    PROTOCOL_VERSION,
    Capabilities,
    DeleteAck,
    Environment,
    ExecuteResult,
    Package,
)
from wispy.errors import WspError, WspErrorCode
from wispy.registry import HandlerRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = ["make_fallback_registry"]


# Name of the venv subdirectory that contains the interpreter. POSIX
# venvs use ``bin``; Windows venvs use ``Scripts``.
_VENV_BIN_DIR = "Scripts" if os.name == "nt" else "bin"

# Name of the interpreter executable inside ``_VENV_BIN_DIR``.
_VENV_PY_NAME = "python.exe" if os.name == "nt" else "python"

# The validators in :mod:`wispy.endpoints` already enforce the format
# ``MAJOR.MINOR(.PATCH)?``, so this regex only has to extract the
# leading two components for ``shutil.which`` lookup.
_PY_MAJOR_MINOR_RE = re.compile(r"^(\d+)\.(\d+)")

# Set in the environment of every command ``environment/execute`` runs:
# the ids of the envs whose locks that command's ancestors hold,
# separated by ``os.pathsep``.
_ENVS_IN_USE_VAR = "WISPY_FALLBACK_ENVS_IN_USE"

# Wall-clock limits for the subprocesses the fallback runs.
_VENV_CREATE_TIMEOUT_S = 300
_EXECUTE_TIMEOUT_S = 300


def _resolve_python(python_version: str) -> str | None:
    """Return the path to a ``pythonX.Y`` interpreter on PATH, if any.

    Used to implement the ``python-version-available`` handler-level
    rule. Returns ``None`` when no compatible
    interpreter is on PATH; the caller turns that into a
    ``PYTHON_VERSION_UNAVAILABLE`` :class:`WspError`.
    """
    match = _PY_MAJOR_MINOR_RE.match(python_version)
    if match is None:
        # Defensive: the validator should have caught a malformed
        # version string before we got here, but we refuse to crash if
        # it did not.
        return None
    major, minor = match.group(1), match.group(2)
    return shutil.which(f"python{major}.{minor}")


def _create_venv(interpreter: str, target: Path) -> None:
    """Create a venv (with pip) at ``target`` based on ``interpreter``.

    Runs ``<interpreter> -m venv`` rather than :class:`venv.EnvBuilder`:
    ``EnvBuilder`` always builds on the interpreter running wispy, which
    would silently ignore the requested Python version.

    Raises on any failure; the caller maps that to ``EXECUTION_FAILED``.
    """
    proc = subprocess.run(
        [interpreter, "-m", "venv", str(target)],
        check=False,
        capture_output=True,
        timeout=_VENV_CREATE_TIMEOUT_S,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        msg = f"{interpreter} -m venv exited with status {proc.returncode}: {stderr}"
        raise RuntimeError(msg)


def _read_index(state: FallbackState) -> list[Environment]:
    """Read the state index, mapping corruption to ``EXECUTION_FAILED``.

    Callers must hold ``state.lock()``.
    """
    try:
        return state.read_index()
    except (TypeError, ValueError) as exc:
        raise WspError(
            int(WspErrorCode.EXECUTION_FAILED),
            f"fallback state index {state.index_path} is corrupt: {exc}",
            data={"reason": "index-corrupt", "path": str(state.index_path)},
        ) from exc


def _not_found(env_id: str) -> WspError:
    """``environment-not-found`` for ``env_id``."""
    return WspError(
        int(WspErrorCode.ENVIRONMENT_NOT_FOUND),
        f"environment {env_id!r} not found",
        data={"id": env_id},
    )


def _find_env(state: FallbackState, env_id: str) -> Environment | None:
    """Return the indexed environment ``env_id``, reading under the lock."""
    with state.lock():
        envs = _read_index(state)
    return next((e for e in envs if e.id == env_id), None)


def _envs_in_use_by_caller() -> set[str]:
    """Ids of the envs an ancestor ``environment/execute`` holds locked."""
    raw = os.environ.get(_ENVS_IN_USE_VAR, "")
    return {env_id for env_id in raw.split(os.pathsep) if env_id}


@contextlib.contextmanager
def _using_env(state: FallbackState, env_id: str) -> Iterator[Environment]:
    """Yield the indexed environment ``env_id`` while holding it shared.

    ``environment/delete`` needs the same lock exclusively, so the
    environment's directory stays whole until the block exits. The id
    is checked after locking, because a delete may have finished while
    this caller waited.

    A caller running inside an ``environment/execute`` of the same env
    skips the lock: its ancestor holds it for longer than this call
    lasts, and on Windows, where a shared lock is exclusive, taking it
    again would wait on that ancestor forever.
    """
    if env_id in _envs_in_use_by_caller():
        lock: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
    else:
        lock = state.env_lock(env_id, shared=True)
    with lock:
        match = _find_env(state, env_id)
        if match is None:
            raise _not_found(env_id)
        yield match


def _warn(message: str) -> None:
    """Write a warning to stderr, which is never the WSP stdout channel."""
    sys.stderr.write(f"wispy: warning: {message}\n")


def _sweep_orphans(state: FallbackState, envs: list[Environment]) -> None:
    """Remove env directories and lock files the index does not list.

    They are left by a delete whose removal failed part-way, a failed
    create whose cleanup failed, a process killed mid-create or
    mid-delete, and (lock files only) requests naming unknown ids.
    Callers must hold ``state.lock()``, which every create holds while
    its directory is unlisted, so a sweep never removes a directory
    that is still being built. A directory that cannot be removed is
    reported on stderr and retried by the next sweep. Lock files that
    cannot be removed are expected (on Windows, another process may
    have one open) and are skipped silently.
    """
    live = {e.id for e in envs}
    for child in state.envs_dir.iterdir():
        if child.name in live:
            continue
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            _warn(f"could not remove leftover environment directory {child}: {exc}")
    if state.env_locks_dir.is_dir():
        live_locks = {state.env_lock_path(env_id).name for env_id in live}
        for lock_file in state.env_locks_dir.glob("*.lock"):
            if lock_file.name not in live_locks:
                with contextlib.suppress(OSError):
                    lock_file.unlink()


def _venv_interpreter_path(envs_dir: Path, env_id: str) -> Path:
    """Return the absolute path to the interpreter inside an env."""
    return envs_dir / env_id / _VENV_BIN_DIR / _VENV_PY_NAME


def _list_installed_packages(interpreter: Path) -> list[Package]:
    """Best-effort enumeration of packages installed in the venv.

    Calls ``<interpreter> -m pip list --format=json`` with a tight
    timeout. Any failure (pip absent, non-zero exit, malformed output,
    timeout) yields ``[]`` rather than raising; ``environment/get``
    must succeed for envs without pip.
    """
    try:
        proc = subprocess.run(
            [str(interpreter), "-m", "pip", "list", "--format=json"],
            check=False, capture_output=True,
            timeout=20,
        )
    except (
        FileNotFoundError,
        PermissionError,
        OSError,
        subprocess.TimeoutExpired,
    ):
        return []
    if proc.returncode != 0:
        return []
    try:
        raw = proc.stdout.decode("utf-8", errors="replace")
        data = _json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return []
    if not isinstance(data, list):
        return []
    out: list[Package] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        version = entry.get("version")
        if isinstance(name, str) and isinstance(version, str):
            out.append(Package(name=name, version=version))
    return out


def make_fallback_registry(
    state_root: Path | None = None,
) -> HandlerRegistry:
    """Build a :class:`HandlerRegistry` wired to the fallback handlers.

    Args:
        state_root: Optional override for the on-disk state directory.
            When ``None``, :class:`FallbackState` resolves the
            location from ``$WISPY_STATE_DIR`` / ``$XDG_STATE_HOME`` /
            ``%LOCALAPPDATA%`` per its documented precedence. Tests
            pass an explicit path so each invocation gets a fresh
            tmpdir.

    Returns:
        A :class:`HandlerRegistry` with bindings for ``initialize``,
        ``shutdown``, and every ``environment/*`` method.
    """
    state = FallbackState(state_root)
    state.ensure_layout()
    registry = HandlerRegistry()

    # ------------------------------------------------------------- #
    # Lifecycle handlers.
    # ------------------------------------------------------------- #

    def _initialize(_params: Any) -> Any:
        return Capabilities(
            methods=tuple(registry.methods()),
            protocol_version=PROTOCOL_VERSION,
        )

    def _shutdown(_params: Any) -> None:
        return None

    # ------------------------------------------------------------- #
    # environment/list.
    # ------------------------------------------------------------- #

    def _environment_list(_params: Any) -> Any:
        with state.lock():
            envs = _read_index(state)
        # Deterministic ordering by ascending id.
        envs_sorted = sorted(envs, key=lambda e: e.id)
        return [e.to_jsonable() for e in envs_sorted]

    # ------------------------------------------------------------- #
    # environment/create.
    # ------------------------------------------------------------- #

    def _environment_create(params: Any) -> Any:
        # ``params`` is a CreateEnvironmentParams dataclass thanks to
        # the dispatcher's WSP_METHODS validation; the only checks
        # left are the handler-level ones.
        with state.lock():
            envs = _read_index(state)

            # Handler-level violations. Order matters for the
            # error-code priority:
            # ``name-conflict`` > ``python-version-available``.
            violations: list[str] = []
            if any(e.name == params.name for e in envs):
                violations.append("name-conflict")
            interp = _resolve_python(params.python_version)
            if interp is None:
                violations.append("python-version-available")

            if violations:
                if "name-conflict" in violations:
                    raise WspError(
                        int(WspErrorCode.ENVIRONMENT_NAME_CONFLICT),
                        f"environment name {params.name!r} already exists",
                        data={
                            "violations": violations,
                            "name": params.name,
                        },
                    )
                # Only python-version-available remains.
                raise WspError(
                    int(WspErrorCode.PYTHON_VERSION_UNAVAILABLE),
                    f"python version {params.python_version!r} is not available on PATH",
                    data={"violations": violations},
                )

            # Clear leftovers of earlier failed or interrupted
            # operations before adding a directory of our own.
            _sweep_orphans(state, envs)

            new_id = f"env-{uuid.uuid4().hex[:12]}"
            target = state.envs_dir / new_id

            # The venv is built on the interpreter resolved for the
            # requested version. ``-m venv`` installs pip, so the env is
            # immediately useful; the property tests stub this call so
            # the cost only matters in real CLI invocations.
            try:
                _create_venv(interp, target)
            except Exception as exc:
                # Clean up the partial directory and
                # leave the index untouched so a retry behaves like a
                # fresh request.
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                raise WspError(
                    int(WspErrorCode.EXECUTION_FAILED),
                    f"failed to create venv: {exc}",
                    data={"reason": "venv-create-failed"},
                ) from exc

            interpreter_path = _venv_interpreter_path(state.envs_dir, new_id)

            # Persist the summary form. The detail-only fields are
            # reconstructed on demand by ``environment/get`` so the
            # index stays compact and the truth lives on disk.
            summary = Environment(
                id=new_id,
                name=params.name,
                python_version=params.python_version,
            )
            envs.append(summary)
            state.atomic_write_index(envs)

        # Return the details form. ``installed_packages`` and
        # ``extra`` are present on the
        # wire even when empty; the to_jsonable serializer preserves
        # that.
        details = Environment(
            id=new_id,
            name=params.name,
            python_version=params.python_version,
            interpreter_path=str(interpreter_path),
            installed_packages=(),
            extra={},
        )
        return details.to_jsonable()

    # ------------------------------------------------------------- #
    # environment/get.
    # ------------------------------------------------------------- #

    def _environment_get(params: Any) -> Any:
        # Reconstruct the details form. ``interpreter_path`` is taken
        # from the on-disk layout (the index never persists it);
        # ``installed_packages`` is queried lazily; ``extra`` is the
        # always-present empty placeholder reserved for tool-specific
        # fields.
        with _using_env(state, params.id) as match:
            interpreter = _venv_interpreter_path(state.envs_dir, match.id)
            packages = _list_installed_packages(interpreter)
        details = Environment(
            id=match.id,
            name=match.name,
            python_version=match.python_version,
            interpreter_path=str(interpreter),
            installed_packages=tuple(packages),
            extra={},
        )
        return details.to_jsonable()

    # ------------------------------------------------------------- #
    # environment/delete.
    # ------------------------------------------------------------- #

    def _environment_delete(params: Any) -> Any:
        if params.id in _envs_in_use_by_caller():
            # The command running this delete was started by an
            # ``environment/execute`` of the same env, which holds the
            # env lock until the command ends: waiting would deadlock.
            if _find_env(state, params.id) is None:
                raise _not_found(params.id)
            raise WspError(
                int(WspErrorCode.EXECUTION_FAILED),
                f"environment {params.id!r} is in use by the command running this delete",
                data={"reason": "env-in-use", "id": params.id},
            )
        # The exclusive env lock waits for running get/execute calls.
        with state.env_lock(params.id, shared=False), state.lock():
            envs = _read_index(state)
            remaining = [e for e in envs if e.id != params.id]
            if len(remaining) == len(envs):
                raise _not_found(params.id)
            # Unlist first: if removing the directory then fails or the
            # process dies, the env is gone either way and only an
            # unlisted directory is left for the next sweep, never a
            # listed env with a half-removed venv.
            state.atomic_write_index(remaining)
            _sweep_orphans(state, remaining)
        return DeleteAck(id=params.id).to_jsonable()

    # ------------------------------------------------------------- #
    # environment/execute.
    # ------------------------------------------------------------- #

    def _environment_execute(params: Any) -> Any:
        # Hold the env shared for the whole run so a concurrent delete
        # cannot remove the venv and leave PATH lookup to fall through
        # to the caller's interpreter.
        with _using_env(state, params.id) as match:
            return _execute_in(match.id, params)

    def _execute_in(env_id: str, params: Any) -> Any:
        """Run ``params.argv`` with the env's bin directory first on PATH."""
        bin_dir = state.envs_dir / env_id / _VENV_BIN_DIR

        # Prepend the env's bin/Scripts directory to
        # PATH so executables installed inside the venv (the
        # interpreter, console-scripts) are picked up by the child.
        # The user's ``env`` overlay (when supplied) sits between the
        # parent's environment and the PATH adjustment, matching the
        # principle of least surprise: ``PATH`` from ``env`` is also
        # respected as the base, but the venv directory still wins.
        child_env: dict[str, str] = dict(os.environ)
        if params.env:
            child_env.update(params.env)
        existing_path = child_env.get("PATH", "")
        child_env["PATH"] = f"{bin_dir}{os.pathsep}{existing_path}" if existing_path else str(bin_dir)
        # Tell nested fallback calls that this env is held for them.
        in_use = child_env.get(_ENVS_IN_USE_VAR, "")
        child_env[_ENVS_IN_USE_VAR] = f"{in_use}{os.pathsep}{env_id}" if in_use else env_id

        try:
            proc = subprocess.run(
                list(params.argv),
                check=False, cwd=params.cwd,
                env=child_env,
                capture_output=True,
                timeout=_EXECUTE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise WspError(
                int(WspErrorCode.EXECUTION_FAILED),
                f"command timed out after {_EXECUTE_TIMEOUT_S}s",
                data={"reason": "timeout"},
            ) from exc
        except OSError as exc:
            # Command launch failure -> WSP error
            # ``execution-failed`` with a human-readable message in
            # ``data`` so clients can surface the cause.
            raise WspError(
                int(WspErrorCode.EXECUTION_FAILED),
                f"failed to launch command: {exc}",
                data={"reason": "spawn-failed"},
            ) from exc

        # Decode bytes with ``errors="replace"`` so
        # arbitrary child output never crashes the response path.
        stdout_bytes = proc.stdout if proc.stdout is not None else b""
        stderr_bytes = proc.stderr if proc.stderr is not None else b""
        return ExecuteResult(
            exit_code=proc.returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        ).to_jsonable()

    # ------------------------------------------------------------- #
    # Registration.
    # ------------------------------------------------------------- #

    registry.register("initialize", _initialize)
    registry.register("shutdown", _shutdown)
    registry.register("environment/list", _environment_list)
    registry.register("environment/create", _environment_create)
    registry.register("environment/get", _environment_get)
    registry.register("environment/delete", _environment_delete)
    registry.register("environment/execute", _environment_execute)
    return registry
