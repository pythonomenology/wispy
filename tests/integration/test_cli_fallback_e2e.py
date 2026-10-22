"""End-to-end integration tests for the WSP_CLI in fallback mode.

These tests drive the ``wsp`` CLI as a real Python subprocess with
neither ``--tool`` nor ``--config`` supplied, exercising the
in-process fallback path through every layer the user sees: argparse
plumbing, the lifecycle FSM, the dispatcher, the
:func:`make_fallback_registry` handlers, on-disk persistence in
``index.json``, and a real ``python -m venv`` materialization for
the create round-trip.

Each test isolates state via ``WISPY_STATE_DIR`` pointed at a fresh
``tmp_path`` so persistence between tests is impossible. The CLI is
invoked via ``python -c "from wispy.cli.main import main; ..."`` so
the tests do not depend on the ``wsp`` script entry being
``pip install``-ed into the active environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Resolve ``src/`` so the spawned child can ``import wispy`` even when
# the package has not been installed editably into the active
# environment. The Hatch test environment installs the package, but
# adding ``src/`` to ``PYTHONPATH`` is a no-cost safety net for runs
# from a checkout where it has not.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"


# Inline driver script. Equivalent to ``python -m wispy.cli.main``,
# except that module currently has no ``if __name__ == "__main__"``
# guard, so we replicate the script entry directly. ``sys.argv[1:]``
# strips the ``-c`` synthetic name so the parser sees only the user
# arguments we appended after.
_CLI_DRIVER = "import sys; from wispy.cli.main import main; sys.exit(main(sys.argv[1:]))"


# Major.minor of the running interpreter, used both for the
# ``--python-version`` flag and for the ``shutil.which`` skip-gate
# below.
_THIS_PY = f"{sys.version_info.major}.{sys.version_info.minor}"


def _probe_venv_with_pip() -> tuple[bool, str]:
    """Probe whether ``pythonX.Y -m venv`` (with pip) works here.

    The fallback handler always bootstraps pip, so the
    round-trip test can only succeed on hosts where ``ensurepip``
    actually runs. Some interpreter installations (notably mise-
    managed Pythons that ship without a working ``ensurepip``
    bundle) abort during pip bootstrap; on those hosts the create
    test is exercising the host environment, not the code under
    test, so we skip it. The probe runs in a temp directory that is
    cleaned up regardless of outcome.
    """
    interpreter = shutil.which(f"python{_THIS_PY}")
    if interpreter is None:
        return False, f"python{_THIS_PY} not on PATH"

    tmpdir = Path(tempfile.mkdtemp(prefix="wispy-venv-probe-"))
    try:
        proc = subprocess.run(
            [interpreter, "-m", "venv", str(tmpdir / "v")],
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0:
            return False, f"python -m venv failed on this host: {proc.stderr.decode('utf-8', errors='replace')}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return True, ""


# Skip the venv-materializing tests when the host cannot create a
# pip-enabled venv for the running Python. The fallback handler hard-
# codes ``with_pip=True`` so a working ``ensurepip`` bundle is a
# precondition for the create round-trip; without it we would be
# testing the host environment, not the WSP_CLI.
_VENV_OK, _VENV_REASON = _probe_venv_with_pip()
_REQUIRES_PYTHON_BIN = pytest.mark.skipif(
    not _VENV_OK,
    reason=_VENV_REASON or f"python{_THIS_PY} venv unavailable",
)


def _run_cli(
    argv: list[str],
    state_dir: Path,
    *,
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[bytes]:
    """Spawn the WSP_CLI in fallback mode against ``state_dir``.

    Sets ``WISPY_STATE_DIR`` so :class:`FallbackState` writes into a
    test-isolated directory and prepends ``src/`` to ``PYTHONPATH``
    so the child can import ``wispy`` without an editable install.
    """
    env = os.environ.copy()
    env["WISPY_STATE_DIR"] = str(state_dir)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-c", _CLI_DRIVER, *argv],
        check=False, capture_output=True,
        env=env,
        timeout=timeout,
    )


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Per-test fallback state directory under ``tmp_path``."""
    return tmp_path / "state"


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_list_empty_state_returns_empty_array(state_dir: Path) -> None:
    """``environment/list`` against a fresh state dir prints ``[]``."""
    result = _run_cli(["environment/list"], state_dir)
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}; stderr={result.stderr.decode('utf-8', errors='replace')!r}"
    )
    assert json.loads(result.stdout) == []


def test_get_unknown_id_exits_one_with_error_json_on_stderr(
    state_dir: Path,
) -> None:
    """``environment/get --id <missing>`` emits the WSP error on stderr.

    The fallback handler raises ``WspError(ENVIRONMENT_NOT_FOUND)``,
    which the dispatcher renders as a JSON-RPC error response. The
    client maps non-``-32601`` errors to exit code 1 and prints the
    error object as compact JSON to stderr.
    """
    result = _run_cli(
        ["environment/get", "--id", "env-does-not-exist"],
        state_dir,
    )
    assert result.returncode == 1, (
        f"expected exit 1, got {result.returncode}; stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # stdout MUST be empty for a failed call: the success-printing
    # branch is the only writer of stdout.
    assert result.stdout == b"", result.stdout

    stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
    # The first/only stderr line is the compact JSON error object.
    error = json.loads(stderr_text.splitlines()[-1])
    assert error["code"] == -31001  # WspErrorCode.ENVIRONMENT_NOT_FOUND
    assert "id" in error.get("data", {})


@_REQUIRES_PYTHON_BIN
def test_create_list_get_delete_round_trip(state_dir: Path) -> None:
    """Full CRUD round-trip through the in-process fallback.

    Walks the canonical lifecycle a user would: create an env, see
    it in ``list``, fetch its details via ``get``, delete it, then
    verify ``get`` returns environment-not-found and ``list`` is
    empty again. A real ``python -m venv`` materializes the venv on
    disk, so this also covers the ``envs/<id>/`` filesystem layout.
    """
    # 1. create -> details JSON on stdout, exit 0.
    create = _run_cli(
        [
            "environment/create",
            "--name",
            "scratch",
            "--python-version",
            _THIS_PY,
        ],
        state_dir,
        # ``python -m venv`` is the slow step:
        # bootstrapping pip can take a few seconds on cold caches, so
        # give the subprocess a generous deadline.
        timeout=120.0,
    )
    assert create.returncode == 0, f"create failed: stderr={create.stderr.decode('utf-8', errors='replace')!r}"
    created = json.loads(create.stdout)
    assert created["name"] == "scratch"
    assert created["python_version"] == _THIS_PY
    env_id = created["id"]
    assert isinstance(env_id, str), env_id
    assert env_id, env_id

    # The interpreter path returned from create MUST resolve to the
    # venv layout under ``state_dir/envs/<id>/``. Cross-check against
    # the disk so we know ``python -m venv`` actually ran.
    interpreter_path = Path(created["interpreter_path"])
    assert interpreter_path.is_absolute(), interpreter_path
    assert interpreter_path.exists(), interpreter_path
    assert (state_dir / "envs" / env_id).is_dir()

    # 2. list -> array containing the new env summary.
    listing = _run_cli(["environment/list"], state_dir)
    assert listing.returncode == 0, listing.stderr
    listed = json.loads(listing.stdout)
    assert isinstance(listed, list)
    assert any(entry["id"] == env_id for entry in listed), listed

    # 3. get -> details with matching id/name/python_version.
    got = _run_cli(["environment/get", "--id", env_id], state_dir)
    assert got.returncode == 0, got.stderr
    details = json.loads(got.stdout)
    assert details["id"] == env_id
    assert details["name"] == "scratch"
    assert details["python_version"] == _THIS_PY
    # ``installed_packages`` and ``extra`` are
    # always present in the details form.
    assert "installed_packages" in details
    assert "extra" in details

    # 4. delete -> ack, exit 0.
    deleted = _run_cli(
        ["environment/delete", "--id", env_id],
        state_dir,
    )
    assert deleted.returncode == 0, deleted.stderr
    ack = json.loads(deleted.stdout)
    assert ack == {"id": env_id}
    assert not (state_dir / "envs" / env_id).exists()

    # 5. subsequent get -> environment-not-found, exit 1.
    missing = _run_cli(
        ["environment/get", "--id", env_id],
        state_dir,
    )
    assert missing.returncode == 1, missing.stderr
    err_text = missing.stderr.decode("utf-8", errors="replace").strip()
    err = json.loads(err_text.splitlines()[-1])
    assert err["code"] == -31001

    # 6. list -> back to ``[]``.
    final_list = _run_cli(["environment/list"], state_dir)
    assert final_list.returncode == 0, final_list.stderr
    assert json.loads(final_list.stdout) == []
