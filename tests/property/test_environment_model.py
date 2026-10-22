"""Property test for environment endpoints model.

The environment endpoints exposed by
:func:`wispy.cli.fallback.make_fallback_registry` are treated as a
stateful system whose abstract state is a map ``id -> Environment``.
A Hypothesis :class:`~hypothesis.stateful.RuleBasedStateMachine`
drives the system through arbitrary sequences of ``create``, ``get``,
``delete``, ``list``, and ``execute`` calls (routed through the WSP
dispatcher so the WSP method validators also run) and a shadow oracle
maintained by the test mirrors the expected state. After every rule the
machine asserts that:

* every id reported by ``environment/list`` matches a record in the
  oracle, the order is ascending by ``id``, and two consecutive
  ``environment/list`` calls return identical results,
* ``environment/get`` for a known id returns the full details object
  with all required fields present,
* ``environment/get`` / ``environment/delete`` / ``environment/execute``
  for an unknown id return an ``environment-not-found`` WSP error and
  leave the oracle unchanged,
* ``environment/create`` for a duplicate name combined with an
  unavailable Python version returns an ``environment-name-conflict``
  error (code precedence) whose ``data.violations`` lists both rules,
* ``environment/execute`` with an empty or non-string ``argv`` returns
  JSON-RPC ``-32602`` and never reaches the handler,
* a successful ``environment/execute`` result has an ``int`` exit code
  and ``str`` stdout/stderr.

The cost of a real venv creation and of every
``subprocess.run`` invocation would dominate this test by orders of
magnitude. Both are reachable at module scope in
:mod:`wispy.cli.fallback` precisely so the test can replace them via
:mod:`unittest.mock`; see the module docstring there for the contract.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

# ``typing.override`` is available natively from Python 3.12. On 3.10 /
# 3.11 we fall back to a no-op decorator so the source still imports
# cleanly. pyrefly runs on 3.12 and picks the native import branch.
if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover - exercised only on 3.10 / 3.11

    def override(arg: Any) -> Any:
        return arg


from hypothesis import HealthCheck, assume, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from wispy.cli.fallback import make_fallback_registry
from wispy.dispatcher import dispatch
from wispy.endpoints import Capabilities
from wispy.errors import JsonRpcErrorCode, WspErrorCode
from wispy.lifecycle import LifecycleManager
from wispy.protocol import JsonRpcError, JsonRpcRequest, JsonRpcResponse

# Versions for which the fake :func:`shutil.which` reports an
# interpreter is on PATH. Anything else fails the
# ``python-version-available`` handler-level rule.
_AVAILABLE_PY_VERSIONS: tuple[str, ...] = ("3.10", "3.11", "3.12")

# A syntactically valid python version that the fake :func:`shutil.which`
# reports as unavailable. Used to drive the violation-precedence rule.
_UNAVAILABLE_PY_VERSION = "9.99"

# Method-name compact aliases so the assertions read closely to the
# design's transition table rather than to the WspErrorCode enum.
_INVALID_PARAMS = int(JsonRpcErrorCode.INVALID_PARAMS)
_ENV_NOT_FOUND = int(WspErrorCode.ENVIRONMENT_NOT_FOUND)
_ENV_NAME_CONFLICT = int(WspErrorCode.ENVIRONMENT_NAME_CONFLICT)
_PY_VERSION_UNAVAILABLE = int(WspErrorCode.PYTHON_VERSION_UNAVAILABLE)


# ---------------------------------------------------------------------- #
# Stubs for the slow / OS-dependent calls the fallback handlers make.
# ---------------------------------------------------------------------- #


def _fake_subprocess_run(argv: Any, **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    """Stub for :func:`subprocess.run` used inside the fallback handlers.

    Returns a successful :class:`subprocess.CompletedProcess` whose
    stdout is the JSON literal ``[]`` (so :func:`_list_installed_packages`
    decodes it as an empty package list) and whose stderr is empty.
    Both call sites in :mod:`wispy.cli.fallback` only require that
    ``returncode`` is an :class:`int` and that the captured byte
    streams decode to UTF-8.
    """
    return subprocess.CompletedProcess(args=argv, returncode=0, stdout=b"[]", stderr=b"")


def _fake_create_venv(_interpreter: Any, _target_dir: Any) -> None:
    """No-op stub for :func:`wispy.cli.fallback._create_venv`.

    The fallback handler does not stat the venv after creation
    (``environment/get`` reconstructs ``interpreter_path`` purely from
    the layout convention) and ``environment/delete`` short-circuits
    on a missing target directory, so the rule machine never observes
    the absence of the on-disk venv.
    """
    return None


def _fake_which(cmd: Any, *_args: Any, **_kwargs: Any) -> str | None:
    """Stub for :func:`shutil.which` used by ``_resolve_python``.

    Returns a fabricated path for any ``pythonX.Y`` whose ``X.Y`` is
    in :data:`_AVAILABLE_PY_VERSIONS`; returns ``None`` otherwise so
    the ``python-version-available`` handler-level rule can fail on
    demand (e.g. for :data:`_UNAVAILABLE_PY_VERSION`).
    """
    if not isinstance(cmd, str):
        return None
    for version in _AVAILABLE_PY_VERSIONS:
        if cmd == f"python{version}":
            return f"/usr/bin/{cmd}"
    return None


# ---------------------------------------------------------------------- #
# Hypothesis strategies.
# ---------------------------------------------------------------------- #


# Restrict the alphabet to non-surrogate, non-control characters so a
# generated name survives the round-trip through ``json.dumps`` /
# ``json.loads`` that ``FallbackState.atomic_write_index`` performs.
# The validator allows any non-empty string up to 256 chars; the
# property under test does not depend on the character set.
_name_text = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs", "Cc"),
    ),
    min_size=1,
    max_size=32,
)

# Versions we draw for create-rule generation. Using a mix of available
# and unavailable versions exercises both the success path and the
# python-version-unavailable handler-level rule.
_python_versions = st.sampled_from((*_AVAILABLE_PY_VERSIONS, _UNAVAILABLE_PY_VERSION))

# Arbitrary id strings used by the unknown-id rules. The fallback
# generates ids of the form ``env-<12 hex chars>`` so the chance of
# collision with a hypothesis-drawn string is negligible, but each
# rule still uses ``assume(...)`` to skip if the draw collides.
_arbitrary_id = st.text(min_size=1, max_size=24)


# ---------------------------------------------------------------------- #
# Rule machine.
# ---------------------------------------------------------------------- #


class EnvironmentModelMachine(RuleBasedStateMachine):
    """Environment endpoints behave as a model.

    See module docstring for the property statement. The machine wires
    a freshly-built fallback :class:`~wispy.registry.HandlerRegistry`
    to a :class:`~wispy.lifecycle.LifecycleManager` already advanced
    past the ``-32002`` initialize gate, so non-``initialize`` rules
    are admitted directly.
    """

    def __init__(self) -> None:
        super().__init__()
        # Patch the slow / OS-dependent calls before constructing the
        # registry so the same patches are observable for the entire
        # rule-machine lifetime. ``teardown`` stops them.
        self._patches: list[Any] = [
            mock.patch(
                "wispy.cli.fallback.subprocess.run",
                side_effect=_fake_subprocess_run,
            ),
            mock.patch(
                "wispy.cli.fallback._create_venv",
                _fake_create_venv,
            ),
            mock.patch(
                "wispy.cli.fallback.shutil.which",
                side_effect=_fake_which,
            ),
        ]
        for patcher in self._patches:
            patcher.start()

        self._tmp = Path(tempfile.mkdtemp(prefix="wispy-fallback-test-"))
        self.registry = make_fallback_registry(self._tmp)
        self.lifecycle = LifecycleManager()
        # Advance the lifecycle past the -32002 gate so the dispatcher
        # admits every non-``initialize`` request.
        self.lifecycle.on_initialize_success(
            Capabilities(
                methods=tuple(self.registry.methods()),
                protocol_version="0.1.0",
            )
        )
        # Shadow oracle: id -> {"name": str, "python_version": str}.
        self.oracle: dict[str, dict[str, str]] = {}

    @override
    def teardown(self) -> None:
        # Stop the patches in reverse start-order so each unwind sees
        # the same state the corresponding ``start`` saw, then drop the
        # tempdir.
        for patcher in reversed(self._patches):
            patcher.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Helpers.
    # ------------------------------------------------------------------ #

    def _dispatch(
        self,
        method: str,
        params: dict[str, Any] | None,
    ) -> JsonRpcResponse:
        """Dispatch a single request and return the resulting response."""
        request = JsonRpcRequest(
            method=method,
            params=params,
            id=1,
            is_notification=False,
        )
        result = asyncio.run(
            dispatch(
                request,
                registry=self.registry,
                lifecycle=self.lifecycle,
                log=lambda _msg: None,
            )
        )
        # The dispatcher's contract guarantees a JsonRpcResponse for
        # every non-notification request; assert here so that any test
        # failure points at the rule rather than at the dispatcher.
        assert isinstance(result, JsonRpcResponse), (
            f"dispatcher returned {type(result).__name__} for {method!r}; expected JsonRpcResponse"
        )
        return result

    @staticmethod
    def _result(resp: JsonRpcResponse) -> Any:
        """Return ``resp.result``, asserting it is a success response."""
        assert not isinstance(resp.error, JsonRpcError), (
            f"expected success response, got error {resp.error.code}: {resp.error.message}"
        )
        return resp.result

    @staticmethod
    def _error(resp: JsonRpcResponse) -> JsonRpcError:
        """Return ``resp.error``, asserting it is an error response."""
        assert isinstance(resp.error, JsonRpcError), f"expected error response, got success result {resp.result!r}"
        return resp.error

    # ------------------------------------------------------------------ #
    # Rules: create.
    # ------------------------------------------------------------------ #

    @rule(name=_name_text, python_version=_python_versions)
    def create(self, name: str, python_version: str) -> None:
        """Environment endpoints behave as a model.

        Drive ``environment/create`` with a name from the safe-text
        strategy and a python_version drawn from the available /
        unavailable mix. Branch on the oracle to predict the outcome:

        * Name conflict (irrespective of version): expect
          ``environment-name-conflict``.
        * Otherwise unavailable version: expect
          ``python-version-unavailable``.
        * Otherwise: expect a success response and add the new id to
          the oracle.
        """
        resp = self._dispatch(
            "environment/create",
            {"name": name, "python_version": python_version},
        )

        name_conflict = any(entry["name"] == name for entry in self.oracle.values())
        python_unavailable = python_version not in _AVAILABLE_PY_VERSIONS

        if name_conflict:
            err = self._error(resp)
            assert err.code == _ENV_NAME_CONFLICT, (
                f"expected ENVIRONMENT_NAME_CONFLICT for duplicate name {name!r}; got code {err.code}"
            )
            assert isinstance(err.data, dict)
            violations = err.data.get("violations")
            assert isinstance(violations, list)
            assert "name-conflict" in violations
            # Every violated rule appears in data.
            if python_unavailable:
                assert "python-version-available" in violations
            return

        if python_unavailable:
            err = self._error(resp)
            assert err.code == _PY_VERSION_UNAVAILABLE
            assert isinstance(err.data, dict)
            violations = err.data.get("violations")
            assert isinstance(violations, list)
            assert "python-version-available" in violations
            return

        # Successful create: every required schema field is present.
        result = self._result(resp)
        assert isinstance(result, dict)
        for field in (
            "id",
            "name",
            "python_version",
            "interpreter_path",
            "installed_packages",
            "extra",
        ):
            assert field in result, f"successful create result missing {field!r}; got keys {sorted(result)}"
        assert isinstance(result["installed_packages"], list)
        assert isinstance(result["extra"], dict)

        new_id = result["id"]
        assert isinstance(new_id, str)
        assert new_id
        assert new_id not in self.oracle, f"fallback emitted duplicate id {new_id!r}"
        assert result["name"] == name
        assert result["python_version"] == python_version

        self.oracle[new_id] = {
            "name": name,
            "python_version": python_version,
        }

    @rule(data=st.data())
    def create_violation_precedence(self, data: st.DataObject) -> None:
        """Violation-precedence sub-property.

        When a ``create`` request violates both the name-uniqueness rule
        and the python-version-available rule, the WSP error code is
        ``environment-name-conflict`` (precedence) and the
        ``data.violations`` field lists both violated rules.
        """
        assume(self.oracle)
        existing_name = data.draw(st.sampled_from(sorted({entry["name"] for entry in self.oracle.values()})))
        resp = self._dispatch(
            "environment/create",
            {
                "name": existing_name,
                "python_version": _UNAVAILABLE_PY_VERSION,
            },
        )
        err = self._error(resp)
        assert err.code == _ENV_NAME_CONFLICT, (
            f"name-conflict must take precedence over python-version-available; got code {err.code}"
        )
        assert isinstance(err.data, dict)
        violations = err.data.get("violations")
        assert isinstance(violations, list)
        assert "name-conflict" in violations
        assert "python-version-available" in violations
        # No environment was persisted by the failed
        # create. The list_matches_oracle_and_is_idempotent invariant
        # runs immediately after this rule and verifies that the
        # ``environment/list`` response still agrees with the oracle.

    # ------------------------------------------------------------------ #
    # Rules: get.
    # ------------------------------------------------------------------ #

    @rule(data=st.data())
    def get_known(self, data: st.DataObject) -> None:
        """Get on a known id returns full details."""
        assume(self.oracle)
        env_id = data.draw(st.sampled_from(sorted(self.oracle.keys())))
        resp = self._dispatch("environment/get", {"id": env_id})
        result = self._result(resp)
        assert isinstance(result, dict)
        # Details object carries every required field
        # including the always-present ``extra``.
        for field in (
            "id",
            "name",
            "python_version",
            "interpreter_path",
            "installed_packages",
            "extra",
        ):
            assert field in result, f"get details missing {field!r}; got keys {sorted(result)}"
        assert result["id"] == env_id
        oracle_entry = self.oracle[env_id]
        assert result["name"] == oracle_entry["name"]
        assert result["python_version"] == oracle_entry["python_version"]
        assert isinstance(result["installed_packages"], list)
        # Each package has ``name`` and ``version``.
        for pkg in result["installed_packages"]:
            assert isinstance(pkg, dict)
            assert isinstance(pkg.get("name"), str)
            assert isinstance(pkg.get("version"), str)
        assert isinstance(result["extra"], dict)

    @rule(env_id=_arbitrary_id)
    def get_unknown(self, env_id: str) -> None:
        """Get on an unknown id returns environment-not-found."""
        assume(env_id not in self.oracle)
        oracle_before = dict(self.oracle)
        resp = self._dispatch("environment/get", {"id": env_id})
        err = self._error(resp)
        assert err.code == _ENV_NOT_FOUND, (
            f"expected ENVIRONMENT_NOT_FOUND for unknown id {env_id!r}; got code {err.code}"
        )
        # The requested id is echoed in data.
        assert isinstance(err.data, dict)
        assert err.data.get("id") == env_id
        # State unchanged.
        assert self.oracle == oracle_before

    # ------------------------------------------------------------------ #
    # Rules: delete.
    # ------------------------------------------------------------------ #

    @rule(data=st.data())
    def delete_known(self, data: st.DataObject) -> None:
        """Delete on a known id removes it."""
        assume(self.oracle)
        env_id = data.draw(st.sampled_from(sorted(self.oracle.keys())))
        resp = self._dispatch("environment/delete", {"id": env_id})
        result = self._result(resp)
        assert isinstance(result, dict)
        # Ack contains at minimum the deleted id.
        assert result.get("id") == env_id
        del self.oracle[env_id]

        # Subsequent get returns environment-not-found.
        followup = self._dispatch("environment/get", {"id": env_id})
        followup_err = self._error(followup)
        assert followup_err.code == _ENV_NOT_FOUND

    @rule(env_id=_arbitrary_id)
    def delete_unknown(self, env_id: str) -> None:
        """Delete on an unknown id is a no-op error."""
        assume(env_id not in self.oracle)
        oracle_before = dict(self.oracle)
        resp = self._dispatch("environment/delete", {"id": env_id})
        err = self._error(resp)
        assert err.code == _ENV_NOT_FOUND
        # State unchanged.
        assert self.oracle == oracle_before

    # ------------------------------------------------------------------ #
    # Rules: execute.
    # ------------------------------------------------------------------ #

    @rule(data=st.data())
    def execute_known(self, data: st.DataObject) -> None:
        """Successful execute result has the right shape."""
        assume(self.oracle)
        env_id = data.draw(st.sampled_from(sorted(self.oracle.keys())))
        resp = self._dispatch(
            "environment/execute",
            {"id": env_id, "argv": ["echo", "hello"]},
        )
        result = self._result(resp)
        assert isinstance(result, dict)
        # Integer exit_code, string stdout/stderr.
        assert isinstance(result.get("exit_code"), int)
        assert not isinstance(result.get("exit_code"), bool)
        assert isinstance(result.get("stdout"), str)
        assert isinstance(result.get("stderr"), str)

    @rule(env_id=_arbitrary_id)
    def execute_unknown(self, env_id: str) -> None:
        """Execute on an unknown id is environment-not-found."""
        assume(env_id not in self.oracle)
        oracle_before = dict(self.oracle)
        resp = self._dispatch(
            "environment/execute",
            {"id": env_id, "argv": ["echo", "hi"]},
        )
        err = self._error(resp)
        assert err.code == _ENV_NOT_FOUND
        assert self.oracle == oracle_before

    @rule(env_id=_arbitrary_id)
    def execute_empty_argv(self, env_id: str) -> None:
        """Empty argv -> -32602."""
        oracle_before = dict(self.oracle)
        resp = self._dispatch(
            "environment/execute",
            {"id": env_id, "argv": []},
        )
        err = self._error(resp)
        assert err.code == _INVALID_PARAMS
        # No command was launched and the oracle is unchanged.
        assert self.oracle == oracle_before

    @rule(env_id=_arbitrary_id)
    def execute_non_string_argv(self, env_id: str) -> None:
        """Non-string argv element -> -32602."""
        oracle_before = dict(self.oracle)
        # ``123`` is not a string; the validator rejects this with the
        # ``argv-element-type`` violation.
        resp = self._dispatch(
            "environment/execute",
            {"id": env_id, "argv": ["echo", 123]},
        )
        err = self._error(resp)
        assert err.code == _INVALID_PARAMS
        assert self.oracle == oracle_before

    # ------------------------------------------------------------------ #
    # Invariants: list-shape, ordering, idempotence, oracle agreement.
    # ------------------------------------------------------------------ #

    @invariant()
    def list_matches_oracle_and_is_idempotent(self) -> None:
        """List invariants.

        Two consecutive ``environment/list`` calls return identical
        arrays, the array is
        ordered by ascending ``id``, and the set of returned ids
        equals the oracle. The summary fields ``id``, ``name`` and
        ``python_version`` per entry agree with the oracle, which
        captures the ``create`` -> ``list`` reflection
        and the ``delete`` -> ``list`` removal.
        """
        first = self._result(self._dispatch("environment/list", {}))
        second = self._result(self._dispatch("environment/list", {}))

        assert isinstance(first, list)
        assert first == second, (
            f"two consecutive environment/list calls returned different results: {first!r} != {second!r}"
        )

        ids = [env["id"] for env in first]
        # Deterministic ascending-id ordering.
        assert ids == sorted(ids), f"environment/list ids {ids!r} are not ordered ascending"

        # Set agreement with the oracle.
        assert set(ids) == set(self.oracle.keys()), (
            f"list ids {sorted(ids)!r} do not match oracle {sorted(self.oracle)!r}"
        )

        # Per-entry summary fields agree with the oracle.
        for env in first:
            oracle_entry = self.oracle[env["id"]]
            assert env["name"] == oracle_entry["name"]
            assert env["python_version"] == oracle_entry["python_version"]


# Keep individual scenarios short enough that the whole test runs in a
# few seconds across hatch-test's parallel matrix while still
# exercising every rule in interesting combinations.
EnvironmentModelMachine.TestCase.settings = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[
        # The rule machine writes to a per-instance tempdir, which
        # Hypothesis flags as slow on cold filesystem runs; the
        # property under test is correctness, not throughput.
        HealthCheck.too_slow,
    ],
)


# pytest discovers the unittest.TestCase auto-generated by Hypothesis
# when it is bound to a module-level name beginning with ``Test``.
TestEnvironmentModel = EnvironmentModelMachine.TestCase
