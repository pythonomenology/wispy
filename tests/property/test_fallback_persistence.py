"""Property test for fallback Workflow_Tool persistence.

The built-in fallback Workflow_Tool is supposed to persist its set of
Environments across WSP_CLI invocations: every Environment created in
one invocation MUST remain listable, retrievable, executable, and
deletable in every subsequent invocation against the same state
directory until it is removed via ``environment/delete``.

This module models that property as a Hypothesis
:class:`~hypothesis.stateful.RuleBasedStateMachine` whose rules each
simulate a fresh CLI invocation:

1. Each rule constructs a brand-new :class:`HandlerRegistry` via
   :func:`~wispy.cli.fallback.make_fallback_registry`, pointing at a
   shared temp state directory. The previous registry is discarded;
   only on-disk state survives between rules. This is exactly what
   the spec calls "process restart" semantics.
2. Each rule performs at most one ``create`` or ``delete`` and runs
   ``list`` to verify the persisted set matches a shadow model.
3. The shadow model -- a plain ``dict[name, id]`` -- is the oracle:
   the persisted set of ``id`` values returned by ``environment/list``
   MUST always equal the multiset of ids in the shadow model.

``subprocess.run``, ``wispy.cli.fallback._create_venv``, and ``shutil.which``
are stubbed so each rule completes in milliseconds and never spawns a
real interpreter. The on-disk state machinery (``index.json``, the
file lock) is the system under test; the venv contents are not.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

# ``typing.override`` is available natively from Python 3.12. On 3.10 /
# 3.11 we fall back to a no-op decorator so the source still imports
# cleanly. pyrefly runs on 3.12 and picks the native import branch.
if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover - exercised only on 3.10 / 3.11

    def override(arg: Any) -> Any:
        return arg


from wispy.cli.fallback import make_fallback_registry
from wispy.dispatcher import dispatch
from wispy.endpoints import PROTOCOL_VERSION, Capabilities
from wispy.lifecycle import LifecycleManager
from wispy.protocol import _UNSET, JsonRpcRequest

if TYPE_CHECKING:
    from wispy.registry import HandlerRegistry

# --------------------------------------------------------------------- #
# Fakes for the subprocess / venv / PATH-resolution boundary.
#
# The fallback handler shells out to ``wispy.cli.fallback._create_venv`` to
# materialize a venv on disk and to ``shutil.which`` to confirm a
# ``pythonX.Y`` binary is on PATH. Real execution would be slow and
# would couple the test to the host's installed interpreter set; we
# stub all three call sites so the rules exercise only the persistence
# layer.
# --------------------------------------------------------------------- #


def _fake_subprocess_run(argv: Any, **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    """Stand in for :func:`subprocess.run` in the fallback handler.

    Returns a successful ``CompletedProcess`` regardless of input. The
    persistence property only invokes ``create``, ``delete``, and
    ``list``; none of those reach ``subprocess.run`` in practice, but
    we patch it for safety so any unintended invocation cannot spawn a
    real child or fail with ``FileNotFoundError``.
    """
    return subprocess.CompletedProcess(args=argv, returncode=0, stdout=b"", stderr=b"")


def _fake_create_venv(_interpreter: Any, target_dir: Any) -> None:
    """Stand in for :func:`wispy.cli.fallback._create_venv`.

    The real method writes a full venv tree under ``target_dir``. Our
    stub merely creates the directory so :func:`shutil.rmtree` (called
    by ``environment/delete``) has something to remove. The fallback
    handler never inspects the venv contents during create/list/delete,
    so an empty directory is sufficient.
    """
    Path(target_dir).mkdir(parents=True, exist_ok=True)


def _fake_which(cmd: Any, *_args: Any, **_kwargs: Any) -> str | None:
    """Stand in for :func:`shutil.which`.

    The fallback handler calls ``shutil.which("pythonX.Y")`` to enforce
    the ``python-version-available`` rule. Returning a non-``None``
    string tells the handler the requested interpreter is available.
    Any other lookup (rare in this test) returns ``None`` so unrelated
    PATH lookups still behave normally.
    """
    if isinstance(cmd, str) and cmd.startswith("python"):
        return f"/usr/bin/{cmd}"
    return None


# --------------------------------------------------------------------- #
# State machine.
# --------------------------------------------------------------------- #


# Names: non-empty, short, no NUL byte. The endpoint validator's
# rules ``name-required`` (non-empty) and ``name-length`` (<= 256
# chars) are both satisfied; we also exclude NUL bytes because they
# can confuse the JSON serializer on some platforms even though the
# WSP spec does not formally forbid them.
_env_names = st.text(min_size=1, max_size=20).filter(lambda s: s != "" and "\x00" not in s)


class FallbackPersistenceMachine(RuleBasedStateMachine):
    """Fallback Workflow_Tool persistence.

    Each rule is a fresh CLI invocation: a brand-new
    :class:`~wispy.registry.HandlerRegistry` is built from
    :func:`~wispy.cli.fallback.make_fallback_registry` against the
    same temp state directory, exercising the persistence path and
    nothing else. The shadow model maintained by the test (``self.envs``,
    a ``dict[name, id]``) is checked against the live
    ``environment/list`` response by the
    :meth:`list_matches_model` invariant after every step.
    """

    def __init__(self) -> None:
        super().__init__()
        # Per-test scratch directory. ``WISPY_STATE_DIR`` precedence is
        # exercised implicitly: ``make_fallback_registry`` is invoked
        # with an explicit ``state_root`` argument, which the
        # :class:`~wispy.cli.state.FallbackState` constructor honors
        # over any environment variable.
        self.tmpdir = Path(tempfile.mkdtemp(prefix="wsp-persist-"))
        # Shadow model: maps the env's user-visible ``name`` to the
        # ``id`` assigned by the handler. Both halves are useful: we
        # need ``id`` to perform deletes (the handler deletes by id)
        # and we need ``name`` to dedupe creates (the handler rejects
        # duplicates with ``-31002`` environment-name-conflict).
        self.envs: dict[str, str] = {}

    @override
    def teardown(self) -> None:
        """Remove the temp state directory at the end of each example."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ----------------------------------------------------------------- #
    # Per-rule helpers: build a fresh registry / lifecycle and dispatch.
    # ----------------------------------------------------------------- #

    def _fresh_registry(self) -> HandlerRegistry:
        """Build a fresh :class:`HandlerRegistry` for this rule.

        Each rule discards the previous registry and constructs a new
        one. This is the core of the property: any state observed in
        the next rule must have come from the on-disk index, not from
        live in-process state.
        """
        return make_fallback_registry(self.tmpdir)

    def _dispatch(
        self,
        registry: HandlerRegistry,
        method: str,
        params: Any,
    ) -> Any:
        """Dispatch one method call against a fresh registry/lifecycle.

        The lifecycle is short-circuited to ``INITIALIZED`` by calling
        :meth:`LifecycleManager.on_initialize_success` directly, which
        mirrors the post-``initialize`` state every WSP_CLI invocation
        ends up in before issuing the user's request.
        """
        lifecycle = LifecycleManager()
        lifecycle.on_initialize_success(
            Capabilities(
                methods=tuple(registry.methods()),
                protocol_version=PROTOCOL_VERSION,
            )
        )
        request = JsonRpcRequest(
            method=method,
            params=params,
            id=1,
            is_notification=False,
        )
        return asyncio.run(
            dispatch(
                request,
                registry=registry,
                lifecycle=lifecycle,
                log=lambda _msg: None,
            )
        )

    # ----------------------------------------------------------------- #
    # Rules: each one is a fresh invocation performing at most one op.
    # ----------------------------------------------------------------- #

    @rule(name=_env_names)
    def create(self, name: str) -> None:
        """Fallback Workflow_Tool persistence.

        Simulate a CLI invocation that performs one ``environment/create``.
        The shadow model is updated only on success (response carries a
        ``result``); name conflicts and any other errors leave the model
        untouched, matching the handler's "no partial creates" guarantee.
        """
        # Skip names already in the shadow model: those would surface
        # as ``-31002`` environment-name-conflict, which is correct
        # handler behavior but adds no information to this property.
        if name in self.envs:
            return
        registry = self._fresh_registry()
        resp = self._dispatch(
            registry,
            "environment/create",
            {"name": name, "python_version": "3.12"},
        )
        if resp is None or resp.result is _UNSET:
            # Error response (or notification, which we never send):
            # do not mutate the shadow.
            return
        # Success: the handler returned the details form, which carries
        # the ``id`` we will use for subsequent deletes / lookups.
        new_id = resp.result["id"]
        assert isinstance(new_id, str)
        assert new_id != ""
        self.envs[name] = new_id

    @rule(data=st.data())
    def delete(self, data: st.DataObject) -> None:
        """Fallback Workflow_Tool persistence.

        Simulate a CLI invocation that performs one
        ``environment/delete`` against an id known to our shadow model.
        The shadow drops the entry only when the handler reports
        success; a failed delete leaves both the persisted set and the
        shadow intact.
        """
        if not self.envs:
            return
        name = data.draw(st.sampled_from(sorted(self.envs.keys())))
        env_id = self.envs[name]
        registry = self._fresh_registry()
        resp = self._dispatch(registry, "environment/delete", {"id": env_id})
        # Success: response carries a ``result`` (the DeleteAck) and
        # ``error`` is unset. We update the shadow only in that case.
        if resp is not None and resp.error is _UNSET:
            del self.envs[name]

    # ----------------------------------------------------------------- #
    # Invariant: a fresh ``list`` invocation matches the shadow.
    # ----------------------------------------------------------------- #

    @invariant()
    def list_matches_model(self) -> None:
        """Fallback Workflow_Tool persistence.

        After every rule, build *another* fresh registry and call
        ``environment/list``. The set of ids in the response MUST equal
        the set of ids in the shadow model. ``environment/list``
        guarantees ascending-by-``id`` ordering; we
        compare sorted sequences so a bug that loses ordering is also
        caught.
        """
        registry = self._fresh_registry()
        resp = self._dispatch(registry, "environment/list", {})
        assert resp is not None
        assert resp.result is not _UNSET, f"environment/list returned an error: {resp.error!r}"
        ids_in_response = [entry["id"] for entry in resp.result]
        # Ascending id order. Verifying both the
        # ordering and the multiset content in one check keeps the
        # invariant tight without splitting it into two.
        assert ids_in_response == sorted(ids_in_response), (
            f"environment/list result is not sorted by ascending id: {ids_in_response!r}"
        )
        assert sorted(ids_in_response) == sorted(self.envs.values()), (
            "persisted set drifted from shadow model: "
            f"persisted={sorted(ids_in_response)!r} "
            f"shadow={sorted(self.envs.values())!r}"
        )


# --------------------------------------------------------------------- #
# Patch fixture and pytest binding.
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _patch_subprocess_and_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the subprocess / venv / PATH boundary for every test method.

    The fixture is module-scoped via ``autouse=True`` and applies to
    the unittest-style ``TestCase`` Hypothesis generates from the
    ``RuleBasedStateMachine``. Since the patches are installed once
    per test method (not per Hypothesis example), they remain in
    effect across every rule the state machine runs.
    """
    monkeypatch.setattr("wispy.cli.fallback.subprocess.run", _fake_subprocess_run)
    monkeypatch.setattr("wispy.cli.fallback._create_venv", _fake_create_venv)
    monkeypatch.setattr("wispy.cli.fallback.shutil.which", _fake_which)


# Hypothesis discovers the ``TestCase`` from the module-level binding;
# pytest then runs it as a unittest-style test class. The settings
# below cap example count for CI speed (each example runs many rules,
# so the effective coverage is high) and suppress the
# ``function_scoped_fixture`` health check because the autouse fixture
# above stays installed across every rule of every example.
TestFallbackPersistence = FallbackPersistenceMachine.TestCase
TestFallbackPersistence.settings = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
