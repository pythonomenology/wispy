"""WSP server lifecycle finite-state machine.

This module implements the lifecycle FSM that gates request dispatch on
the WSP server's initialization state. The dispatcher consults
:meth:`LifecycleManager.admit` before invoking any handler; the result
tells the dispatcher whether to dispatch the request, emit a JSON-RPC
error, or terminate the process with a specific exit status.

State diagram (see ``design.md``)::

    UNINITIALIZED --initialize/success--> INITIALIZED
    UNINITIALIZED --any other method-->   UNINITIALIZED  (-32002)
    UNINITIALIZED --exit notification-->  EXITED         (status 1)

    INITIALIZED   --any registered method--> INITIALIZED (success/domain err)
    INITIALIZED   --initialize-->            INITIALIZED (-32600, caps kept)
    INITIALIZED   --shutdown/success-->      SHUTTING_DOWN
    INITIALIZED   --exit notification-->     EXITED      (status 1)

    SHUTTING_DOWN --any non-exit message--> SHUTTING_DOWN (-32600)
    SHUTTING_DOWN --exit notification-->    EXITED        (status 0)

The FSM itself does not perform I/O and never raises. ``admit`` returns
one of three :data:`AdmitDecision` variants:

* :class:`Allow` -- the dispatcher should look up the handler and run it.
* :class:`RejectWith` -- the dispatcher should emit the carried
  :class:`~wispy.protocol.JsonRpcError` (and skip handler dispatch).
* :class:`Exit` -- the runtime should drain pending writes and exit the
  process with the carried status code.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Final

from wispy.errors import JsonRpcErrorCode
from wispy.protocol import JsonRpcError

if TYPE_CHECKING:
    # ``Capabilities`` lives in ``wispy.endpoints`` which currently has a
    # circular relationship with the lifecycle (the dispatcher wires
    # them together). Importing only under ``TYPE_CHECKING`` keeps the
    # runtime import graph acyclic while preserving precise typing.
    from wispy.endpoints import Capabilities


__all__ = [
    "ALLOW",
    "AdmitDecision",
    "Allow",
    "Exit",
    "LifecycleManager",
    "RejectWith",
    "ServerState",
]


# --------------------------------------------------------------------- #
# State enum and decision algebra.
# --------------------------------------------------------------------- #


class ServerState(Enum):
    """The lifecycle state of a :class:`LifecycleManager`.

    The four states correspond to the WSP server lifecycle nodes from
    ``design.md``: the server starts in
    :attr:`UNINITIALIZED`, advances to :attr:`INITIALIZED` after a
    successful ``initialize`` request, advances to :attr:`SHUTTING_DOWN`
    after a successful ``shutdown`` request, and reaches the terminal
    :attr:`EXITED` once the runtime has acknowledged an ``exit``
    notification.
    """

    UNINITIALIZED = auto()
    INITIALIZED = auto()
    SHUTTING_DOWN = auto()
    EXITED = auto()


@dataclass(frozen=True)
class Allow:
    """Admit decision: the dispatcher should run the registered handler.

    Carries no payload; the singleton :data:`ALLOW` is the canonical
    instance, but any ``Allow()`` is equal to any other.
    """


@dataclass(frozen=True)
class RejectWith:
    """Admit decision: emit the carried JSON-RPC error and skip dispatch.

    Used for the protocol-level rejections defined by the lifecycle FSM
    (``-32002`` for pre-initialize calls, ``-32600`` for re-initialize
    or post-shutdown traffic).
    """

    error: JsonRpcError


@dataclass(frozen=True)
class Exit:
    """Admit decision: terminate the process with the carried status.

    Produced only for ``exit`` notifications. The runtime is expected
    to drain pending writes before exiting.
    """

    status: int


# Algebraic type alias used in dispatcher signatures and tests. Python's
# union-of-dataclasses pattern stands in for a sealed type here.
AdmitDecision = Allow | RejectWith | Exit


# Canonical instance of :class:`Allow`. Allocating a single instance and
# returning it from :meth:`LifecycleManager.admit` avoids creating a new
# dataclass instance on every dispatched request.
ALLOW: Final[Allow] = Allow()


# --------------------------------------------------------------------- #
# Pre-built rejection responses.
# --------------------------------------------------------------------- #


def _server_not_initialized() -> JsonRpcError:
    """``-32002``: the server has not yet completed ``initialize``."""
    return JsonRpcError(
        code=int(JsonRpcErrorCode.SERVER_NOT_INITIALIZED),
        message="server not initialized",
    )


def _already_initialized() -> JsonRpcError:
    """``-32600``: a second ``initialize`` was received while initialized.

    The cached capabilities from the first successful ``initialize``
    are preserved unchanged; this error simply
    informs the client that re-initialization is not supported.
    """
    return JsonRpcError(
        code=int(JsonRpcErrorCode.INVALID_REQUEST),
        message="server already initialized",
    )


def _shutting_down() -> JsonRpcError:
    """``-32600``: a non-``exit`` message arrived after ``shutdown``."""
    return JsonRpcError(
        code=int(JsonRpcErrorCode.INVALID_REQUEST),
        message="server is shutting down",
    )


def _post_exit_invalid_request() -> JsonRpcError:
    """``-32600``: defensive rejection if any traffic reaches us post-EXIT.

    The runtime should have terminated before this branch is reached;
    the FSM still returns a structurally valid decision rather than
    raising.
    """
    return JsonRpcError(
        code=int(JsonRpcErrorCode.INVALID_REQUEST),
        message="server has exited",
    )


# --------------------------------------------------------------------- #
# LifecycleManager.
# --------------------------------------------------------------------- #


class LifecycleManager:
    """Tracks the WSP server lifecycle state and admits or rejects traffic.

    The manager is a small state machine with one piece of cached side
    state -- the :class:`Capabilities` object returned by the first
    successful ``initialize`` request. The dispatcher calls
    :meth:`admit` before doing anything else with each incoming message;
    after the dispatcher has executed the registered handler and
    obtained a result, it informs the manager via
    :meth:`on_initialize_success` or :meth:`on_shutdown_success` so the
    state advances on the right edges of the diagram.

    The manager performs no I/O and never raises during ``admit``.
    """

    state: ServerState
    capabilities: Capabilities | None

    def __init__(self) -> None:
        self.state = ServerState.UNINITIALIZED
        self.capabilities = None

    # ----------------------------------------------------------------- #
    # Admission: pure function over (state, method, is_notification).
    # ----------------------------------------------------------------- #

    def admit(self, method: str, is_notification: bool) -> AdmitDecision:  # noqa: FBT001 - boolean reflects message kind
        """Decide what to do with an incoming request.

        Args:
            method: The JSON-RPC method name.
            is_notification: ``True`` iff the message had no ``id``
                member (i.e. it is a JSON-RPC notification).

        Returns:
            An :data:`AdmitDecision`:

            * :data:`ALLOW` -- the dispatcher should look up the handler
              and run it.
            * :class:`RejectWith` -- the dispatcher should emit the
              carried error and skip dispatch.
            * :class:`Exit` -- the runtime should drain and exit the
              process with the carried status. Produced only for the
              ``exit`` notification.
        """
        # Defensive: if any traffic somehow arrives after EXITED, refuse
        # it rather than transitioning further. The runtime should have
        # terminated before this branch is reachable.
        if self.state is ServerState.EXITED:
            return RejectWith(_post_exit_invalid_request())

        # The ``exit`` notification is the only message that can move
        # the FSM to EXITED. Its status depends on whether ``shutdown``
        # was received first.
        if method == "exit" and is_notification:
            if self.state is ServerState.SHUTTING_DOWN:
                return Exit(0)
            return Exit(1)

        if self.state is ServerState.UNINITIALIZED:
            if method == "initialize":
                return ALLOW
            # Any other method before initialize completes: -32002.
            return RejectWith(_server_not_initialized())

        if self.state is ServerState.INITIALIZED:
            if method == "initialize":
                # A second initialize is rejected with
                # -32600 and the cached capabilities are preserved.
                return RejectWith(_already_initialized())
            return ALLOW

        # SHUTTING_DOWN. The exit-notification case was handled above;
        # everything else (including a non-notification ``exit``
        # request) is rejected with -32600.
        return RejectWith(_shutting_down())

    # ----------------------------------------------------------------- #
    # Edge transitions invoked by the dispatcher after handler success.
    # ----------------------------------------------------------------- #

    def on_initialize_success(self, caps: Capabilities) -> None:
        """Record a successful ``initialize`` and cache the capabilities.

        Idempotent guard: if the manager is already INITIALIZED, the
        cached capabilities are left untouched. This preserves the
        invariant that the very first successful
        ``initialize`` defines the capabilities for the lifetime of the
        server, even if some caller invokes this method redundantly.
        """
        if self.state is ServerState.INITIALIZED:
            return
        self.state = ServerState.INITIALIZED
        self.capabilities = caps

    def snapshot(self) -> LifecycleManager:
        """Return a detached copy of the manager's current state.

        The runtime admits a non-lifecycle message against a snapshot
        taken when the message was read, so a task that first runs
        after a later ``shutdown`` or ``exit`` still sees the state its
        message arrived in. Such messages never advance the FSM, so
        nothing written to the copy needs to reach the original.
        """
        copy = LifecycleManager()
        copy.state = self.state
        copy.capabilities = self.capabilities
        return copy

    def on_shutdown_success(self) -> None:
        """Record a successful ``shutdown``."""
        self.state = ServerState.SHUTTING_DOWN

    def on_exit(self) -> int:
        """Transition to EXITED and report the desired process status.

        Returns ``0`` if ``shutdown`` was received first, else ``1``.
        Always advances state to
        :attr:`ServerState.EXITED`.
        """
        status = 0 if self.state is ServerState.SHUTTING_DOWN else 1
        self.state = ServerState.EXITED
        return status
