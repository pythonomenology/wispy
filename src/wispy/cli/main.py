"""WSP_CLI argparse entry point.

This module wires up the user-facing ``wsp`` command. The user-facing
surface is:

* :func:`build_parser` -- builds the argparse layout dynamically from
  the WSP method registry in :mod:`wispy.endpoints` so subcommands and
  parameter flags stay in lock-step with the protocol surface.
* :func:`merge_params` -- merges ``--params-json -`` JSON with explicit
  ``--<param>`` flag values, with flags shadowing JSON entries.
* :class:`ExitCode` -- the canonical exit-code constants from the
  design's table.
* :func:`main` -- the argv entry point referenced by
  ``[project.scripts] wsp``. Parses arguments, enforces the
  ``--tool`` / ``--config`` mutual-exclusion rule,
  and routes to one of three execution modes:

  1. ``--tool ARGV...``: launch the named child Workflow_Tool
     subprocess and route the call through :class:`WspClient`.
  2. ``--config PATH``: launch ``python -m wispy --config PATH`` as a
     child and route the call through :class:`WspClient`.
  3. Neither flag: run the built-in fallback Workflow_Tool
     **in-process**. The design narrates a ``--serve-fallback``
     self-spawn for uniformity with the subprocess-based modes; we
     deliberately deviate and call the dispatcher directly against
     :func:`make_fallback_registry` because (a) it still satisfies
     the requirement that the fallback is invoked when no
     target/config is given, (b) it avoids a recursive
     ``python -m wispy.cli`` step that adds latency without any
     protocol benefit, and (c) the
     dispatcher and lifecycle code paths are identical regardless of
     transport. Behaviour-wise the in-process path is indistinguishable
     from the subprocess path for the user.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from wispy.cli._exit_codes import ExitCode
from wispy.cli.client import _RESPONSE_TIMEOUT_S, WspClient, _result_from_error_response
from wispy.cli.fallback import make_fallback_registry
from wispy.config import configured_timeout
from wispy.dispatcher import ExitDispatch, dispatch
from wispy.endpoints import (
    PROTOCOL_VERSION,
    Capabilities,
    CreateEnvironmentParams,
    DeleteEnvironmentParams,
    ExecuteParams,
    GetEnvironmentParams,
)
from wispy.lifecycle import LifecycleManager
from wispy.protocol import _UNSET, JsonRpcRequest, JsonRpcResponse, _json_default

__all__ = ["ExitCode", "build_parser", "main", "merge_params"]


# Maximum size of a ``--params-json -`` stdin payload. Anything larger
# is rejected with a usage error so we never buffer unbounded input.
_MAX_STDIN_BYTES = 1024 * 1024  # 1 MiB

# Slack added to a Config_File entry's own ``timeout`` when sizing how
# long to wait for its response, so the child's timeout error (which
# names the command) reaches the user before ours does.
_CONFIG_TIMEOUT_SLACK_S = 5.0


# Method names exposed as CLI subcommands. Excludes lifecycle methods
# (``initialize``, ``shutdown``) because the CLI wraps those around
# every call automatically -- they are not user-facing.
_CLI_METHODS: tuple[str, ...] = (
    "environment/list",
    "environment/create",
    "environment/get",
    "environment/delete",
    "environment/execute",
)

# Mapping from WSP method name to the dataclass that defines its
# normalized parameter shape. ``environment/list`` takes no params, so
# it has no entry here -- the parser builder will simply not add any
# ``--<param>`` flags for that subcommand.
_METHOD_PARAMS_CLASS: dict[str, type] = {
    "environment/create": CreateEnvironmentParams,
    "environment/get": GetEnvironmentParams,
    "environment/delete": DeleteEnvironmentParams,
    "environment/execute": ExecuteParams,
}


def _env_pair(value: str) -> tuple[str, str]:
    """argparse ``type`` for ``--env KEY=VALUE``."""
    key, sep, val = value.partition("=")
    if not sep or not key:
        msg = f"expected KEY=VALUE, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return key, val


# Params fields that are not plain strings need their own flag shape.
# ``argv`` takes every remaining argument (so the command may contain
# its own ``-flags``) and must therefore come last; ``env`` may be
# repeated and is collected into a dict by ``_flag_params_for_method``.
_FIELD_FLAG_KWARGS: dict[str, dict[str, Any]] = {
    "argv": {
        "nargs": argparse.REMAINDER,
        "metavar": "ARG",
        "help": "Command to run; must be the last flag (everything after it is the command).",
    },
    "env": {
        "action": "append",
        "type": _env_pair,
        "metavar": "KEY=VALUE",
        "help": "Environment variable for the command; may be repeated.",
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the WSP_CLI argument parser.

    The parser carries the top-level ``--tool`` / ``--config`` flags
    plus one subparser per user-facing WSP method. Each subparser
    exposes one ``--<param>`` flag per field on the method's params
    dataclass and a ``--params-json`` option for reading JSON from
    stdin (with ``-`` as the value).

    The exclusivity check between ``--tool`` and ``--config`` is
    enforced by :func:`main` rather than by an argparse mutex group:
    ``--tool`` uses ``argparse.REMAINDER`` so it can capture an
    arbitrary child argv, and ``REMAINDER`` interacts poorly with
    ``add_mutually_exclusive_group``. For the same reason the
    subcommand ends up inside ``--tool``'s capture; :func:`main` splits
    it back out (see :func:`_split_tool_argv`).
    """
    parser = argparse.ArgumentParser(prog="wsp")
    parser.add_argument(
        "--tool",
        nargs=argparse.REMAINDER,
        default=None,
        help=(
            "Argv of the target Workflow_Tool's WSP server, followed by the method subcommand: "
            "wsp --tool mytool serve environment/list. The child argv ends at the first "
            "argument naming a method."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Path to a Config_File to launch instead of a Workflow_Tool. Mutually exclusive with --tool."),
    )
    parser.add_argument(
        "--params-json",
        dest="params_json",
        default=None,
        help=("Read JSON params from stdin (use '-' as the value). May also be supplied after the subcommand name."),
    )

    # ``required=False`` lets users invoke ``wsp --help`` without
    # supplying a method; main() validates that a method is present
    # for any non-help invocation.
    subparsers = parser.add_subparsers(dest="method", required=False)
    for method in _CLI_METHODS:
        sp = subparsers.add_parser(method, help=f"Invoke WSP method {method}.")
        # Allow ``--params-json -`` after the subcommand name as well,
        # so users do not have to remember which side of the
        # subcommand the flag goes on.
        # ``SUPPRESS`` matters: a ``None`` default here would overwrite
        # a top-level ``--params-json -`` given before the subcommand.
        sp.add_argument(
            "--params-json",
            dest="params_json",
            default=argparse.SUPPRESS,
        )
        params_cls = _METHOD_PARAMS_CLASS.get(method)
        if params_cls is not None and is_dataclass(params_cls):
            for f in fields(params_cls):
                # Convert ``snake_case`` field names into the
                # conventional ``--kebab-case`` flag spelling, but
                # keep the snake_case name on the ``args`` namespace
                # so it round-trips into JSON-RPC params unchanged.
                sp.add_argument(
                    f"--{f.name.replace('_', '-')}",
                    dest=f.name,
                    **_FIELD_FLAG_KWARGS.get(f.name, {}),
                )
    return parser


def _read_stdin_json(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Read up to 1 MiB from stdin and parse it as JSON.

    Calls :meth:`argparse.ArgumentParser.error` (which exits with
    status 2, matching :data:`ExitCode.USAGE_OR_UNSUPPORTED`) on cap
    exceeded, malformed UTF-8, malformed JSON, or a non-object value.
    Returns an empty dict for empty input so callers can treat
    "stdin had nothing to say" as "no params provided via JSON."
    """
    # Read one byte past the cap so we can detect overflow without
    # buffering more than a tiny constant of slack.
    raw = sys.stdin.buffer.read(_MAX_STDIN_BYTES + 1)
    if len(raw) > _MAX_STDIN_BYTES:
        parser.error(f"--params-json - cap exceeded ({_MAX_STDIN_BYTES} bytes)")
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(f"--params-json - is not valid JSON: {exc}")
    if not isinstance(value, dict):
        parser.error("--params-json - must be a JSON object")
    return value  # type: ignore[no-any-return]


def merge_params(stdin_params: dict[str, Any], flag_params: dict[str, Any]) -> dict[str, Any]:
    """Merge stdin-derived and flag-derived params.

    Start with the JSON document supplied via
    ``--params-json -`` and overlay flag values for keys that were
    actually provided on the CLI. ``None`` flag values (i.e. flags
    the user did not pass) are treated as "not provided" and do not
    shadow the stdin entry, so users can JSON-supply a value and have
    it survive even when the corresponding flag exists on the parser.

    The returned dict contains exactly the union of stdin keys and
    explicitly-set flag keys -- no extras.
    """
    merged: dict[str, Any] = dict(stdin_params)
    for k, v in flag_params.items():
        if v is None:
            continue
        merged[k] = v
    return merged


def _flag_params_for_method(args: argparse.Namespace, method: str) -> dict[str, Any]:
    """Extract explicitly-set ``--<param>`` flag values for ``method``.

    Walks the params dataclass for ``method`` (if any) and pulls
    matching attributes off the parsed namespace. Attributes left at
    their default of ``None`` are skipped so :func:`merge_params` can
    treat them as "flag not provided" and let stdin-supplied values
    survive.
    """
    flag_params: dict[str, Any] = {}
    params_cls = _METHOD_PARAMS_CLASS.get(method)
    if params_cls is None or not is_dataclass(params_cls):
        return flag_params
    for f in fields(params_cls):
        value = getattr(args, f.name, None)
        if value is None:
            continue
        if f.name == "env":
            # ``--env`` is an ``append`` of (KEY, VALUE) pairs.
            value = dict(value)
        flag_params[f.name] = value
    return flag_params


def _build_params(
    args: argparse.Namespace,
    method: str,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    """Build the JSON-RPC params dict for the requested ``method``.

    Reads stdin (when ``--params-json -`` was given) and overlays
    explicit flag values. Returns ``{}`` for params-less methods so
    callers can pass it through unchanged.
    """
    stdin_params: dict[str, Any] = {}
    if args.params_json == "-":
        stdin_params = _read_stdin_json(parser)
    flag_params = _flag_params_for_method(args, method)
    return merge_params(stdin_params, flag_params)


def _config_response_timeout(config_path: str, method: str) -> float:
    """How long to wait for ``method``'s response from a ``--config`` child.

    The default per-response timeout, or longer when the Config_File
    gives ``method`` a ``timeout`` that would not fit in it.
    """
    entry_timeout = configured_timeout(Path(config_path), method)
    if entry_timeout is None:
        return _RESPONSE_TIMEOUT_S
    return max(_RESPONSE_TIMEOUT_S, entry_timeout + _CONFIG_TIMEOUT_SLACK_S)


async def _run_with_client(
    argv: list[str],
    method: str,
    params: dict[str, Any],
    *,
    response_timeout: float = _RESPONSE_TIMEOUT_S,
) -> int:
    """Run a single WSP call through :class:`WspClient` over a child."""
    async with WspClient(argv, response_timeout=response_timeout) as client:
        result = await client.call(method, params)
    if result.exit_code == ExitCode.SUCCESS:
        # ``result.value`` is the JSON-RPC ``result`` payload from the
        # child server, already JSON-serializable. Use ensure_ascii=False
        # so non-ASCII output round-trips faithfully, and append a
        # trailing newline to match standard CLI tool conventions.
        sys.stdout.write(json.dumps(result.value, ensure_ascii=False) + "\n")
    return result.exit_code


async def _run_fallback_inprocess(method: str, params: dict[str, Any]) -> int:
    """Run a single WSP call against the in-process fallback registry.

    Builds a fresh :class:`HandlerRegistry` and
    :class:`LifecycleManager` per invocation -- consistent with the
    subprocess modes, where each child process is also fresh -- and
    feeds the request straight through :func:`dispatch`. The lifecycle
    is pre-advanced to ``INITIALIZED`` so non-``initialize`` methods
    are admitted without a separate handshake.
    """
    registry = make_fallback_registry()
    lifecycle = LifecycleManager()
    # Pre-advance to INITIALIZED. This mirrors what the subprocess
    # modes accomplish via the ``initialize`` -> requested-method
    # handshake performed by ``WspClient.call``; here we skip the
    # round-trip because we already have the registry in hand.
    lifecycle.on_initialize_success(
        Capabilities(
            methods=tuple(registry.methods()),
            protocol_version=PROTOCOL_VERSION,
        )
    )

    request = JsonRpcRequest(
        method=method,
        params=params if params else None,
        id=1,
        is_notification=False,
    )
    response = await dispatch(
        request,
        registry=registry,
        lifecycle=lifecycle,
        log=lambda _msg: None,
    )

    # ``dispatch`` for a single non-notification request always returns
    # a JsonRpcResponse (or an ExitDispatch, but we never send
    # ``exit``). Anything else is an internal error.
    if isinstance(response, ExitDispatch) or not isinstance(response, JsonRpcResponse):
        sys.stderr.write("wsp: internal error: unexpected dispatch result\n")
        return ExitCode.GENERIC_ERROR

    if response.error is not _UNSET:
        # Reuse the same error -> exit-code mapping the subprocess
        # path uses, by reconstructing the wire-form error dict the
        # helper expects (``-32601`` -> 2, anything else -> 1, with a
        # human-readable rendering on stderr).
        err = response.error
        error_dict: dict[str, Any] = {
            "code": err.code,
            "message": err.message,
        }
        if err.data is not _UNSET:
            error_dict["data"] = err.data
        wire = {"jsonrpc": "2.0", "id": response.id, "error": error_dict}
        return _result_from_error_response(wire).exit_code

    # Success path. Encode wire models the same way the server does, so
    # a handler result that serializes over stdio also prints here.
    sys.stdout.write(json.dumps(response.result, ensure_ascii=False, default=_json_default) + "\n")
    return ExitCode.SUCCESS


def _split_tool_argv(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Split the method subcommand back out of ``--tool``'s capture.

    ``--tool`` is ``argparse.REMAINDER``, so it swallows the subcommand
    and its flags too. The child argv ends at the first argument that
    names a CLI method; everything from there on is parsed as the
    subcommand. Top-level flags given before ``--tool`` are kept.
    """
    tool: list[str] = args.tool
    split = next((i for i, arg in enumerate(tool) if arg in _CLI_METHODS), None)
    if split is None:
        return args
    sub_args = parser.parse_args(tool[split:])
    sub_args.tool = tool[:split]
    sub_args.config = args.config
    if sub_args.params_json is None:
        sub_args.params_json = args.params_json
    return sub_args


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and route to the appropriate mode.

    Three modes:

    * ``--tool ARGV...``: launch the named Workflow_Tool subprocess
      and route the WSP call through :class:`WspClient`.
    * ``--config PATH``: launch ``python -m wispy --config PATH`` as a
      child and route the WSP call through :class:`WspClient`.
    * Neither flag: run the built-in fallback in-process (see module
      docstring for the rationale behind this deliberate deviation
      from the design's ``--serve-fallback`` self-spawn).

    Returns the desired CLI exit status. Argparse-level errors (e.g.
    unknown subcommand) terminate via ``parser.error`` with status 2
    rather than returning here.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tool is not None and args.method is None:
        args = _split_tool_argv(parser, args)

    # The two flags are mutually exclusive. We
    # enforce it here rather than via ``add_mutually_exclusive_group``
    # because ``--tool`` uses ``argparse.REMAINDER`` and the two
    # interact poorly.
    if args.tool is not None and args.config is not None:
        parser.error("--tool and --config are mutually exclusive")
        return ExitCode.USAGE_OR_UNSUPPORTED  # pragma: no cover

    if args.tool is not None and not args.tool:
        parser.error("--tool requires the argv of a WSP server before the method")
        return ExitCode.USAGE_OR_UNSUPPORTED  # pragma: no cover

    method = args.method
    if method is None:
        # No subcommand supplied. Argparse cannot enforce this with
        # ``required=True`` on the subparsers because we need to allow
        # ``wsp --help`` to print the top-level help; emit a usage
        # error here instead so the exit code is 2.
        parser.error("a WSP method subcommand is required")
        return ExitCode.USAGE_OR_UNSUPPORTED  # pragma: no cover

    params = _build_params(args, method, parser)

    try:
        if args.tool is not None:
            # ``args.tool`` is the REMAINDER list, i.e. everything
            # after ``--tool``. Pass it straight to the client as the
            # child argv.
            return asyncio.run(_run_with_client(args.tool, method, params))
        if args.config is not None:
            # Spawn ourselves in config-server mode. ``sys.executable``
            # ensures we use the same interpreter the user invoked
            # ``wsp`` with, which matters in venvs and multi-Python
            # systems.
            child_argv = [
                sys.executable,
                "-m",
                "wispy",
                "--config",
                args.config,
            ]
            return asyncio.run(
                _run_with_client(
                    child_argv,
                    method,
                    params,
                    response_timeout=_config_response_timeout(args.config, method),
                )
            )
        return asyncio.run(_run_fallback_inprocess(method, params))
    except (asyncio.TimeoutError, FileNotFoundError):
        # ``WspClient.__aenter__`` re-raises both; it has already
        # written a human-readable message to stderr, so we just
        # surface the generic-error exit code.
        return ExitCode.GENERIC_ERROR
    except Exception as exc:  # noqa: BLE001 - surface anything as exit 1
        # Defensive catch-all. Anything that escapes the mode
        # implementations is by definition unexpected; render it
        # without a traceback so the user sees a single-line error.
        sys.stderr.write(f"wsp: {exc}\n")
        return ExitCode.GENERIC_ERROR
