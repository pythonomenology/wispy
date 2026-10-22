"""Entry point for ``python -m wispy --config <path>``.

Loads a WSP Config_File, builds a :class:`~wispy.registry.HandlerRegistry`
from its entries plus the built-in ``initialize``/``shutdown`` handlers,
and runs the stdio server. Config-time failures are surfaced on stderr
in the form ``wispy: config error: <path>: <reason>`` and the process
exits with status 1 *before* any transport is constructed.

A request for a defined WSP method that the
loaded Config_File does not configure surfaces as ``-32601`` from the
dispatcher's normal method-not-found path; this entry point is
responsible only for registering whatever the Config_File asks for
plus the two lifecycle methods.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wispy.config import ConfigError, load_config, make_handler
from wispy.endpoints import PROTOCOL_VERSION, Capabilities
from wispy.errors import DuplicateRegistrationError
from wispy.registry import HandlerRegistry
from wispy.server import run_stdio

if TYPE_CHECKING:
    from collections.abc import Sequence


def _config_error(path: Path, reason: str) -> int:
    """Write the conventional config-error line to stderr and return 1.

    Centralised so every config-time exit point produces an identical
    line shape: ``wispy: config error: <path>: <reason>\\n``.
    """
    sys.stderr.write(f"wispy: config error: {path}: {reason}\n")
    return 1


def _build_registry(
    entries: Sequence[Any],
    config_path: Path,
) -> HandlerRegistry | int:
    """Build the registry, returning an exit code on failure.

    The built-in ``initialize`` and ``shutdown`` handlers are bound
    first, so a Config_File that names ``initialize`` or ``shutdown``
    (which it must not, since those are reserved by the lifecycle
    layer) surfaces as a duplicate-registration config error rather
    than silently overriding the built-ins.

    """
    registry = HandlerRegistry()

    def _initialize(_params: Any) -> Any:
        return Capabilities(
            methods=tuple(registry.methods()),
            protocol_version=PROTOCOL_VERSION,
        )

    def _shutdown(_params: Any) -> None:
        return None

    # The built-ins are registered first so duplicates from the
    # Config_File are reported via DuplicateRegistrationError rather
    # than silently shadowing the lifecycle handlers.
    registry.register("initialize", _initialize)
    registry.register("shutdown", _shutdown)

    for entry in entries:
        try:
            handler = make_handler(entry)
        except ConfigError as exc:
            return _config_error(config_path, str(exc))
        try:
            registry.register(entry.method, handler)
        except DuplicateRegistrationError as exc:
            return _config_error(config_path, str(exc))

    return registry


def main(argv: list[str] | None = None) -> int:
    """Parse ``--config PATH`` and run the WSP stdio server.

    Returns the desired process exit status; ``sys.exit`` is invoked
    only when this module is run as ``__main__``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m wispy",
        description=("Run the wispy WSP server over stdio with handlers loaded from a Config_File."),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to a TOML or JSON Config_File.",
    )
    args = parser.parse_args(argv)
    config_path: Path = args.config

    # Every config-time rejection exits with
    # a non-zero status BEFORE the transport is constructed.
    try:
        entries = load_config(config_path)
    except ConfigError as exc:
        return _config_error(config_path, str(exc))

    built = _build_registry(entries, config_path)
    if isinstance(built, int):
        return built
    registry = built

    return asyncio.run(run_stdio(registry))


if __name__ == "__main__":
    sys.exit(main())
