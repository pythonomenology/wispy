"""Config_File loading and handler resolution (TOML/JSON).

This module reads a WSP server Config_File and translates it into a
list of :class:`ConfigEntry` records that the ``__main__`` bootstrap
later turns into :data:`~wispy.registry.Handler` callables via
:func:`make_handler`.

The schema is intentionally narrow:

* The top-level document is a table/object. Unknown top-level keys are
  ignored so that future server-wide settings can be added without
  breaking older configs.
* The ``handlers`` key is a table whose keys are WSP method names
  (validated against :data:`~wispy.endpoints.WSP_METHODS`) and whose
  values are *handler entries*.
* A handler entry must specify exactly one of:

  - ``import`` -- a dotted Python attribute path resolving to a callable.
  - ``command`` -- a non-empty array of strings forming an argv
    template. Curly-brace tokens like ``{name}`` are substituted from
    the request params at call time. The token ``{argv}`` is special:
    it splats a JSON array param into multiple argv positions.

* Subprocess entries may also carry a ``result`` mode and an optional
  ``template``. The mode tells wispy how to construct the WSP result
  from the subprocess's outcome:

  - ``"json"``     -- parse stdout as JSON; that is the result.
  - ``"template"`` -- render the ``template`` table with ``{key}``
    substitutions from the request params; that is the result.
    Requires ``template``.
  - ``"exec"``     -- result is ``{exit_code, stdout, stderr}`` taken
    directly from the subprocess.
  - ``"none"``     -- result is ``null``; success is exit code 0.

  Each WSP method has a sensible default mode so most callers do not
  need to specify ``result`` explicitly. See :data:`_DEFAULT_RESULT_MODE`.

* Subprocess entries may also carry a ``timeout``: the wall-clock limit,
  in seconds, for one run of the command (default 30).

* Any other key inside a handler entry is rejected.

All rejection paths raise :class:`ConfigError` with a message that
identifies the offending entry; the caller in ``__main__`` translates
these to ``wispy: config error: <path>: <reason>`` on stderr and
exits 1.

The Subprocess_Handler factory wraps the configured ``argv`` and
result strategy in an async coroutine that spawns the process per
request, closes the child's stdin immediately (CLIs adapted by this
flow do not read stdin), and constructs the WSP result per the
declared mode. Failures map to :class:`~wispy.errors.WspError` with
``EXECUTION_FAILED``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from wispy.registry import Handler

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

from wispy.endpoints import WSP_METHODS
from wispy.errors import WspError, WspErrorCode

# Default wall-clock timeout for one Subprocess_Handler invocation,
# overridable per entry with ``timeout``. Exceeding it is mapped to
# ``EXECUTION_FAILED``.
_SUBPROCESS_TIMEOUT = 30.0

# A handler entry that names both ``import`` and ``command`` has both
# keys set; this constant names the count for PLR2004 compliance.
_BOTH_KINDS_COUNT = 2


class ResultMode(str, Enum):
    """How wispy constructs a WSP method's result from a subprocess.

    See the module docstring for what each mode means.
    """

    JSON = "json"
    TEMPLATE = "template"
    EXEC = "exec"
    NONE = "none"


# The default ``result`` mode per WSP method. A Config_File entry that
# does not specify ``result`` falls back to this map. Methods whose
# default is :data:`ResultMode.TEMPLATE` still require an explicit
# ``template`` table (load_config will reject the entry otherwise).
_DEFAULT_RESULT_MODE: dict[str, ResultMode] = {
    "initialize": ResultMode.JSON,
    "shutdown": ResultMode.NONE,
    "environment/list": ResultMode.JSON,
    "environment/get": ResultMode.JSON,
    "environment/create": ResultMode.TEMPLATE,
    "environment/delete": ResultMode.TEMPLATE,
    "environment/execute": ResultMode.EXEC,
}


__all__ = [
    "ConfigEntry",
    "ConfigError",
    "PythonHandlerSpec",
    "ResultMode",
    "SubprocessHandlerSpec",
    "configured_timeout",
    "load_config",
    "make_handler",
]


@dataclass(frozen=True)
class PythonHandlerSpec:
    """A handler implemented as an in-process Python callable.

    ``import_path`` is a dotted path of the form ``pkg.module.attr``;
    the trailing component is resolved via ``getattr`` after importing
    the leading components as a module.
    """

    import_path: str


@dataclass(frozen=True)
class SubprocessHandlerSpec:
    """A handler implemented as an external command.

    Attributes:
        argv_template: Non-empty tuple of strings forming an argv
            template. Tokens like ``"{name}"`` are substituted from the
            request params at call time. The element ``"{argv}"`` is
            special: it splats a JSON array param into multiple argv
            positions (used by ``environment/execute``).
        result_mode: How to construct the WSP result from the
            subprocess's outcome. See :class:`ResultMode`.
        result_template: Required when ``result_mode`` is
            :data:`ResultMode.TEMPLATE`; a JSON-like value (dict, list,
            or scalar) whose strings are rendered with the same
            ``{key}`` substitution as ``argv_template``.
        timeout: Wall-clock limit, in seconds, for one run of the
            command.
    """

    argv_template: tuple[str, ...]
    result_mode: ResultMode
    result_template: Any = None
    timeout: float = _SUBPROCESS_TIMEOUT


@dataclass(frozen=True)
class ConfigEntry:
    """One Config_File handler binding."""

    method: str
    spec: PythonHandlerSpec | SubprocessHandlerSpec


class ConfigError(Exception):
    """Raised by :func:`load_config` and :func:`make_handler` on rejection.

    The message text is the human-readable reason that the
    ``__main__`` bootstrap concatenates into
    ``wispy: config error: <path>: <reason>`` before exiting with
    status 1.
    """


def load_config(path: Path) -> list[ConfigEntry]:
    """Read and validate a TOML or JSON Config_File.

    The extension match is case-insensitive (``.toml`` / ``.json``);
    any other extension is rejected. Unknown top-level keys are
    accepted, but unknown keys *inside* a handler entry are rejected.
    Each method name is validated against
    :data:`~wispy.endpoints.WSP_METHODS` so a typo surfaces here
    rather than at first request.
    """
    doc = _read_document(path)

    handlers = doc.get("handlers", {})
    if not isinstance(handlers, dict):
        msg = "'handlers' must be a table/object"
        raise ConfigError(msg)

    entries: list[ConfigEntry] = []
    for method, spec_obj in handlers.items():
        if method not in WSP_METHODS:
            msg = f"unknown WSP method {method!r}"
            raise ConfigError(msg)
        if not isinstance(spec_obj, dict):
            msg = f"handler entry for {method!r} must be a table/object"
            raise ConfigError(msg)
        entries.append(_load_entry(method, spec_obj))

    return entries


def configured_timeout(path: Path, method: str) -> float | None:
    """Return the ``timeout`` the Config_File at ``path`` sets for ``method``.

    Returns ``None`` when the file cannot be read, ``method`` has no
    entry, or the entry sets no valid ``timeout``. Unlike
    :func:`load_config` this neither validates the rest of the file nor
    imports ``import`` handlers, so the WSP_CLI can size its own
    response timeout without running the user's code; the child server
    still validates the file in full.
    """
    try:
        doc = _read_document(path)
    except ConfigError:
        return None
    handlers = doc.get("handlers")
    entry = handlers.get(method) if isinstance(handlers, dict) else None
    timeout = entry.get("timeout") if isinstance(entry, dict) else None
    return float(timeout) if _is_valid_timeout(timeout) else None


def _read_document(path: Path) -> dict[str, Any]:
    """Parse ``path`` as TOML or JSON by extension; the top level must be a table."""
    suffix = path.suffix.lower()
    if suffix == ".toml":
        try:
            with open(path, "rb") as f:
                doc = tomllib.load(f)
        except OSError as exc:
            msg = f"failed to read config file: {exc}"
            raise ConfigError(msg) from exc
        except tomllib.TOMLDecodeError as exc:
            msg = f"failed to parse TOML: {exc}"
            raise ConfigError(msg) from exc
    elif suffix == ".json":
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except OSError as exc:
            msg = f"failed to read config file: {exc}"
            raise ConfigError(msg) from exc
        except json.JSONDecodeError as exc:
            msg = f"failed to parse JSON: {exc}"
            raise ConfigError(msg) from exc
    else:
        msg = f"unsupported config extension {suffix!r}; expected .toml or .json"
        raise ConfigError(msg)

    if not isinstance(doc, dict):
        msg = "top-level config must be a table/object"
        raise ConfigError(msg)
    return doc


def _is_valid_timeout(value: Any) -> bool:
    """``True`` iff ``value`` is a positive number (and not a bool)."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value > 0


def _load_entry(method: str, spec_obj: dict[str, Any]) -> ConfigEntry:
    """Validate one handler entry and produce a :class:`ConfigEntry`.

    Splits the per-entry validation out of :func:`load_config` so the
    cyclomatic complexity of the loader stays tractable.
    """
    keys = set(spec_obj.keys())
    kinds = keys & {"import", "command"}
    if len(kinds) == 0:
        msg = f"handler entry for {method!r} must specify either 'import' or 'command'"
        raise ConfigError(msg)
    if len(kinds) == _BOTH_KINDS_COUNT:
        msg = f"handler entry for {method!r} must not specify both 'import' and 'command'"
        raise ConfigError(msg)

    if "import" in spec_obj:
        return ConfigEntry(method=method, spec=_load_python_spec(method, spec_obj))
    return ConfigEntry(method=method, spec=_load_subprocess_spec(method, spec_obj))


def _load_python_spec(method: str, spec_obj: dict[str, Any]) -> PythonHandlerSpec:
    """Validate the ``import``-shaped handler entry shape.

    Python handlers do not accept ``result`` or ``template`` -- those
    are subprocess-only knobs. Any other key is rejected.
    """
    unknown = set(spec_obj.keys()) - {"import"}
    if unknown:
        msg = (
            f"handler entry for {method!r} has unknown keys: "
            f"{sorted(unknown)!r} (Python handlers accept only 'import')"
        )
        raise ConfigError(msg)

    import_path = spec_obj["import"]
    if not isinstance(import_path, str) or not import_path:
        msg = f"handler entry for {method!r}: 'import' must be a non-empty string"
        raise ConfigError(msg)

    # Validate the dotted path resolves to a callable now;
    # make_handler re-resolves later, but surfacing the error
    # at startup is the whole point.
    try:
        _resolve_import(import_path)
    except ConfigError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"handler entry for {method!r}: failed to resolve {import_path!r}: {exc}"
        raise ConfigError(msg) from exc
    return PythonHandlerSpec(import_path=import_path)


def _load_subprocess_spec(method: str, spec_obj: dict[str, Any]) -> SubprocessHandlerSpec:
    """Validate the ``command``-shaped handler entry shape.

    Accepts the keys ``command`` (required), ``result`` (optional),
    ``template`` (required when ``result == "template"``), and
    ``timeout`` (optional, a positive number of seconds).
    """
    allowed = {"command", "result", "template", "timeout"}
    unknown = set(spec_obj.keys()) - allowed
    if unknown:
        msg = f"handler entry for {method!r} has unknown keys: {sorted(unknown)!r}"
        raise ConfigError(msg)

    command = spec_obj["command"]
    if not isinstance(command, list) or not command:
        msg = f"handler entry for {method!r}: 'command' must be a non-empty array"
        raise ConfigError(msg)
    for i, arg in enumerate(command):
        if not isinstance(arg, str):
            msg = f"handler entry for {method!r}: 'command' element {i} is not a string"
            raise ConfigError(msg)
    if shutil.which(command[0]) is None:
        msg = f"handler entry for {method!r}: command[0] {command[0]!r} not on PATH"
        raise ConfigError(msg)

    result_raw = spec_obj.get("result")
    if result_raw is None:
        result_mode = _DEFAULT_RESULT_MODE.get(method, ResultMode.JSON)
    else:
        if not isinstance(result_raw, str):
            msg = f"handler entry for {method!r}: 'result' must be a string"
            raise ConfigError(msg)
        try:
            result_mode = ResultMode(result_raw)
        except ValueError as exc:
            valid = ", ".join(repr(m.value) for m in ResultMode)
            msg = f"handler entry for {method!r}: 'result' is {result_raw!r}; expected one of {valid}"
            raise ConfigError(msg) from exc

    template = spec_obj.get("template")
    if result_mode is ResultMode.TEMPLATE:
        if template is None:
            msg = f"handler entry for {method!r}: result = 'template' requires a 'template' table"
            raise ConfigError(msg)
    elif template is not None:
        msg = (
            f"handler entry for {method!r}: 'template' is only valid when "
            f"result = 'template' (got result = {result_mode.value!r})"
        )
        raise ConfigError(msg)

    timeout = spec_obj.get("timeout", _SUBPROCESS_TIMEOUT)
    if not _is_valid_timeout(timeout):
        msg = f"handler entry for {method!r}: 'timeout' must be a positive number of seconds"
        raise ConfigError(msg)

    return SubprocessHandlerSpec(
        argv_template=tuple(command),
        result_mode=result_mode,
        result_template=template,
        timeout=float(timeout),
    )


def _resolve_import(import_path: str) -> Any:
    """Import the dotted path and return the resolved attribute.

    Raises :class:`ConfigError` if the module cannot be imported, the
    attribute is missing, or the resolved object is not callable.
    """
    if "." not in import_path:
        msg = f"import path {import_path!r} must be a dotted path like 'pkg.module.attr'"
        raise ConfigError(msg)
    module_name, _, attr = import_path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        msg = f"cannot import module {module_name!r}: {exc}"
        raise ConfigError(msg) from exc
    if not hasattr(module, attr):
        msg = f"module {module_name!r} has no attribute {attr!r}"
        raise ConfigError(msg)
    obj = getattr(module, attr)
    if not callable(obj):
        msg = f"{import_path!r} resolved but is not callable: {type(obj).__name__}"
        raise ConfigError(msg)
    return obj


# --------------------------------------------------------------------- #
# Substitution.
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class _SubstitutionError(Exception):
    """Raised internally when a ``{key}`` token cannot be resolved.

    The factory wraps this into a ``WspError(EXECUTION_FAILED)`` with
    a ``data.reason`` discriminator so the caller can distinguish
    template-rendering failures from spawn / exit / parse failures.
    """

    reason: str
    detail: str


def _coerce_to_str(value: Any, *, key: str) -> str:
    """Render ``value`` as the string form to insert into argv.

    Booleans and None are rejected because their string forms are
    Python-specific (``"True"`` / ``"None"``) and almost never what
    the wrapped CLI expects. NUL bytes are rejected because POSIX
    argv cannot carry them.
    """
    if value is None:
        msg = f"argv template references {key!r} but the param is null"
        raise _SubstitutionError(reason="missing-template-key", detail=msg)
    if isinstance(value, bool):
        msg = f"argv template references {key!r} but the param is a boolean; coerce to a string in the handler"
        raise _SubstitutionError(reason="unsupported-template-value", detail=msg)
    if isinstance(value, (int, float)):
        rendered = str(value)
    elif isinstance(value, str):
        rendered = value
    else:
        msg = (
            f"argv template references {key!r} but the param is a "
            f"{type(value).__name__}; only strings, integers, and floats are supported"
        )
        raise _SubstitutionError(reason="unsupported-template-value", detail=msg)
    if "\x00" in rendered:
        msg = f"argv template references {key!r} but the value contains a NUL byte (POSIX argv forbids NUL)"
        raise _SubstitutionError(reason="unsupported-template-value", detail=msg)
    return rendered


def _render_argv(template: tuple[str, ...], params: Any) -> list[str]:
    """Render an argv template by substituting from ``params``.

    Whole-element tokens of the form ``"{key}"`` are substituted with
    the matching value from ``params`` (coerced to a string). The
    special token ``"{argv}"`` splats a JSON array into multiple argv
    positions; it is intended for ``environment/execute`` and similar
    methods. Embedded substitution (e.g. ``"prefix-{name}-suffix"``)
    is also supported for non-array values.

    ``params`` may be any JSON value. Top-level non-dicts are passed
    through untouched (no substitutions are possible).
    """
    out: list[str] = []
    for elem in template:
        if elem == "{argv}":
            argv_value = _lookup_param(params, "argv")
            if not isinstance(argv_value, list) or not all(isinstance(a, str) for a in argv_value):
                msg = "argv template references {argv} but the 'argv' param is not a list of strings"
                raise _SubstitutionError(reason="unsupported-template-value", detail=msg)
            out.extend(argv_value)
            continue
        if elem.startswith("{") and elem.endswith("}") and elem[1:-1].isidentifier():
            key = elem[1:-1]
            value = _lookup_param(params, key)
            out.append(_coerce_to_str(value, key=key))
            continue
        # General-purpose embedded substitution: replace every
        # ``{ident}`` token inside the string. ``str.format_map`` is
        # the natural primitive but it raises ``KeyError`` rather than
        # our typed ``_SubstitutionError`` -- iterate manually so
        # diagnostics stay consistent.
        out.append(_render_string(elem, params))
    return out


def _render_string(template: str, params: Any) -> str:
    """Render ``{key}`` tokens inside a single string."""
    chunks: list[str] = []
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch == "{":
            # Allow ``{{`` as a literal ``{``.
            if i + 1 < n and template[i + 1] == "{":
                chunks.append("{")
                i += 2
                continue
            close = template.find("}", i + 1)
            if close == -1:
                msg = f"argv template {template!r} has an unclosed '{{'"
                raise _SubstitutionError(reason="malformed-template", detail=msg)
            key = template[i + 1 : close]
            if not key.isidentifier():
                msg = f"argv template {template!r} contains an invalid token {{{key}}}"
                raise _SubstitutionError(reason="malformed-template", detail=msg)
            value = _lookup_param(params, key)
            chunks.append(_coerce_to_str(value, key=key))
            i = close + 1
            continue
        if ch == "}":
            # Allow ``}}`` as a literal ``}``.
            if i + 1 < n and template[i + 1] == "}":
                chunks.append("}")
                i += 2
                continue
            msg = f"argv template {template!r} has an unmatched '}}'"
            raise _SubstitutionError(reason="malformed-template", detail=msg)
        chunks.append(ch)
        i += 1
    return "".join(chunks)


def _params_as_json(params: Any) -> Any:
    """Return ``params`` in its JSON wire shape for template rendering.

    For WSP methods the dispatcher hands handlers a normalized params
    dataclass (e.g. :class:`~wispy.endpoints.ExecuteParams`), but
    templates address params by their wire names. Convert the dataclass
    back to a dict, turning tuples into lists (so ``{argv}`` splats) and
    omitting unset optional fields (so a reference to one reports
    "missing"). Any other value is returned unchanged.
    """
    if not dataclasses.is_dataclass(params) or isinstance(params, type):
        return params
    out: dict[str, Any] = {}
    for f in dataclasses.fields(params):
        value = getattr(params, f.name)
        if value is None:
            continue
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, Mapping):
            value = dict(value)
        out[f.name] = value
    return out


def _lookup_param(params: Any, key: str) -> Any:
    """Look up ``key`` in ``params`` (a dict)."""
    if not isinstance(params, dict):
        msg = f"argv template references {key!r} but params is not a JSON object"
        raise _SubstitutionError(reason="missing-template-key", detail=msg)
    if key not in params:
        msg = f"argv template references {key!r} but the param is missing"
        raise _SubstitutionError(reason="missing-template-key", detail=msg)
    return params[key]


def _render_template(value: Any, params: Any) -> Any:
    """Render a result template by substituting from ``params``.

    Walks ``value`` recursively. Strings are rendered through
    :func:`_render_string`; dicts and lists have their elements
    rendered in place; other scalars are passed through.
    """
    if isinstance(value, str):
        return _render_string(value, params)
    if isinstance(value, list):
        return [_render_template(v, params) for v in value]
    if isinstance(value, dict):
        return {k: _render_template(v, params) for k, v in value.items()}
    return value


# --------------------------------------------------------------------- #
# Subprocess handler factory.
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Outcome:
    """The captured outcome of one subprocess run."""

    returncode: int
    stdout: bytes
    stderr: bytes
    argv: list[str] = field(default_factory=list)


async def _spawn_and_capture(argv: list[str], timeout: float = _SUBPROCESS_TIMEOUT) -> _Outcome:
    """Spawn ``argv``, close stdin, and capture stdout/stderr.

    Raises :class:`WspError` on any of: spawn failure, timeout. The
    caller decides whether a non-zero return code is itself a failure
    (it is for every result mode wispy currently supports). If the
    calling task is cancelled (e.g. by the runtime's drain timeout),
    the child is killed rather than left running.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        msg = f"failed to spawn subprocess handler: {exc}"
        raise WspError(
            int(WspErrorCode.EXECUTION_FAILED),
            msg,
            data={"reason": "spawn-failed", "argv": argv},
        ) from exc

    # Real CLIs adapted by this flow never read stdin; close it
    # immediately so the child sees EOF on read attempts.
    if proc.stdin is not None:
        proc.stdin.close()
        with contextlib.suppress(Exception):
            await proc.stdin.wait_closed()

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        # Reap the child so its transport and pipes are closed rather
        # than leaked until the loop closes.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        raise
    except asyncio.TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        try:
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=2.0)
        except Exception:  # noqa: BLE001 - teardown after timeout must be robust
            stderr = b""
        msg = f"subprocess handler timed out after {timeout:g}s"
        raise WspError(
            int(WspErrorCode.EXECUTION_FAILED),
            msg,
            data={
                "reason": "timeout",
                "argv": argv,
                "stderr": stderr.decode("utf-8", errors="replace"),
            },
        ) from exc

    return _Outcome(returncode=proc.returncode or 0, stdout=stdout, stderr=stderr, argv=argv)


def _make_subprocess_handler(spec: SubprocessHandlerSpec) -> Handler:
    """Build a :data:`~wispy.registry.Handler` for a Subprocess_Handler.

    The returned coroutine renders ``spec.argv_template`` against the
    request params, spawns the process via
    :func:`asyncio.create_subprocess_exec`, and constructs the WSP
    result per the declared :class:`ResultMode`.

    ``shutil.which`` has already been checked at config load, so we
    do not re-validate here, but the spawn itself is still wrapped in
    a try/except in case the executable disappears from PATH between
    load and call.
    """
    template = spec.argv_template
    mode = spec.result_mode
    result_template = spec.result_template
    timeout = spec.timeout

    async def handler(params: Any) -> Any:
        params = _params_as_json(params)
        try:
            argv = _render_argv(template, params)
        except _SubstitutionError as exc:
            raise WspError(
                int(WspErrorCode.EXECUTION_FAILED),
                exc.detail,
                data={"reason": exc.reason},
            ) from exc

        outcome = await _spawn_and_capture(argv, timeout)

        if mode is ResultMode.EXEC:
            # The result IS the captured outcome; non-zero exit codes
            # are not failures here, they are the value the caller
            # wants to inspect.
            return {
                "exit_code": outcome.returncode,
                "stdout": outcome.stdout.decode("utf-8", errors="replace"),
                "stderr": outcome.stderr.decode("utf-8", errors="replace"),
            }

        # Every other mode treats a non-zero exit as a failure.
        if outcome.returncode != 0:
            msg = f"subprocess handler exited with non-zero status {outcome.returncode}"
            raise WspError(
                int(WspErrorCode.EXECUTION_FAILED),
                msg,
                data={
                    "reason": "non-zero-exit",
                    "returncode": outcome.returncode,
                    "argv": argv,
                    "stderr": outcome.stderr.decode("utf-8", errors="replace"),
                },
            )

        if mode is ResultMode.NONE:
            return None

        if mode is ResultMode.JSON:
            try:
                return json.loads(outcome.stdout.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WspError(
                    int(WspErrorCode.EXECUTION_FAILED),
                    f"subprocess handler did not produce valid JSON: {exc}",
                    data={
                        "reason": "invalid-json-output",
                        "argv": argv,
                        "stderr": outcome.stderr.decode("utf-8", errors="replace"),
                    },
                ) from exc

        # mode is ResultMode.TEMPLATE.
        try:
            return _render_template(result_template, params)
        except _SubstitutionError as exc:
            raise WspError(
                int(WspErrorCode.EXECUTION_FAILED),
                exc.detail,
                data={"reason": exc.reason, "argv": argv},
            ) from exc

    return handler


def make_handler(entry: ConfigEntry) -> Handler:
    """Build a :data:`~wispy.registry.Handler` callable for ``entry``.

    For :class:`PythonHandlerSpec`, re-resolve the dotted path and
    return the callable directly. :func:`load_config` has already
    validated the path, so this should not fail in practice.

    For :class:`SubprocessHandlerSpec`, return an async wrapper that
    renders the configured ``argv_template`` against the request
    params, spawns the process, and constructs the WSP result per
    the declared :class:`ResultMode`. Failures are translated to
    :class:`~wispy.errors.WspError` carrying ``EXECUTION_FAILED``.
    """
    if isinstance(entry.spec, PythonHandlerSpec):
        return _resolve_import(entry.spec.import_path)
    if isinstance(entry.spec, SubprocessHandlerSpec):
        return _make_subprocess_handler(entry.spec)
    msg = f"unknown handler spec: {entry.spec!r}"
    raise TypeError(msg)
