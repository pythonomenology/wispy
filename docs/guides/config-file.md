# Ship a Config_File

If you'd rather not depend on `wispy` at runtime — for example because
you want to keep your library footprint small, or because your tool is
implemented in another language — you can describe a WSP server with a
TOML or JSON config file and let `wispy` itself launch it.

The Config_File flow is designed to **adapt an existing CLI** to WSP
without writing a single line of Python. You declare which subcommand
of your CLI maps to each WSP method, how to thread the request params
into the argv, and how to construct the WSP result from what your CLI
already does.

## Schema

The schema is the same in TOML and JSON. The top-level document is an
object; the `handlers` key maps WSP method names to *handler entries*;
each handler entry has exactly one of `import` or `command`.

```toml
# my-tool.toml
[handlers."environment/list"]
command = ["my-tool", "env", "show", "--json"]
result = "json"

[handlers."environment/create"]
command = ["my-tool", "env", "create", "{name}", "-p", "{python_version}"]
result = "template"
template = { id = "{name}", name = "{name}", python_version = "{python_version}", interpreter_path = "", installed_packages = [], extra = {} }
```

The extension is matched case-insensitively (`.toml` or `.json`); any
other extension is rejected at startup.

## Handler kinds

### `command` — adapt an existing CLI

This is the primary path the Config_File flow exists for. You declare
the argv your tool wants to run; wispy substitutes the WSP request
params into it, spawns the process with stdin closed (your CLI does
not need to read JSON on stdin), and constructs the WSP result based
on your declared `result` mode.

#### Argv substitution

Tokens of the form `"{key}"` in `command` are replaced with the value
of `params.key` from the request:

```toml
command = ["my-tool", "env", "create", "{name}", "-p", "{python_version}"]
```

Calling `environment/create` with
`{"name": "scratch", "python_version": "3.12"}` becomes:

```
my-tool env create scratch -p 3.12
```

Substitution rules:

- `"{name}"` — the entire element is replaced with the param value.
- `"prefix-{name}-suffix"` — embedded substitution inside a single
  argv element.
- `"{argv}"` — special: splats a JSON array of strings into multiple
  argv positions. Use this for `environment/execute`, where the
  request's `argv` field is the inner command to run.
- `"{{"` and `"}}"` produce literal `{` and `}` characters.

Values are coerced to strings:

| Param type | Treatment                                                |
|-----------:|----------------------------------------------------------|
| `string`   | Inserted as-is.                                          |
| `integer`  | Decimal string.                                          |
| `float`    | Decimal string.                                          |
| `null`     | Rejected (`-31004` `EXECUTION_FAILED`).                  |
| `boolean`  | Rejected — `"True"` is rarely what wrapped CLIs expect.  |
| anything else | Rejected.                                             |

A reference to a missing key, or a value that cannot be coerced, is
mapped to `WspError(EXECUTION_FAILED, data={"reason": "missing-template-key"})`
(or `"unsupported-template-value"` for the type-mismatch case).

#### Result modes

The `result` field declares how wispy builds the WSP method's result
from the subprocess's outcome:

| Mode         | Behaviour                                                                                              | Best for                                                              |
|--------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `"json"`     | Parse stdout as JSON. That value is the result. Non-zero exit → `EXECUTION_FAILED`.                    | Tools whose subcommand emits JSON (typically with a `--json` flag).    |
| `"template"` | Render the configured `template` table from the request params. Stdout is ignored. Non-zero exit → `EXECUTION_FAILED`. | Tools whose subcommand doesn't emit the metadata WSP wants in the result; you synthesize it from the request. |
| `"exec"`     | Result is `{exit_code, stdout, stderr}`. Non-zero exit is **not** a failure — the exit code is the value the caller wants. | `environment/execute` and similar "run something and report what happened" methods. |
| `"none"`     | Result is `null`. Stdout is ignored. Non-zero exit → `EXECUTION_FAILED`.                               | Methods whose schema permits `null` (e.g. `shutdown`).                |

Each WSP method has a sensible default mode, so most callers don't
need to specify `result` explicitly:

| Method                  | Default mode  |
|-------------------------|---------------|
| `initialize`            | `"json"`      |
| `shutdown`              | `"none"`      |
| `environment/list`      | `"json"`      |
| `environment/get`       | `"json"`      |
| `environment/create`    | `"template"`  |
| `environment/delete`    | `"template"`  |
| `environment/execute`   | `"exec"`      |

Methods whose default is `"template"` still require an explicit
`template` table — wispy can't guess your CLI's data model.

#### Templates

A `template` is a JSON-like table whose strings get the same
`{key}` substitution as `command`:

```toml
[handlers."environment/create".template]
id = "{name}"
name = "{name}"
python_version = "{python_version}"
interpreter_path = ""
installed_packages = []
extra = {}
```

Substitution recurses through dicts and lists. Non-string scalars
(integers, booleans, null) pass through untouched, so you can mix
synthesized fields with literal ones.

#### Stdin and timeouts

- The child's stdin is closed immediately. Your CLI does not need to
  read anything from stdin.
- Each invocation has a wall-clock timeout of 30 seconds, which an
  entry can override with `timeout = <seconds>` (any positive number;
  raise it for slow operations such as `environment/create`).
  Exceeding it kills the child and raises `EXECUTION_FAILED` with
  `data.reason = "timeout"`. `wsp --config` reads the entry's
  `timeout` and waits that long (plus five seconds) for the response,
  so a raised timeout is not cut off by `wsp`'s own 30-second limit.
- If the server abandons a request (for example when draining after
  stdin closes), the child is killed rather than left running.

### `import` — host a Python callable

For methods whose data model is too far from WSP's to be bridged with
substitution and templating, fall back to a Python `import`:

```toml
[handlers."environment/get"]
import = "my_tool_wsp.handlers.get_env"
```

The dotted path is resolved with `importlib.import_module` plus
`getattr`. The trailing component must point at a callable. The
handler receives the request `params` as a single argument and returns
a JSON-serializable value, exactly the same contract as the
[programmatic flow](embed-library.md).

Use this when:

- Your CLI's output uses different field names than WSP (Hatch calls
  it `python`; WSP calls it `python_version`).
- The result requires rich data your CLI doesn't print
  (`installed_packages` for `environment/get`, for example).
- You need to call multiple subcommands and merge their output.

The Python handler can shell out to your CLI using `subprocess`,
`asyncio.create_subprocess_exec`, or anything else. wispy doesn't care
how it produces the result.

## Validation

At startup, wispy validates:

- The file extension and parse.
- That the document is an object and `handlers` is an object.
- That every method name is a defined WSP method.
- That every handler entry has exactly one of `import` and `command`,
  and no unknown keys (Python entries accept only `import`; subprocess
  entries accept `command`, `result`, and `template`).
- That every `import` resolves to a callable.
- That every `command[0]` resolves on `PATH`.
- That `result = "template"` is paired with a `template` table.
- That `template` is only present when `result = "template"`.

Any failure is reported on stderr and the process exits with status
`1` without serving any requests.

## Launching

Once you have a config file, any WSP client can launch your tool with:

```sh
python -m wispy --config my-tool.toml
```

Or via the bundled CLI:

```sh
wsp --config my-tool.toml environment/list
wsp --config my-tool.toml environment/create --name scratch --python-version 3.12
```

## What's not in the config

Two things WSP defines but the Config_File doesn't carry:

- **`initialize` and `shutdown`.** wispy registers built-in handlers
  for these automatically. You don't write them, and the Config_File
  rejects attempts to override them.
- **The `exit` notification.** It's a transport-level concern, not a
  handler.

## Method not configured

If a client requests a defined WSP method that your Config_File
doesn't bind, the server returns JSON-RPC error `-32601` (method not
found). The `Capabilities` object returned from `initialize` only
advertises the methods you explicitly bound, so well-behaved clients
will skip those calls in the first place.

## Worked example

The [Adapt an existing CLI](adapt-cli.md) page walks through every
WSP endpoint as if Hatch had decided to adopt the Config_File flow,
including the limitations and where you'd reach for an `import`
handler instead.
