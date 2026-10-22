# Use the wsp CLI

`wsp` is the command-line client that ships with `wispy`. It performs the
per-invocation lifecycle for you (`initialize` → method → `shutdown` →
`exit`), prints the JSON result of each call to stdout, and chooses an
exit code based on the response.

## Three modes

`wsp` runs in one of three modes, picked by the top-level flags:

- **`--tool ARGV... METHOD`** — launches the supplied argv as the WSP
  server and routes the call through it, e.g.
  `wsp --tool my-tool wsp-serve environment/list`. The server argv ends
  at the first argument that names a method subcommand. Use this when
  you have a workflow tool that knows how to run as a WSP server
  (because it embedded `wispy` directly, or because it implements WSP
  itself).

  If the server fails before answering (for example a `--config` child
  rejecting its Config_File), its stderr is echoed after `wsp`'s own
  error message.
- **`--config PATH`** — launches `python -m wispy --config PATH` as the
  WSP server. Use this with the [Config_File flow](config-file.md).
- **No flag** — runs the built-in fallback workflow tool in-process. Use
  this when you just want to manage `venv`-backed environments without
  setting anything else up.

`--tool` and `--config` are mutually exclusive; passing both is a usage
error and exits with status `2`.

## Subcommands

Every WSP method has a corresponding subcommand:

| Subcommand               | What it does                                        |
|--------------------------|-----------------------------------------------------|
| `wsp environment/list`   | List managed environments.                          |
| `wsp environment/create` | Create a new environment.                           |
| `wsp environment/get`    | Fetch full details about one.                       |
| `wsp environment/delete` | Remove an environment.                              |
| `wsp environment/execute`| Run a command inside an environment.                |

`initialize` and `shutdown` are not exposed as user-facing subcommands —
they're wrapped around every call automatically.

Each subcommand exposes one `--<param>` flag per parameter on the WSP
method, with `snake_case` field names spelled in `kebab-case`. For
example, `environment/create` accepts `--name` and `--python-version`.

## Passing parameters

There are three ways to set request parameters:

1. **Per-flag:** `wsp environment/create --name scratch --python-version 3.12`
2. **JSON on stdin:** `echo '{"name":"scratch","python_version":"3.12"}' | wsp environment/create --params-json -`
3. **Both:** the JSON on stdin sets defaults; flags shadow JSON entries
   for the same key. Flags left at their default of `None` do not shadow
   the JSON entry, so you can JSON-supply a value and have it survive
   even when the corresponding flag exists on the parser.

The JSON form has a 1 MiB cap. Anything larger is rejected with a usage
error and status `2`.

## Output and exit codes

On success, `wsp` writes the JSON-RPC `result` to stdout as a JSON
document followed by a newline, and exits with status `0`.

On failure, the JSON-RPC `error` object goes to stderr and `wsp` chooses
an exit code based on the code:

| Status | Meaning                                                                 |
|-------:|-------------------------------------------------------------------------|
| `0`    | Successful result.                                                       |
| `1`    | Any JSON-RPC error other than `-32601`, or a launch/response timeout.    |
| `2`    | `-32601` (method not supported) or a usage error.                        |

See [Exit codes](../reference/exit-codes.md) for the rationale.

## Timeouts

`wsp` enforces two 30-second budgets per invocation:

- **Launch:** the child server has 30 seconds to start.
- **Per-call:** each request has 30 seconds to receive a response. In
  `--config` mode, when the Config_File gives the requested method a
  longer `timeout`, `wsp` waits that long plus five seconds instead.

Exceeding either prints a one-line error to stderr and exits with status
`1`. Responses are not size-limited.

## Termination

When `wsp` exits — for any reason — it sends `shutdown` and `exit` to its
child server (if it spawned one), waits up to 10 seconds for the child to
terminate gracefully, and then `proc.kill()`s it. You don't have to babysit
child processes.

## Examples

### Listing environments managed by the fallback

```sh
wsp environment/list
```

```json
[
  {"id": "env-1f0eafbb96d3", "name": "scratch", "python_version": "3.12"}
]
```

### Creating an environment with the fallback

```sh
wsp environment/create --name scratch --python-version 3.12
```

```json
{
  "id": "env-1f0eafbb96d3",
  "name": "scratch",
  "python_version": "3.12",
  "interpreter_path": "/.../wispy/fallback/envs/env-1f0eafbb96d3/bin/python",
  "installed_packages": [],
  "extra": {}
}
```

### Driving a Config_File-based tool

```sh
wsp --config my-tool.toml environment/list
```

### Executing a command in an environment

`--argv` takes every remaining argument as the command, so it must be
the last flag. `--env KEY=VALUE` may be repeated:

```sh
wsp environment/execute --id env-1f0eafbb96d3 --env PYTHONWARNINGS=error --argv python -V
```

The same request as JSON:

```sh
echo '{"id": "env-1f0eafbb96d3", "argv": ["python", "-V"]}' \
  | wsp environment/execute --params-json -
```

```json
{"exit_code": 0, "stdout": "Python 3.12.4\n", "stderr": ""}
```

## State directory

The fallback workflow tool persists its environment registry under:

- `${WISPY_STATE_DIR}` if you set it.
- `${XDG_STATE_HOME:-~/.local/state}/wispy/fallback/` on POSIX otherwise.
- `%LOCALAPPDATA%\wispy\fallback\` on Windows otherwise.

Pointing `WISPY_STATE_DIR` at a temp directory is the recommended way to
isolate experiments.
