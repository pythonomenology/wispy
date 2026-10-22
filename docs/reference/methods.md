# WSP methods

This page documents the request and response shapes for every WSP method
defined in this iteration. Anything not listed here will surface as
JSON-RPC error `-32601` (method not found).

All field names are JSON keys; types are JSON types unless otherwise
noted.

## Lifecycle

### `initialize`

Negotiate the session and discover the server's capabilities.

**Params**

| Field                     | Type   | Constraints                       |
|---------------------------|--------|-----------------------------------|
| `client_name`             | string | non-empty, 1–255 chars            |
| `client_protocol_version` | string | non-empty, 1–64 chars             |

**Result** — `Capabilities`

| Field              | Type             | Notes                                                |
|--------------------|------------------|------------------------------------------------------|
| `methods`          | array of string  | Every WSP method bound on this server, exactly once. |
| `protocol_version` | string           | Semver `MAJOR.MINOR.PATCH`. Currently `"0.1.0"`.     |

### `shutdown`

Asks the server to stop accepting new requests. After a successful
`shutdown`, only the `exit` notification is honoured.

**Params:** none.
**Result:** `null`.

### `exit` (notification)

Terminates the server process. Sent as a JSON-RPC notification (no `id`,
no response). The exit status depends on whether `shutdown` ran first:

- After a successful `shutdown`: status `0`.
- Otherwise: status `1`.

## Environments

### `environment/list`

Enumerate every environment managed by the server.

**Params:** none (or empty object/array).
**Result:** array of `Environment` summaries, ordered by ascending `id`.

```json
[
  {"id": "env-1f0eafbb96d3", "name": "scratch", "python_version": "3.12"}
]
```

### `environment/create`

Provision a new environment.

**Params**

| Field            | Type   | Constraints                                                  |
|------------------|--------|--------------------------------------------------------------|
| `name`           | string | non-empty, 1–64 chars                                        |
| `python_version` | string | `MAJOR.MINOR` or `MAJOR.MINOR.PATCH`, components non-negative |

**Result** — `Environment` details (full form, including
`interpreter_path`, `installed_packages`, and `extra`).

**Errors**

- `-31002` `ENVIRONMENT_NAME_CONFLICT` if `name` collides with an
  existing environment. The error's `data` lists every violated
  validation rule.
- `-31003` `PYTHON_VERSION_UNAVAILABLE` if `python_version` cannot be
  provisioned.
- `-32602` for syntactic violations of the params schema (with `data`
  carrying the violation list).
- `-31004` `EXECUTION_FAILED` for any non-protocol failure during
  provisioning. The partial environment, if any, is removed before the
  error is raised; the index is left unchanged.

### `environment/get`

Fetch full details for an environment.

**Params**

| Field | Type   | Constraints                  |
|-------|--------|------------------------------|
| `id`  | string | non-empty, 1–128 chars       |

**Result** — `Environment` details:

| Field                | Type             | Notes                                              |
|----------------------|------------------|----------------------------------------------------|
| `id`                 | string           | Matches the requested id.                          |
| `name`               | string           | Same as the summary form.                          |
| `python_version`     | string           | Same as the summary form.                          |
| `interpreter_path`   | string           | Absolute path to the interpreter inside the env.   |
| `installed_packages` | array of `Package`| `[]` when the env has no packages.                |
| `extra`              | object           | Always present; reserved for tool-specific fields. |

`Package` has `name` and `version`, both strings.

**Errors**

- `-31001` `ENVIRONMENT_NOT_FOUND` with `data = {"id": <id>}` when the
  id is unknown.

### `environment/delete`

Remove an environment.

**Params:** `{"id": <string>}`.

**Result** — `DeleteAck`:

| Field | Type   | Notes                            |
|-------|--------|----------------------------------|
| `id`  | string | Echoes the deleted environment's id. |

**Errors**

- `-31001` `ENVIRONMENT_NOT_FOUND` if the id is unknown.
- `-31004` `EXECUTION_FAILED` if the id was known but deletion failed.

The built-in fallback removes the environment from its index before
deleting its files. If deleting the files then fails, the call still
succeeds, a warning naming the directory is written to stderr, and the
leftover files are removed by a later `environment/create` or
`environment/delete`. A delete waits for any `environment/get` or
`environment/execute` already using that environment to finish.

A command run by the fallback's `environment/execute` cannot delete the
environment it runs in: that delete would wait for its own parent, so it
fails at once with `EXECUTION_FAILED` and `data.reason = "env-in-use"`.
The command can still `get` or `execute` in that environment.

### `environment/execute`

Run a command inside an environment.

**Params**

| Field   | Type                   | Constraints                                                          |
|---------|------------------------|----------------------------------------------------------------------|
| `id`    | string                 | non-empty                                                            |
| `argv`  | array of string        | non-empty; every element is a string                                 |
| `cwd`   | string \| null         | optional; absolute path inside the environment                       |
| `env`   | object<string,string> \| null | optional; environment variables to set for the child           |

**Result** — `ExecuteResult`:

| Field       | Type    | Notes                                                |
|-------------|---------|------------------------------------------------------|
| `exit_code` | integer | The command's exit status.                           |
| `stdout`    | string  | Captured stdout, decoded UTF-8 with `errors="replace"`. |
| `stderr`    | string  | Captured stderr, decoded UTF-8 with `errors="replace"`. |

**Errors**

- `-32602` if `argv` is empty or any element is not a string.
- `-31001` `ENVIRONMENT_NOT_FOUND` if the id is unknown.
- `-31004` `EXECUTION_FAILED` if the command can't be launched at all.

## Length and format constraints

The constraints in this page are upper bounds on what the parameter
validators accept. Servers may impose tighter limits, in which case
they'll reject with `-32602` and a violation list in `data`.
