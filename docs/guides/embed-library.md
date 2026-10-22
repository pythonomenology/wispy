# Embed wispy in a workflow tool

If your workflow tool is written in Python, the most direct way to expose a
WSP server is to take a runtime dependency on `wispy`, register Python
callables for the methods you support, and call `run_stdio`. This guide
walks through it end to end.

## What you'll build

A minimal "hello world" workflow tool that supports `initialize`,
`shutdown`, and `environment/list`. Every other WSP method will surface as
`-32601` (method not found) at call time.

## Add the dependency

```sh
pip install wispy
```

Or, if you use a `pyproject.toml`:

```toml
[project]
dependencies = [
  "wispy",
]
```

## Wire up the registry

```python
# my_tool/server.py
import asyncio
import sys

from wispy import Capabilities, HandlerRegistry, run_stdio


registry = HandlerRegistry()


def initialize(_params: object) -> dict:
    """Return capabilities. wispy preserves the Capabilities tuple/list shape."""
    return Capabilities(
        methods=tuple(registry.methods()),
        protocol_version="0.1.0",
    ).to_jsonable()


def shutdown(_params: object) -> None:
    return None


def list_envs(_params: object) -> list[dict]:
    return [
        {"id": "default", "name": "default", "python_version": "3.12"},
    ]


registry.register("initialize", initialize)
registry.register("shutdown", shutdown)
registry.register("environment/list", list_envs)


def main() -> int:
    return asyncio.run(run_stdio(registry))


if __name__ == "__main__":
    sys.exit(main())
```

A few things worth noting:

- Handlers receive the request `params` value as a single argument and
  return a JSON-serializable result. The wire data models
  (`Capabilities`, `Environment`, `ExecuteResult`, `DeleteAck`) may be
  returned directly; they are encoded through their `to_jsonable()`. They can be either `def` or
  `async def`; sync handlers run on the default executor so they don't
  block the event loop.
- `registry.methods()` is what you should advertise from `initialize`.
  Return `wispy.Capabilities(...)` or its `to_jsonable()` dict; either
  way the lifecycle FSM caches it as `Capabilities`.
- A second `initialize` will be rejected by the lifecycle FSM with
  `-32600`; the cached capabilities from the first successful call are
  preserved verbatim.

## Add a project script

In `pyproject.toml`:

```toml
[project.scripts]
my-tool-wsp = "my_tool.server:main"
```

After `pip install -e .`, your tool can be launched as a WSP server with
`my-tool-wsp`, and `wsp` can drive it:

```sh
wsp --tool my-tool-wsp environment/list
```

## Raising errors

Use `WspError` for application-level failures the client should be able to
react to:

```python
from wispy import WspError, WspErrorCode


def get_env(params):
    env_id = params["id"]
    if env_id not in INDEX:
        raise WspError(
            WspErrorCode.ENVIRONMENT_NOT_FOUND,
            f"environment {env_id!r} not found",
            data={"id": env_id},
        )
    return INDEX[env_id].to_jsonable()
```

The dispatcher catches `WspError`, copies `(code, message, data)` onto the
JSON-RPC error response verbatim, and omits `data` from the wire when you
don't supply one. See [Error model](../concepts/errors.md) for the full
exception-mapping table.

## Async handlers

```python
import asyncio

async def slow_handler(_params):
    await asyncio.sleep(1.0)
    return {"slept": True}

registry.register("environment/execute", slow_handler)
```

Async handlers are awaited directly. If stdin reaches EOF while a handler
is still running, the runtime gives it up to `drain_timeout` seconds (five
by default) to finish before cancelling.

## Custom drain timeout

`run_stdio` accepts a `drain_timeout` keyword argument:

```python
asyncio.run(run_stdio(registry, drain_timeout=15.0))
```

You can also inject a custom transport (typically only useful in tests):

```python
from wispy import StdioTransport

transport = StdioTransport(stdin=..., stdout=..., stderr=...)
asyncio.run(run_stdio(registry, transport=transport))
```

## Don't want a runtime dependency?

If you'd rather not import `wispy`, see
[Ship a Config_File](config-file.md). The Config_File flow lets you map WSP
methods to dotted import paths *or* to external commands, so your tool
never needs to know `wispy` exists.
