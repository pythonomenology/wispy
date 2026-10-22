# Testing

`wispy` is built around the idea that the protocol implementation should
be verified against universal properties before any I/O is wired in. The
test suite is organised in three tiers, each with a different scope and
guarantee.

## Layout

```
tests/
├── unit/           # Fast, deterministic, single-module tests.
├── property/       # Hypothesis-driven property tests.
└── integration/    # End-to-end tests that spawn real subprocesses.
```

Run any tier on its own:

```sh
hatch test --python 3.12 tests/unit/
hatch test --python 3.12 tests/property/
hatch test --python 3.12 tests/integration/
```

Or all of them together:

```sh
hatch test --python 3.12
```

## Properties under test

The property tests are the canonical specification of how the protocol
is supposed to behave. Each one corresponds to a universal claim about
the implementation:

- **JSON-RPC response well-formedness and id preservation.** Every
  response carries `jsonrpc == "2.0"`, exactly one of `result` /
  `error`, and an `id` whose JSON type and value match the request.
- **Notifications produce no response.** A request without an `id`
  field never causes the dispatcher to emit any output, even on error.
- **Error-code priority.** Inputs that satisfy more than one of
  {parse error, invalid request, method not found, invalid params}
  produce only the highest-priority code.
- **Batch ordering and notification omission.** The response array's
  length equals the number of non-notification entries, and the order
  matches the input.
- **Frame round-trip and recovery.** The framing decoder is invariant
  to chunking and recovers from arbitrary garbage prefixes.
- **Handler dispatch round-trip.** A handler bound to a method is
  invoked exactly once with the request params, and its return value
  becomes the response `result`.
- **Duplicate registration leaves the registry unchanged.** Attempting
  to register a method twice raises and the registry's state is
  bitwise-equal to its pre-call snapshot.
- **Handler exception mapping.** `WspError` round-trips, other
  `ProtocolError`s remap to `-31004`, anything else becomes `-32603`.
- **Subprocess handler round-trip and failure mapping.** Round-trip
  works for any JSON-serializable value; non-zero exit, garbled stdout,
  and timeout all surface as `WspError(EXECUTION_FAILED)`.
- **Lifecycle FSM transitions.** Driven by a `RuleBasedStateMachine`
  that mirrors the design's state diagram against a shadow model.
- **Capabilities reflect the registry exactly.** The `Capabilities`
  cached after `initialize` enumerates every registered method, and
  `protocol_version` is semver-shaped.
- **Environment endpoints behave as a model.** A
  `RuleBasedStateMachine` over `create` / `get` / `delete` / `list` /
  `execute`, with an in-memory dictionary as the oracle.
- **WSP error codes are in the reserved range.** Every member of
  `WspErrorCode` lies in `[-31999, -31000]`.
- **CLI param merge.** Stdin-supplied JSON and explicit flags merge
  the way the spec describes; flags with `None` values don't shadow.
- **CLI exit code mapping.** Successful results map to `0`, `-32601`
  to `2`, anything else to `1`.
- **Fallback Workflow_Tool persistence.** Each rule rebuilds the
  registry from scratch (simulating a fresh CLI invocation), and the
  on-disk `index.json` is the only thing that carries state forward.

## Integration tests

The integration tests spawn real Python subprocesses and exercise the
full pipeline:

- `test_stdio_transport.py` — frame round-trip and EOF drain.
- `test_lifecycle_runtime.py` — `initialize` → `shutdown` → `exit`
  status codes, and pre-`initialize` rejections.
- `test_cli_fallback_e2e.py` — full CRUD round-trip through `wsp`
  against the in-process fallback (creates a real venv on disk,
  optional and skipped when `with_pip=True` venv creation isn't
  supported on the host).
- `test_cli_modes_e2e.py` — `--config` and `--tool` modes.

## Running a single property test

Property tests are conventional pytest functions; you can run one in
isolation:

```sh
hatch test --python 3.12 tests/property/test_lifecycle_fsm.py::TestLifecycleFSM
```

To increase the example budget for a deeper run, set the Hypothesis
profile via env var:

```sh
hatch test --python 3.12 -- --hypothesis-profile=ci tests/property/
```

## Why so much property-based testing?

The pure parts of the codebase — the JSON-RPC codec, the framing
decoder, the dispatcher, the registry, the lifecycle FSM, the error
model, the environment endpoint validators — have universal claims that
*example-based* tests are bad at expressing. Hypothesis generates
adversarial inputs across the same input space and shrinks failures
into minimal counterexamples, which is exactly the right tool for
verifying a protocol implementation.

## Formal models

Three hand-written models sit beside the tests. TLA+ specs live in
`tla/` and are checked by the TLC model checker, which tries every
ordering of the modelled steps at small sizes. The Lean 4 model lives
in `lean/` and is a proof for every input size. Each model's header
comment names the source lines it follows and its assumptions. Each
`.mutations` file lists the deliberate bugs its checks must catch.

| Model | What it checks | Boundary configuration |
| --- | --- | --- |
| `tla/ServerAdmission` | `run_stdio` admits each message against the lifecycle state at the time it was read; every request that is not cancelled gets exactly one response; nothing is written after return; the exit status is 0 only when `shutdown` preceded `exit`; `run_stdio` returns after EOF or `exit`. | `reqexit shipped`: `initialize, req, exit` in one stdin chunk with the built-in sync handlers. |
| `tla/FallbackStore` | Concurrent `environment/*` calls never lose an index update or duplicate a name or id; a crash never leaves `index.json` torn; every listed env has a complete venv, even after a crash or failed removal; `execute` and `get` keep their env until they finish; leftover directories from a crash or failed removal are gone after the next create or delete. They are not prevented. | `EX h=2 pre=1`: one `execute` and one `delete` of the same env. `OD h=1 faults=crash`: a create killed mid-build leaves a directory until the next sweep. |
| `lean/Model/FrameDecoder` | Every valid frame round-trips; any chunking yields the same payloads; an oversized frame is skipped by exactly its length; the garbage discard keeps 3 bytes so a split terminator is still found; every step shrinks the buffer. | Header blocks of at most 8,193 bytes; longer ones make the output depend on chunk boundaries. |

Recheck with the `agent-formal-verify` plugin's helpers, where
`<skill-dir>` is its installed `formal-verify` skill directory:

```sh
bash <skill-dir>/scripts/tlc-matrix.sh "$PWD/tla/ServerAdmission.matrix"
bash <skill-dir>/scripts/tlc-matrix.sh "$PWD/tla/FallbackStore.matrix"
bash <skill-dir>/scripts/lean-check.sh "$PWD/lean"
python3 <skill-dir>/scripts/mutate.py "$PWD/tla/ServerAdmission.mutations"
python3 <skill-dir>/scripts/mutate.py "$PWD/tla/FallbackStore.mutations"
python3 <skill-dir>/scripts/mutate.py "$PWD/lean/Model/FrameDecoder.mutations"
```
