# EPIC: Modular LangGraph Instrumentation Refactor

## Status

- Branch: `epic-refactor`
- Parent GitHub issue: [#1](https://github.com/mylonasc/langgraph_instrumentation/issues/1)
- Delivery model: clean break; no compatibility wrappers for the prototype API
- First persistence implementations: in-memory and SQLite
- First additional observability backend: OpenTelemetry OTLP
- Perfetto support: retained as a first-class file exporter

## Objective

Turn the current instrumentation prototype into a maintainable Python library in
which callback collection, trace representation, persistence, and export are
independent components.

The target design must allow:

1. Replacing the in-memory store with SQLite or a future PostgreSQL adapter
   without changing LangGraph callback handling or exporters.
2. Adding an exporter without changing storage implementations.
3. Producing Perfetto-compatible Trace Event JSON from stored neutral trace
   records.
4. Sending completed spans to an OTLP collector without making OpenTelemetry a
   required dependency for Perfetto-only users.
5. Testing lifecycle, storage, and export behavior without calling a real LLM.

## Current State

Most behavior is coupled in `src/observability/instrumentation.py`:

- LangChain/LangGraph callback adaptation
- active run tracking
- sanitization, redaction, and message deduplication
- token usage aggregation
- execution-lane assignment
- Perfetto event construction
- in-memory event buffering
- filesystem output
- console logging

The repository currently has no stable import package, tests, explicit build
backend, or CI. Examples have broken imports and undeclared optional
dependencies. Generated build and package metadata also exist in the worktree.

## Architectural Direction

```text
LangGraph / LangChain callbacks
              |
              v
    LangGraphCallbackHandler
              |
              v
         TraceRecorder
          |         |
          v         v
      TraceStore   SpanProcessor(s)
   memory/SQLite   simple/batch/composite
          |         |
          v         v
     TraceBundle   OTLP exporter
          |
          v
    Perfetto exporter
```

The intended package layout is:

```text
src/langgraph_instrumentation/
    __init__.py
    models.py
    clock.py
    policies.py
    recorder.py
    processors.py
    adapters/
        AGENT.md
        langgraph.py
    stores/
        AGENT.md
        base.py
        memory.py
        sqlite.py
    exporters/
        AGENT.md
        base.py
        perfetto.py
        otlp.py
examples/
tests/
```

Dependency direction is mandatory:

- Adapters depend on the recorder and neutral models.
- The recorder depends on neutral models and store/processor protocols.
- Stores depend only on neutral models.
- Exporters depend only on neutral models and their optional third-party SDK.
- Core modules must not import SQLite, Perfetto, OTLP, or demo-agent code.
- Examples are consumers of the public API, never library dependencies.

## Cross-Cutting Decisions

### Neutral records

Perfetto dictionaries must not remain the internal data model. The internal
model will represent traces, spans, events, metrics, status, attributes, and
execution lanes. Exporters are responsible for translating that model.

### Storage versus export

Persistence and observability delivery are different extension points:

- `TraceStore` records and queries traces.
- `SpanExporter` sends completed spans to an external backend.
- `SpanProcessor` controls when and how exporters are called.
- `PerfettoExporter` exports a queried `TraceBundle` to a file.

### Time and identifiers

Spans need Unix timestamps for OTLP and monotonic timestamps for reliable local
duration measurement. Trace and span IDs must be valid OpenTelemetry-width IDs.
LangChain callback UUIDs are correlation attributes, not the canonical IDs.

### Data safety

Prompt, response, state, and tool payload capture must be policy-driven.
Metadata-only bounded capture should be the safe default. Redaction and
sanitization happen before records enter a store or exporter.

### Synchronous callback boundary

The store protocol remains synchronous because LangChain callback hooks are
frequently synchronous, including calls made by async paths. Slow stores can be
supported later through a bounded buffering decorator rather than duplicating
the full API with sync and async variants.

## Feature Index

| Feature | GitHub issue | Title | Depends on | Stage |
| --- | --- | --- | --- | --- |
| FRK-01 | [#2](https://github.com/mylonasc/langgraph_instrumentation/issues/2) | Package foundation and repository hygiene | Parent epic | 1 |
| FRK-02 | [#3](https://github.com/mylonasc/langgraph_instrumentation/issues/3) | Neutral trace domain, clocks, and identifiers | FRK-01 | 2 |
| FRK-03 | [#4](https://github.com/mylonasc/langgraph_instrumentation/issues/4) | Capture, sanitization, redaction, and usage policies | FRK-02 | 2 |
| FRK-04 | [#5](https://github.com/mylonasc/langgraph_instrumentation/issues/5) | Trace recorder and lifecycle engine | FRK-02, FRK-03 | 3 |
| FRK-05 | [#6](https://github.com/mylonasc/langgraph_instrumentation/issues/6) | TraceStore contract and in-memory implementation | FRK-02, FRK-04 | 3 |
| FRK-06 | [#7](https://github.com/mylonasc/langgraph_instrumentation/issues/7) | Thin LangGraph callback adapter | FRK-04, FRK-05 | 4 |
| FRK-07 | [#8](https://github.com/mylonasc/langgraph_instrumentation/issues/8) | Perfetto exporter and compatibility fixtures | FRK-02, FRK-05 | 4 |
| FRK-08 | [#9](https://github.com/mylonasc/langgraph_instrumentation/issues/9) | SQLite trace store | FRK-05 | 5 |
| FRK-09 | [#10](https://github.com/mylonasc/langgraph_instrumentation/issues/10) | Span processors and OpenTelemetry OTLP export | FRK-04, FRK-05 | 6 |
| FRK-10 | [#11](https://github.com/mylonasc/langgraph_instrumentation/issues/11) | Examples, documentation, CI, and release hardening | FRK-06 through FRK-09 | 7 |

## GitHub Delivery Checklist

- [ ] FRK-01: #2
- [ ] FRK-02: #3
- [ ] FRK-03: #4
- [ ] FRK-04: #5
- [ ] FRK-05: #6
- [ ] FRK-06: #7
- [ ] FRK-07: #8
- [ ] FRK-08: #9
- [ ] FRK-09: #10
- [ ] FRK-10: #11

## Delivery Stages

### Stage 1: Foundation

Deliver FRK-01. Establish package boundaries, tooling, a clean build, and
characterization fixtures before replacing implementation behavior.

### Stage 2: Domain and policy

Deliver FRK-02 and FRK-03. These may proceed in sequence on the same integration
branch; policy APIs consume domain attribute types and therefore follow the
domain model.

### Stage 3: Core collection and persistence

Deliver FRK-04 and FRK-05. The recorder must be tested against the store
protocol, while store contract tests must remain independent of LangGraph.

### Stage 4: Vertical feature parity

Deliver FRK-06 and FRK-07. At the end of this stage a deterministic LangGraph
test graph must produce a Perfetto-compatible trace through the new stack.

### Stage 5: Durable local storage

Deliver FRK-08. The same integration and Perfetto export tests must pass when
the memory store is replaced with SQLite.

### Stage 6: Additional observability

Deliver FRK-09. OTLP is optional and consumes the same completed spans used by
the rest of the system. It must not become a dependency of the core package.

### Stage 7: Productization

Deliver FRK-10. Documentation, examples, CI, packaging checks, and release
criteria complete the epic.

## Detailed Features

### FRK-01: Package Foundation and Repository Hygiene

**Goal**

Create a conventional installable package and establish quality gates without
changing the intended tracing behavior.

**Implementation**

- Configure an explicit PEP 517 build backend and `src` package discovery in
  `pyproject.toml`.
- Create `src/langgraph_instrumentation/__init__.py` with an intentionally small
  public API; expand exports only as features land.
- Move demo applications to `examples/` and ensure they import the installed
  package rather than `src.*` or top-level namespace packages.
- Separate core, OTLP, example, and development dependencies into extras.
- Remove obsolete generated `build/`, `*.egg-info`, bytecode, and generated
  trace files from version control where applicable; expand `.gitignore`.
- Move the useful content from `src/README.md` into the root README.
- Add test, lint, formatting, and static type-check configuration.
- Add baseline tests that characterize current Perfetto event categories,
  nesting, flows, counters, and errors before the old module is removed.
- Fix examples' broken imports, missing logger method usage, and ignored model
  argument as part of moving them.
- Create scoped `AGENT.md` files when the `adapters/`, `stores/`, and
  `exporters/` directories are introduced. Each must document local dependency
  rules and required tests.

**Acceptance criteria**

- A wheel builds and installs in a clean environment.
- `import langgraph_instrumentation` succeeds from outside the repository.
- Unit tests do not require OpenAI, Ollama, network access, or credentials.
- Ruff, type checking, and pytest commands are documented and passing.
- Generated artifacts no longer appear as normal source files.
- The current trace behavior has executable characterization coverage.

### FRK-02: Neutral Trace Domain, Clocks, and Identifiers

**Goal**

Define an immutable, exporter-independent representation of collected data.

**Implementation**

- Add typed models for `Trace`, `Span`, `SpanEvent`, `MetricPoint`,
  `TraceBundle`, `TraceSummary`, and `TraceQuery`.
- Add enums for span kind and status, including graph, node, chain, LLM, tool,
  retriever, custom, success, error, cancelled, and abandoned values.
- Use JSON-compatible attribute values and validate values at construction.
- Represent canonical trace IDs as 128-bit values and span IDs as 64-bit
  values, with stable hexadecimal serialization.
- Retain callback UUIDs and LangGraph execution identifiers as attributes.
- Add `Clock` and `IdGenerator` protocols plus system and deterministic test
  implementations.
- Record Unix nanoseconds for interoperability and monotonic nanoseconds while
  spans are active for duration computation.
- Include a neutral execution-lane field that Perfetto may map to a thread ID.
- Keep models independent from LangChain, LangGraph, Perfetto, SQLite, and OTel.

**Acceptance criteria**

- Models round-trip through a documented JSON representation.
- IDs satisfy OpenTelemetry width and non-zero requirements.
- Deterministic clock/ID tests can reproduce identical trace bundles.
- Invalid timestamps, IDs, statuses, and attribute values fail clearly.
- No domain module imports an adapter, store implementation, or exporter.

### FRK-03: Capture, Sanitization, Redaction, and Usage Policies

**Goal**

Make payload collection safe, bounded, provider-tolerant, and independently
testable.

**Implementation**

- Add `CapturePolicy` modes for none, metadata-only, bounded content, and full
  content.
- Extract a cycle-safe `Sanitizer` with configurable depth, item, string, and
  byte limits.
- Add `RedactionPolicy` with default case-insensitive secret keys and custom
  key/predicate support.
- Add `MessageProjector` to normalize LangChain messages and replace repeated
  message bodies with bounded references.
- Add `UsageExtractor` to normalize token usage from current supported
  `LLMResult` and message metadata shapes.
- Add configurable metadata projection rather than a hard-coded whitelist.
- Ensure sanitization and redaction run before values reach stores, logs, or
  exporters.
- Define behavior for unsupported objects without relying on exporter-side
  `default=str` fallbacks.

**Acceptance criteria**

- Tests cover cycles, deep structures, oversized collections/strings, bytes,
  secret keys, custom redactors, and unsupported objects.
- Metadata-only capture does not persist prompt, response, state, or tool body.
- Usage extraction handles absent, partial, and provider-specific token data.
- Policy behavior is deterministic and does not mutate caller-owned objects.

### FRK-04: Trace Recorder and Lifecycle Engine

**Goal**

Centralize span lifecycle, parent resolution, concurrency safety, aggregation,
and processor dispatch outside framework adapters.

**Implementation**

- Implement `TraceRecorder.start_span`, `add_event`, `record_metric`,
  `end_span`, `force_flush`, and `close`.
- Track active spans by callback correlation ID while assigning canonical trace
  and span IDs.
- Resolve roots and parents even when callbacks end on a different thread or
  asyncio task.
- Handle duplicate starts, unknown ends, out-of-order callbacks, errors,
  cancellation, and shutdown with active spans.
- Mark unfinished spans abandoned during orderly shutdown.
- Aggregate token usage under appropriate synchronization and emit neutral
  metric records.
- Assign stable execution lanes without embedding Perfetto-specific offsets in
  the public model.
- Dispatch completed immutable spans to processors without holding recorder
  locks.
- Make instrumentation failures non-fatal by default, with strict mode for
  tests and diagnostics.

**Acceptance criteria**

- Lifecycle tests cover roots, nesting, fan-out/fan-in, errors, cancellation,
  duplicate/orphan callbacks, and close with active spans.
- Threaded and asyncio concurrency tests pass without lost spans or counters.
- Recorder tests use fake stores/processors and do not import LangGraph.
- Processor/store errors follow the documented strict/non-strict policy.
- `close()` and `force_flush()` are idempotent.

### FRK-05: TraceStore Contract and In-Memory Implementation

**Goal**

Create the persistence extension point and a fast default implementation.

**Implementation**

- Define a synchronous `TraceStore` protocol for starting and completing spans,
  recording events and metrics, querying traces, listing summaries, retention,
  and closing resources.
- Define atomicity and error semantics for each operation.
- Implement `MemoryTraceStore` with thread-safe storage grouped by trace ID.
- Return snapshots or immutable records so callers cannot mutate store state.
- Add configurable limits and an explicit, documented eviction policy.
- Build a reusable store contract test suite that accepts a store factory.
- Document the requirements a future PostgreSQL implementation must satisfy.
- Add `src/langgraph_instrumentation/stores/AGENT.md` with schema independence,
  transaction, concurrency, and contract-test rules.

**Acceptance criteria**

- The memory implementation passes the complete store contract suite.
- Queries return spans in deterministic order.
- Concurrent writes and reads preserve valid trace state.
- Eviction never leaves silently corrupted partial references.
- No consumer depends on memory-store internals.

### FRK-06: Thin LangGraph Callback Adapter

**Goal**

Translate LangChain/LangGraph callbacks into recorder operations without owning
storage, export, or lifecycle policy.

**Implementation**

- Implement `LangGraphCallbackHandler` as a `BaseCallbackHandler` adapter.
- Cover chain, graph root/node, chat model, LLM, tool, error, and cancellation
  lifecycle callbacks.
- Add retriever and streaming-token support where available in the supported
  LangChain version; make high-volume token capture configurable.
- Normalize names, kinds, callback IDs, parent IDs, model data, tool-call IDs,
  and LangGraph metadata before calling the recorder.
- Share private translation functions between sync and async callbacks instead
  of duplicating business logic.
- Keep active-run state, token totals, locks, and execution-lane state in the
  recorder, not the adapter.
- Add `src/langgraph_instrumentation/adapters/AGENT.md` describing supported
  callback signatures, version boundaries, and adapter-only responsibilities.

**Acceptance criteria**

- Deterministic fake-model/tool LangGraph tests produce correct root, node, LLM,
  and tool relationships.
- Sync and async graphs produce equivalent semantic traces.
- Fan-out/fan-in execution records sibling spans with one common parent.
- Adapter tests assert recorder calls independently of store/export behavior.
- The module has no imports from concrete stores or exporters.

### FRK-07: Perfetto Exporter and Compatibility Fixtures

**Goal**

Retain Perfetto as a first-class output while isolating Trace Event JSON details
from collection and storage.

**Implementation**

- Implement `PerfettoExporter` over `TraceBundle`.
- Map completed spans to `X` events, span events to instant events, metrics to
  `C` events, and process/lane metadata to `M` events.
- Derive flow arrows from parent-child relationships rather than persisting
  Perfetto flow records.
- Map neutral execution lanes to stable `tid` values and deterministic sort
  indexes.
- Support paths and text file objects, pretty/compact output, and atomic path
  replacement.
- Validate all serialized values before writing; remove `default=str` behavior.
- Add deterministic golden fixtures for sequential, nested, fan-out/fan-in,
  tools, token usage, and failures.
- Add `src/langgraph_instrumentation/exporters/AGENT.md` documenting exporter
  isolation, optional dependencies, and fixture expectations.

**Acceptance criteria**

- Produced JSON is accepted by Perfetto's Trace Event importer.
- Existing useful visual behavior is represented by characterization fixtures.
- Re-exporting the same bundle produces byte-identical compact JSON.
- Interrupted export cannot leave a partially replaced destination file.
- The exporter works with bundles loaded from either memory or SQLite.

### FRK-08: SQLite Trace Store

**Goal**

Provide durable local trace storage behind the same contract as the memory
store.

**Implementation**

- Implement `SQLiteTraceStore` using the standard `sqlite3` module.
- Create versioned schema management for traces, spans, events, and metrics.
- Store attributes as validated JSON and preserve all neutral model fields.
- Add indexes for trace ID, parent span ID, status, and start time.
- Make span completion transactional and define behavior after process restart.
- Enable and document WAL and busy-timeout settings.
- Avoid sharing unsafe connection/cursor state across threads.
- Implement retention/deletion without orphaned events or metrics.
- Run the shared store contract suite against temporary SQLite databases.
- Add reopen, migration, concurrent writer, rollback, and persistence tests.

**Acceptance criteria**

- SQLite passes every generic store contract test used by the memory store.
- A stored trace can be reopened and exported to the same semantic Perfetto
  output.
- Failed transactions do not produce partially completed spans.
- Schema version mismatches fail with actionable errors.
- Replacing `MemoryTraceStore` with `SQLiteTraceStore` requires no adapter or
  exporter changes.

### FRK-09: Span Processors and OpenTelemetry OTLP Export

**Goal**

Export completed neutral spans to OTLP while keeping OpenTelemetry optional.

**Implementation**

- Define `SpanExporter` and `ExportResult` contracts with export, flush, and
  shutdown semantics.
- Implement simple, batching, and composite processors.
- Bound queues and define overflow, timeout, retry, shutdown, and error policy.
- Implement an OTLP exporter using supported OpenTelemetry SDK APIs.
- Map canonical IDs, parentage, kind, status, timestamps, exceptions, model
  information, token usage, tool attributes, and span events.
- Use applicable stable `gen_ai.*` semantic conventions while retaining
  namespaced `langgraph.*` attributes.
- Package OTLP dependencies behind an optional extra and provide a clear error
  when it is unavailable.
- Export token usage as span attributes/events initially; defer a separate OTel
  Metrics pipeline until trace delivery is stable.
- Test against an in-memory OTel exporter and add an optional mock/local
  collector integration test.

**Acceptance criteria**

- Core and Perfetto-only installs do not install or import OpenTelemetry.
- Parent-child relationships and timestamps survive OTLP mapping.
- Batch flush and shutdown deliver queued spans within documented timeouts.
- Export failures do not break agent execution unless strict mode is enabled.
- Composite processing can deliver the same completed span to multiple
  exporters without mutation.

### FRK-10: Examples, Documentation, CI, and Release Hardening

**Goal**

Make the refactored library understandable, reproducible, and releasable.

**Implementation**

- Replace the placeholder root README with installation, quickstart,
  architecture, Perfetto, SQLite, OTLP, capture-policy, and troubleshooting
  documentation.
- Provide deterministic no-network examples plus optional OpenAI/Ollama demos.
- Demonstrate memory-to-Perfetto, SQLite-to-Perfetto, and memory-plus-OTLP
  configurations.
- Document lifecycle requirements: context management, `force_flush`, and
  `close`.
- Document how to implement a PostgreSQL store using the store protocol and
  contract suite.
- Document how to add exporters and processors without modifying adapters.
- Explain differences from OpenInference auto-instrumentation.
- Add CI for lint, formatting, types, tests, coverage, package build, and clean
  wheel installation/import.
- Add a supported-version matrix for Python, LangChain, and LangGraph.
- Review all scoped `AGENT.md` files for accuracy after implementation.

**Acceptance criteria**

- Every documented snippet is exercised by tests or smoke checks.
- CI passes from a clean checkout without credentials or network LLM calls.
- Core, OTLP, examples, and development installation modes are documented.
- Package metadata contains a meaningful description and correct dependencies.
- The old prototype module and imports are removed after parity is verified.
- The epic's end-to-end acceptance scenarios pass with both memory and SQLite.

## Epic Acceptance Scenarios

The epic is complete when all of the following work through public APIs:

1. A deterministic synchronous LangGraph run is stored in memory and exported
   to valid Perfetto JSON.
2. The same graph runs asynchronously with equivalent semantic parentage.
3. A parallel graph displays separate lanes and parent-child flows in Perfetto.
4. Replacing memory with SQLite requires changing only store construction.
5. A trace persists across SQLite close/reopen and remains exportable.
6. Enabling OTLP requires only the optional extra and processor configuration.
7. Disabling payload capture prevents prompts, responses, state, and tool bodies
   from entering any store or exporter.
8. Export or store failures follow documented non-strict behavior and never
   conceal the agent's original exception.
9. All functionality is covered without requiring external model providers.

## Out of Scope for This Epic

- A PostgreSQL store implementation; this epic establishes and validates the
  contract it will implement.
- A full OpenTelemetry Metrics pipeline.
- Automatic monkey-patching comparable to OpenInference.
- A hosted trace viewer or trace-query service.
- Backward-compatible wrappers for prototype classes/import paths.

## Issue Linking Rules

- Every FRK issue must link the parent epic issue.
- Every FRK issue must list prerequisite FRK IDs and their GitHub issue links.
- A dependent issue should not be marked ready until prerequisite acceptance
  criteria are met on the integration branch.
- Pull requests should include the FRK ID in their title and link both the FRK
  issue and parent epic.
- Scope changes that alter dependency direction or public contracts require an
  update to this document and the parent epic issue.
