# Agent Guide

## Purpose

This repository is being refactored from a prototype into a modular LangGraph
instrumentation library. Read `EPIC_REFACTOR.md` before implementing any FRK
feature. Work against the feature's acceptance criteria and dependency list.

## Repository Rules

- Use the `src/langgraph_instrumentation/` package for library code.
- Keep examples outside the library package.
- Do not restore prototype import paths; this refactor intentionally makes a
  clean API break.
- Preserve existing unrelated worktree changes. Generated artifacts currently
  present in the worktree must not be silently committed or deleted.
- Never use real credentials or require paid/network LLM calls in tests.
- Treat prompt, response, state, metadata, and tool payloads as potentially
  sensitive.

## Dependency Direction

- Neutral models import no adapter, store implementation, or exporter.
- Stores depend only on neutral models and store-level protocols.
- Exporters depend only on neutral models, exporter protocols, and optional SDKs.
- Framework adapters depend on recorder-facing interfaces, never concrete stores
  or exporters.
- The recorder coordinates stores and processors but contains no Perfetto,
  SQLite, OTLP, or demo-specific behavior.
- Optional dependencies must not be imported by the core package at module
  import time.

## Implementation Practices

- Prefer small protocols and composition over concrete base-class hierarchies.
- Keep public APIs typed and deliberately exported from `__init__.py`.
- Use immutable or snapshot records across component boundaries.
- Sanitize and redact before persistence or export.
- Do not hold recorder/store locks while invoking user code or exporters.
- Make flush and close behavior explicit, bounded, and idempotent.
- Preserve an agent application's original exception if instrumentation also
  fails.
- Use deterministic clocks and ID generators in tests.

## Required Verification

Each feature must add tests at its own abstraction boundary. Before considering
an FRK complete, run the repository's configured formatter, linter, type checker,
unit tests, and package build. If a tool is not configured yet, FRK-01 owns that
configuration; do not introduce an undocumented replacement.

Tests must not depend on:

- OpenAI or Ollama availability
- external OTLP collectors in the default suite
- network access
- local developer environment variables
- callback timing based on real sleeps

## Scoped Agent Guides

Create scoped `AGENT.md` files with the directory they govern:

- `src/langgraph_instrumentation/adapters/AGENT.md` in FRK-06
- `src/langgraph_instrumentation/stores/AGENT.md` in FRK-05
- `src/langgraph_instrumentation/exporters/AGENT.md` in FRK-07

The nearest guide supplements this file and should document local protocols,
forbidden dependencies, and the contract or fixture suite that implementations
must pass. FRK-10 must review all guides against the final architecture.

## Pull Requests

- Include the feature ID, for example `FRK-05`, in the title.
- Link the feature issue and parent epic.
- State which acceptance criteria are satisfied.
- Identify deferred work explicitly instead of adding speculative compatibility
  or abstraction layers.
- Record non-blocking deferred work in
  [FRK-Techdept #12](https://github.com/mylonasc/langgraph_instrumentation/issues/12)
  with its origin, impact, rationale, priority, and completion condition. Never
  use technical debt to bypass correctness, security, data-loss,
  public-contract, or required-test failures.
