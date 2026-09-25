# LangGraph Instrumentation

Instrumentation for understanding LangGraph agent execution locally in
[Perfetto](https://ui.perfetto.dev), with a staged refactor toward interchangeable
trace stores and observability exporters.

The current release contains the characterized prototype API. The modular
architecture, SQLite persistence, and OTLP support are tracked in
[`EPIC_REFACTOR.md`](EPIC_REFACTOR.md) and GitHub epic
[#1](https://github.com/mylonasc/langgraph_instrumentation/issues/1).

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management,
virtual environments, locking, and builds. Python 3.12 or newer is required.

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv build --out-dir dist
```

Install optional example dependencies with:

```bash
uv sync --group dev --extra examples
```

The future OTLP implementation has its own optional dependency set:

```bash
uv sync --extra otlp
```

`uv.lock` is committed so local development and CI resolve the same dependency
versions.

## Current API

```python
from langgraph_instrumentation import (
    LangGraphInstrumentationHandler,
    PerfettoLogger,
    PerfettoTracer,
)

tracer = PerfettoTracer(process_name="langgraph-agent")
logger = PerfettoLogger()
callback = LangGraphInstrumentationHandler(tracer=tracer, logger=logger)

# Pass callback through a LangGraph invocation config:
# app.invoke(inputs, config={"callbacks": [callback]})

tracer.save("trace.json")
```

Open the generated JSON file in `ui.perfetto.dev`.

The public API above is retained only while its behavior is characterized. The
epic intentionally introduces a clean-break recorder/store/exporter API rather
than permanent compatibility wrappers.

## Examples

Provider-backed demonstrations live under `examples/` and are not imported by
the library:

```bash
uv run --extra examples python examples/run_parallel.py
```

The parallel example expects Ollama and the configured model to be available.
The ReAct example expects an OpenAI credential. Tests never require either.

Example Perfetto output:

![Perfetto trace](src/assets/perfetto_example.png)

Corresponding graph:

![Parallel graph](src/assets/graph.png)

## Planned Architecture

The refactor separates four concerns:

- A LangGraph callback adapter translates framework callbacks.
- A recorder manages trace and span lifecycle.
- A `TraceStore` persists neutral records in memory, SQLite, or future stores.
- Exporters translate neutral records to Perfetto or OTLP.

See `EPIC_REFACTOR.md` for feature dependencies, design constraints, and
acceptance scenarios. Deferred non-blocking work is tracked in
[FRK-Techdept #12](https://github.com/mylonasc/langgraph_instrumentation/issues/12).

## OpenInference Alternative

[OpenInference](https://github.com/Arize-ai/openinference) provides automatic
OpenTelemetry instrumentation by patching LangChain/LangGraph integrations. It
is a useful alternative when local Perfetto export, explicit trace storage, or
the component boundaries planned by this project are not required.
