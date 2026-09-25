import json
from collections.abc import Iterator
from typing import cast
from uuid import UUID

from langchain_core.outputs import LLMResult

from langgraph_instrumentation import (
    LangGraphInstrumentationHandler,
    PerfettoLogger,
    PerfettoTracer,
)


class NullLogger:
    def log_step(
        self,
        phase: str,
        category: str,
        name: str,
        details: str = "",
    ) -> None:
        pass

    def log_state(self, node_name: str, state_summary: str) -> None:
        pass


def _clock(values: list[int]) -> Iterator[int]:
    return iter(values)


def _logger() -> PerfettoLogger:
    return cast(PerfettoLogger, NullLogger())


def test_tracer_writes_perfetto_trace_event_json(tmp_path) -> None:
    tracer = PerfettoTracer(process_name="test-agent")
    tracer.add_complete_event(
        "Node: plan",
        "graph_node",
        start_ts_us=100,
        dur_us=25,
        tid=7,
        args={"result": "ok"},
    )
    tracer.add_counter("Token Usage", {"total": 3}, ts_us=125, tid=7)

    destination = tmp_path / "trace.json"
    tracer.save(str(destination), pretty=False)

    payload = json.loads(destination.read_text())
    assert set(payload) == {"traceEvents"}
    assert any(
        event["ph"] == "M" and event["name"] == "process_name" for event in payload["traceEvents"]
    )
    assert any(event["ph"] == "X" and event["dur"] == 25 for event in payload["traceEvents"])
    assert any(
        event["ph"] == "C" and event["args"]["total"] == 3 for event in payload["traceEvents"]
    )


def test_handler_preserves_parent_flow_and_complete_spans(monkeypatch) -> None:
    tracer = PerfettoTracer()
    handler = LangGraphInstrumentationHandler(tracer, _logger())
    timestamps = _clock([100, 200, 300, 400])
    monkeypatch.setattr(handler, "_now_us", lambda: next(timestamps))

    parent_id = UUID(int=1)
    child_id = UUID(int=2)
    handler.on_chain_start({}, {"value": 1}, run_id=parent_id, name="root")
    handler.on_chain_start(
        {},
        {"value": 2},
        run_id=child_id,
        parent_run_id=parent_id,
        name="child",
    )
    handler.on_chain_end({"value": 3}, run_id=child_id)
    handler.on_chain_end({"value": 4}, run_id=parent_id)

    flows = [event for event in tracer.trace_events if event["cat"] == "flow"]
    complete = [event for event in tracer.trace_events if event["ph"] == "X"]

    assert [(event["ph"], event["ts"]) for event in flows] == [("s", 100), ("f", 200)]
    assert [(event["name"], event["ts"], event["dur"]) for event in complete] == [
        ("child", 200, 100),
        ("root", 100, 300),
    ]
    assert complete[0]["args"]["parent_run_id"] == str(parent_id)


def test_handler_emits_cumulative_token_counter(monkeypatch) -> None:
    tracer = PerfettoTracer()
    handler = LangGraphInstrumentationHandler(tracer, _logger())
    timestamps = _clock([100, 150, 200])
    monkeypatch.setattr(handler, "_now_us", lambda: next(timestamps))
    run_id = UUID(int=3)

    handler.on_llm_start({}, ["hello"], run_id=run_id, invocation_params={"model": "test"})
    handler.on_llm_end(
        LLMResult(
            generations=[],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                }
            },
        ),
        run_id=run_id,
    )

    counters = [event for event in tracer.trace_events if event["ph"] == "C"]
    assert counters[-1]["args"] == {"total": 5, "input": 2, "output": 3}


def test_handler_records_errors_on_completed_span(monkeypatch) -> None:
    tracer = PerfettoTracer()
    handler = LangGraphInstrumentationHandler(tracer, _logger())
    timestamps = _clock([100, 125])
    monkeypatch.setattr(handler, "_now_us", lambda: next(timestamps))
    run_id = UUID(int=4)

    handler.on_tool_start({"name": "lookup"}, "query", run_id=run_id)
    handler.on_tool_error(RuntimeError("provider failed"), run_id=run_id)

    completed = [event for event in tracer.trace_events if event["ph"] == "X"]
    assert len(completed) == 1
    assert completed[0]["cat"] == "tool"
    assert completed[0]["args"]["error"] == "provider failed"


def test_sanitizer_is_bounded_cycle_safe_and_redacts() -> None:
    handler = LangGraphInstrumentationHandler(
        PerfettoTracer(),
        _logger(),
        sanitize_max_str=4,
        sanitize_max_items=2,
    )
    cyclic: list[object] = []
    cyclic.append(cyclic)

    assert handler._sanitize({"token": "secret", "value": "abcdef"}) == {
        "token": "<redacted>",
        "value": "abcd…<trunc>",
    }
    assert handler._sanitize(cyclic) == ["<cycle>"]
