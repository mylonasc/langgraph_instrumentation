import ast
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from langgraph_instrumentation import (
    DeterministicClock,
    DeterministicIdGenerator,
    ExecutionLane,
    MetricPoint,
    Span,
    SpanEvent,
    SpanId,
    SpanKind,
    SpanStatus,
    Trace,
    TraceBundle,
    TraceId,
    TraceQuery,
    TraceSummary,
)


def _bundle() -> TraceBundle:
    trace_id = TraceId(1)
    root_span_id = SpanId(1)
    child_span_id = SpanId(2)
    child_event = SpanEvent(
        trace_id=trace_id,
        span_id=child_span_id,
        name="response.received",
        time_unix_ns=1_175,
        time_monotonic_ns=275,
        attributes={"provider": "test"},
    )
    child = Span(
        trace_id=trace_id,
        span_id=child_span_id,
        parent_span_id=root_span_id,
        name="LLM: deterministic",
        kind=SpanKind.LLM,
        start_time_unix_ns=1_150,
        start_time_monotonic_ns=250,
        end_time_unix_ns=1_200,
        end_time_monotonic_ns=300,
        status=SpanStatus.SUCCESS,
        attributes={"gen_ai.request.model": "deterministic"},
        events=(child_event,),
        execution_lane=ExecutionLane("task-1", "agent task", 1),
    )
    root = Span(
        trace_id=trace_id,
        span_id=root_span_id,
        name="Graph: demo",
        kind=SpanKind.GRAPH,
        start_time_unix_ns=1_100,
        start_time_monotonic_ns=200,
        end_time_unix_ns=1_300,
        end_time_monotonic_ns=400,
        status=SpanStatus.SUCCESS,
    )
    metric = MetricPoint(
        trace_id=trace_id,
        span_id=child_span_id,
        name="token.usage",
        time_unix_ns=1_190,
        time_monotonic_ns=290,
        values={"input": 2, "output": 3, "total": 5},
        attributes={"unit": "token"},
    )
    trace = Trace(
        trace_id=trace_id,
        name="demo",
        service_name="test-agent",
        start_time_unix_ns=1_100,
        end_time_unix_ns=1_300,
        attributes={"langgraph.thread_id": "thread-1"},
        resource_attributes={"deployment.environment": "test"},
    )
    return TraceBundle(trace=trace, spans=(child, root), metrics=(metric,))


def test_trace_bundle_round_trips_through_json() -> None:
    bundle = _bundle()

    serialized = json.dumps(bundle.to_dict(), sort_keys=True)
    restored = TraceBundle.from_dict(json.loads(serialized))

    assert restored == bundle
    assert [span.span_id for span in restored.spans] == [SpanId(1), SpanId(2)]
    assert restored.spans[0].duration_ns == 200
    assert restored.spans[1].events[0].name == "response.received"


def test_summary_and_query_round_trip() -> None:
    summary = TraceSummary(
        trace_id=TraceId(5),
        name="demo",
        service_name="test-agent",
        start_time_unix_ns=100,
        end_time_unix_ns=200,
        span_count=3,
        status=SpanStatus.ERROR,
    )
    query = TraceQuery(
        start_time_unix_ns=50,
        end_time_unix_ns=250,
        service_name="test-agent",
        status=SpanStatus.ERROR,
        limit=10,
    )

    assert TraceSummary.from_dict(summary.to_dict()) == summary
    assert TraceQuery.from_dict(query.to_dict()) == query


def test_deterministic_components_reproduce_identical_bundles() -> None:
    def build_bundle() -> TraceBundle:
        clock = DeterministicClock(unix_time_ns=1_000, monotonic_time_ns=100)
        identifiers = DeterministicIdGenerator()
        trace_id = identifiers.new_trace_id()
        start = clock.now()
        span_id = identifiers.new_span_id()
        end = clock.advance(50)
        return TraceBundle(
            trace=Trace(
                trace_id=trace_id,
                name="deterministic",
                service_name="test",
                start_time_unix_ns=start.unix_time_ns,
                end_time_unix_ns=end.unix_time_ns,
            ),
            spans=(
                Span(
                    trace_id=trace_id,
                    span_id=span_id,
                    name="root",
                    kind=SpanKind.GRAPH,
                    start_time_unix_ns=start.unix_time_ns,
                    start_time_monotonic_ns=start.monotonic_time_ns,
                    end_time_unix_ns=end.unix_time_ns,
                    end_time_monotonic_ns=end.monotonic_time_ns,
                    status=SpanStatus.SUCCESS,
                ),
            ),
        )

    assert build_bundle() == build_bundle()


def test_records_deeply_freeze_caller_owned_attributes() -> None:
    source = {"nested": {"items": [1, 2]}}
    trace = Trace(
        trace_id=TraceId(1),
        name="immutable",
        service_name="test",
        start_time_unix_ns=1,
        attributes=source,
    )
    source["nested"]["items"].append(3)  # type: ignore[index, union-attr]

    assert trace.to_dict()["attributes"] == {"nested": {"items": [1, 2]}}
    with pytest.raises(TypeError):
        cast(Any, trace.attributes)["new"] = "value"
    with pytest.raises(FrozenInstanceError):
        cast(Any, trace).name = "changed"


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    [
        (
            lambda: Trace(
                trace_id=TraceId(1),
                name="trace",
                service_name="service",
                start_time_unix_ns=-1,
            ),
            ValueError,
            "non-negative",
        ),
        (
            lambda: Trace(
                trace_id=TraceId(1),
                name="trace",
                service_name="service",
                start_time_unix_ns=2,
                end_time_unix_ns=1,
            ),
            ValueError,
            "cannot precede",
        ),
        (
            lambda: Span(
                trace_id=TraceId(1),
                span_id=SpanId(1),
                name="active",
                kind=SpanKind.CHAIN,
                start_time_unix_ns=1,
                start_time_monotonic_ns=1,
                status=SpanStatus.SUCCESS,
            ),
            ValueError,
            "active span",
        ),
        (
            lambda: Trace(
                trace_id=TraceId(1),
                name="trace",
                service_name="service",
                start_time_unix_ns=1,
                attributes={"invalid": float("nan")},
            ),
            ValueError,
            "NaN or infinity",
        ),
        (
            lambda: Trace(
                trace_id=TraceId(1),
                name="trace",
                service_name="service",
                start_time_unix_ns=1,
                attributes={"invalid": b"bytes"},
            ),
            TypeError,
            "unsupported value type bytes",
        ),
        (
            lambda: TraceQuery(limit=0),
            ValueError,
            "greater than zero",
        ),
    ],
)
def test_invalid_domain_values_fail_clearly(factory, error_type, message: str) -> None:
    with pytest.raises(error_type, match=message):
        factory()


def test_invalid_serialized_enum_fails_with_allowed_values() -> None:
    data = _bundle().spans[0].to_dict()
    data["kind"] = "database"

    with pytest.raises(ValueError, match="kind must be one of"):
        Span.from_dict(data)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(parent_span_id=""),
        lambda data: data.update(status=""),
    ],
)
def test_invalid_optional_serialized_values_are_not_treated_as_absent(mutate) -> None:
    data = _bundle().spans[0].to_dict()
    mutate(data)

    with pytest.raises(ValueError):
        Span.from_dict(data)


def test_span_rejects_events_outside_its_lifetime() -> None:
    event = SpanEvent(
        trace_id=TraceId(1),
        span_id=SpanId(1),
        name="late",
        time_unix_ns=300,
        time_monotonic_ns=300,
    )
    with pytest.raises(ValueError, match="cannot follow"):
        Span(
            trace_id=event.trace_id,
            span_id=event.span_id,
            name="span",
            kind=SpanKind.CUSTOM,
            start_time_unix_ns=100,
            start_time_monotonic_ns=100,
            end_time_unix_ns=200,
            end_time_monotonic_ns=200,
            status=SpanStatus.SUCCESS,
            events=(event,),
        )


def test_bundle_rejects_cross_trace_records_and_duplicate_spans() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="duplicate span ID"):
        TraceBundle(trace=bundle.trace, spans=(bundle.spans[0], bundle.spans[0]))

    foreign_metric = MetricPoint(
        trace_id=TraceId(2),
        name="count",
        time_unix_ns=1,
        time_monotonic_ns=1,
        values={"value": 1},
    )
    with pytest.raises(ValueError, match="metrics.*share the trace ID"):
        TraceBundle(trace=bundle.trace, metrics=(foreign_metric,))


def test_event_and_metric_ties_have_canonical_total_order() -> None:
    trace = Trace(TraceId(1), "trace", "service", 100)
    events = tuple(
        SpanEvent(trace.trace_id, SpanId(1), "event", 110, 110, {"order": value})
        for value in ("b", "a")
    )
    span = Span(
        trace.trace_id,
        SpanId(1),
        "span",
        SpanKind.CUSTOM,
        100,
        100,
        end_time_unix_ns=120,
        end_time_monotonic_ns=120,
        status=SpanStatus.SUCCESS,
        events=events,
    )
    metrics = tuple(
        MetricPoint(trace.trace_id, "metric", 110, 110, {"value": 1}, attributes={"order": value})
        for value in ("b", "a")
    )

    forward = TraceBundle(trace, (span,), metrics)
    reverse = TraceBundle(trace, (Span.from_dict(span.to_dict()),), tuple(reversed(metrics)))

    assert [event.attributes["order"] for event in forward.spans[0].events] == ["a", "b"]
    assert [metric.attributes["order"] for metric in forward.metrics] == ["a", "b"]
    assert forward == reverse


def test_domain_modules_do_not_import_framework_or_backend_dependencies() -> None:
    package_root = Path(__file__).parents[1] / "src" / "langgraph_instrumentation"
    forbidden_roots = {
        "langchain",
        "langchain_core",
        "langgraph",
        "opentelemetry",
        "sqlite3",
    }

    for module_name in ("clock.py", "identifiers.py", "models.py"):
        tree = ast.parse((package_root / module_name).read_text())
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".", maxsplit=1)[0])
        assert imported_roots.isdisjoint(forbidden_roots), module_name
