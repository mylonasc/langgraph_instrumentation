from collections.abc import Callable
from dataclasses import replace

import pytest

from langgraph_instrumentation import (
    MemoryTraceStore,
    MetricPoint,
    Span,
    SpanEvent,
    SpanId,
    SpanKind,
    SpanStatus,
    Trace,
    TraceId,
    TraceQuery,
    TraceStore,
)

StoreFactory = Callable[[], TraceStore]


@pytest.fixture(params=[MemoryTraceStore], ids=["memory"])
def store_factory(request: pytest.FixtureRequest) -> StoreFactory:
    return request.param


def make_trace(value: int = 1, *, start: int = 100, service: str = "test") -> Trace:
    return Trace(
        trace_id=TraceId(value),
        name=f"trace-{value}",
        service_name=service,
        start_time_unix_ns=start,
        attributes={"trace": value},
        resource_attributes={"environment": "test"},
    )


def make_span(
    trace: Trace,
    value: int = 1,
    *,
    parent_span_id: SpanId | None = None,
    start: int | None = None,
) -> Span:
    timestamp = trace.start_time_unix_ns if start is None else start
    return Span(
        trace_id=trace.trace_id,
        span_id=SpanId(value),
        parent_span_id=parent_span_id,
        name=f"span-{value}",
        kind=SpanKind.CUSTOM,
        start_time_unix_ns=timestamp,
        start_time_monotonic_ns=timestamp,
    )


def finish(span: Span, *, end: int | None = None, status: SpanStatus = SpanStatus.SUCCESS) -> Span:
    timestamp = span.start_time_unix_ns + 10 if end is None else end
    return replace(
        span,
        end_time_unix_ns=timestamp,
        end_time_monotonic_ns=timestamp,
        status=status,
    )


def test_contract_records_and_returns_a_deterministic_snapshot(store_factory: StoreFactory) -> None:
    store = store_factory()
    trace = make_trace()
    root = make_span(trace)
    child = make_span(trace, 2, parent_span_id=root.span_id, start=101)
    store.start_span(trace, root)
    store.start_span(trace, child)
    event = SpanEvent(trace.trace_id, child.span_id, "event", 102, 102, {"key": "value"})
    metric = MetricPoint(trace.trace_id, "tokens", 103, 103, {"total": 2}, child.span_id)
    store.record_event(event)
    store.record_metric(metric)
    store.complete_span(replace(finish(child), events=(event,)))
    store.complete_span(finish(root, end=120))

    bundle = store.get_trace(trace.trace_id)
    assert bundle is not None
    assert bundle.trace.end_time_unix_ns == 120
    assert [span.span_id for span in bundle.spans] == [root.span_id, child.span_id]
    assert bundle.spans[1].events == (event,)
    assert bundle.metrics == (metric,)
    assert store.get_trace(TraceId(999)) is None


def test_contract_reopens_finalized_trace_for_delayed_child(store_factory: StoreFactory) -> None:
    store = store_factory()
    trace = make_trace()
    root = make_span(trace)
    store.start_span(trace, root)
    store.complete_span(finish(root))
    finalized = store.get_trace(trace.trace_id)
    assert finalized is not None and finalized.trace.end_time_unix_ns is not None

    child = make_span(trace, 2, parent_span_id=root.span_id, start=120)
    store.start_span(trace, child)
    reopened = store.get_trace(trace.trace_id)
    assert reopened is not None and reopened.trace.end_time_unix_ns is None
    store.complete_span(finish(child))

    bundle = store.get_trace(trace.trace_id)
    assert bundle is not None
    assert len(bundle.spans) == 2
    assert bundle.trace.end_time_unix_ns == 130


def test_contract_lists_filters_and_summarizes_deterministically(
    store_factory: StoreFactory,
) -> None:
    store = store_factory()
    for value, start, service, status in (
        (1, 100, "alpha", SpanStatus.SUCCESS),
        (2, 200, "beta", SpanStatus.ERROR),
        (3, 200, "beta", SpanStatus.SUCCESS),
    ):
        trace = make_trace(value, start=start, service=service)
        span = make_span(trace, value)
        store.start_span(trace, span)
        store.complete_span(finish(span, status=status))

    assert [item.trace_id for item in store.list_traces()] == [TraceId(3), TraceId(2), TraceId(1)]
    matches = store.list_traces(TraceQuery(service_name="beta", status=SpanStatus.ERROR))
    assert len(matches) == 1
    assert matches[0].trace_id == TraceId(2)
    assert matches[0].span_count == 1


def test_contract_deletion_and_retention_remove_whole_traces(store_factory: StoreFactory) -> None:
    store = store_factory()
    for value, start in ((1, 100), (2, 200)):
        trace = make_trace(value, start=start)
        span = make_span(trace, value)
        store.start_span(trace, span)
        store.record_metric(MetricPoint(trace.trace_id, "count", start, start, {"value": 1}))
        store.complete_span(finish(span))

    assert store.delete_traces(before_unix_ns=150) == 1
    assert store.get_trace(TraceId(1)) is None
    assert store.delete_trace(TraceId(2)) is True
    assert store.delete_trace(TraceId(2)) is False
    assert store.list_traces() == ()


def test_contract_failures_are_atomic_and_close_is_idempotent(store_factory: StoreFactory) -> None:
    store = store_factory()
    trace = make_trace()
    root = make_span(trace)
    store.start_span(trace, root)
    before = store.get_trace(trace.trace_id)

    with pytest.raises(ValueError, match="already exists"):
        store.start_span(trace, root)
    with pytest.raises(ValueError, match="parent span"):
        store.start_span(trace, make_span(trace, 2, parent_span_id=SpanId(99)))
    with pytest.raises(ValueError, match="span does not exist"):
        store.record_event(SpanEvent(trace.trace_id, SpanId(99), "event", 101, 101))
    assert store.get_trace(trace.trace_id) == before

    assert store.close() is True
    assert store.close() is True
    with pytest.raises(RuntimeError, match="closed"):
        store.get_trace(trace.trace_id)
