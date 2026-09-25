import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, cast

import pytest
from test_store_contract import finish, make_span, make_trace

from langgraph_instrumentation import (
    MemoryTraceStore,
    MetricPoint,
    SpanEvent,
    SpanStatus,
    TraceBundle,
    TraceId,
    TraceQuery,
)


def test_max_traces_evicts_oldest_finalized_whole_trace() -> None:
    store = MemoryTraceStore(max_traces=2)
    for value, start in ((2, 100), (1, 100), (3, 300)):
        trace = make_trace(value, start=start)
        span = make_span(trace, value)
        store.start_span(trace, span)
        store.complete_span(finish(span))

    assert store.get_trace(TraceId(1)) is None
    assert [summary.trace_id for summary in store.list_traces()] == [TraceId(3), TraceId(2)]


def test_capacity_never_evicts_an_active_trace() -> None:
    store = MemoryTraceStore(max_traces=1)
    active_trace = make_trace()
    store.start_span(active_trace, make_span(active_trace))
    new_trace = make_trace(2)

    with pytest.raises(RuntimeError, match="active traces"):
        store.start_span(new_trace, make_span(new_trace, 2))

    assert store.get_trace(active_trace.trace_id) is not None
    assert store.get_trace(new_trace.trace_id) is None


def test_failed_new_trace_validation_does_not_trigger_eviction() -> None:
    store = MemoryTraceStore(max_traces=1)
    retained = make_trace()
    retained_span = make_span(retained)
    store.start_span(retained, retained_span)
    store.complete_span(finish(retained_span))
    invalid = make_trace(2)

    with pytest.raises(ValueError, match="parent span"):
        store.start_span(invalid, make_span(invalid, 2, parent_span_id=retained_span.span_id))

    assert store.get_trace(retained.trace_id) is not None
    assert store.get_trace(invalid.trace_id) is None


def test_snapshots_do_not_change_after_later_writes() -> None:
    store = MemoryTraceStore()
    trace = make_trace()
    root = make_span(trace)
    store.start_span(trace, root)
    snapshot = store.get_trace(trace.trace_id)
    store.complete_span(finish(root))

    assert snapshot is not None
    assert snapshot.trace.end_time_unix_ns is None
    assert snapshot.spans[0].status is SpanStatus.UNSET


def test_concurrent_trace_lifecycles_and_queries_preserve_complete_state() -> None:
    store = MemoryTraceStore()
    barrier = threading.Barrier(16)

    def record(value: int) -> None:
        trace = make_trace(value, start=value * 100)
        span = make_span(trace, value)
        barrier.wait()
        store.start_span(trace, span)
        assert store.get_trace(trace.trace_id) is not None
        store.complete_span(finish(span))

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(record, range(1, 17)))

    summaries = store.list_traces()
    assert len(summaries) == 16
    assert [item.trace_id for item in summaries] == [TraceId(value) for value in range(16, 0, -1)]
    assert all(item.status is SpanStatus.SUCCESS for item in summaries)


def test_concurrent_tied_events_and_metrics_use_canonical_order() -> None:
    store = MemoryTraceStore()
    trace = make_trace()
    span = make_span(trace)
    store.start_span(trace, span)
    events = tuple(
        SpanEvent(trace.trace_id, span.span_id, "event", 100, 100, {"order": value})
        for value in range(16)
    )
    metrics = tuple(
        MetricPoint(
            trace.trace_id,
            "metric",
            100,
            100,
            {"value": value},
            span.span_id,
            {"order": value},
        )
        for value in range(16)
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(store.record_event, reversed(events)))
        list(executor.map(store.record_metric, reversed(metrics)))

    snapshot = store.get_trace(trace.trace_id)
    expected = TraceBundle(trace, (replace(span, events=events),), metrics)
    assert snapshot is not None
    assert snapshot.spans[0].events == expected.spans[0].events
    assert snapshot.metrics == expected.metrics


def test_close_timeout_does_not_mark_store_closed() -> None:
    store = MemoryTraceStore()
    lock_held = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with store._lock:
            lock_held.set()
            assert release.wait(1)

    with ThreadPoolExecutor(max_workers=1) as executor:
        holding = executor.submit(hold_lock)
        assert lock_held.wait(1)
        assert store.close(0) is False
        release.set()
        holding.result()

    assert store.list_traces() == ()
    assert store.close(0) is True


def test_closed_error_takes_precedence_for_every_data_operation() -> None:
    store = MemoryTraceStore()
    trace = make_trace()
    span = make_span(trace)
    store.close()
    operations = (
        lambda: store.start_span(trace, finish(span)),
        lambda: store.record_event(SpanEvent(trace.trace_id, span.span_id, "event", 100, 100)),
        lambda: store.record_metric(MetricPoint(trace.trace_id, "metric", 100, 100, {"v": 1})),
        lambda: store.complete_span(span),
        lambda: store.get_trace(trace.trace_id),
        lambda: store.list_traces(TraceQuery()),
        lambda: store.delete_trace(trace.trace_id),
        lambda: store.delete_traces(before_unix_ns=-1),
    )

    for operation in operations:
        with pytest.raises(RuntimeError, match="closed"):
            cast(Any, operation)()
