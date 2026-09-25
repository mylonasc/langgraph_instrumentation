import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest

from langgraph_instrumentation import (
    DeterministicClock,
    DeterministicIdGenerator,
    MemoryTraceStore,
    RecorderError,
    Span,
    SpanEvent,
    SpanId,
    SpanKind,
    SpanStatus,
    Trace,
    TraceId,
    TraceRecorder,
)


class FakeStore:
    def __init__(self) -> None:
        self.started: list[Span] = []
        self.traces: list[Trace] = []
        self.events: list[SpanEvent] = []
        self.metrics = []
        self.completed: list[Span] = []
        self.flushes = 0
        self.closes = 0

    def start_span(self, trace, span):
        self.traces.append(trace)
        self.started.append(span)

    def record_event(self, event):
        self.events.append(event)

    def record_metric(self, metric):
        self.metrics.append(metric)

    def complete_span(self, span):
        self.completed.append(span)

    def get_trace(self, trace_id):
        return None

    def list_traces(self, query=None):
        return ()

    def delete_trace(self, trace_id):
        return False

    def delete_traces(self, *, before_unix_ns):
        return 0

    def force_flush(self, timeout=None):
        self.flushes += 1
        return True

    def close(self, timeout=None):
        self.closes += 1
        return True


class FakeProcessor:
    def __init__(self, recorder_lock=None) -> None:
        self.spans: list[Span] = []
        self.flushes = 0
        self.shutdowns = 0
        self.recorder_lock = recorder_lock

    def on_end(self, span):
        if self.recorder_lock is not None:
            acquired = self.recorder_lock.acquire(blocking=False)
            assert acquired
            self.recorder_lock.release()
        self.spans.append(span)

    def force_flush(self, timeout=None):
        self.flushes += 1
        return True

    def shutdown(self, timeout=None):
        self.shutdowns += 1
        return True


def make_recorder(*, strict=False, processors=()):
    store = FakeStore()
    recorder = TraceRecorder(
        store,
        processors=processors,
        clock=DeterministicClock(unix_time_ns=1_000, monotonic_time_ns=100),
        id_generator=DeterministicIdGenerator(),
        strict=strict,
    )
    return recorder, store


def test_root_nested_fanout_and_parent_resolution_after_parent_end() -> None:
    recorder, store = make_recorder()
    root = recorder.start_span("root", "graph", SpanKind.GRAPH, lane_id="graph")
    left = recorder.start_span("left", "left", SpanKind.NODE, parent_correlation_id="root")
    right = recorder.start_span(
        "right", "right", SpanKind.NODE, parent_correlation_id="root", lane_id="worker"
    )
    recorder.end_span("root")
    late = recorder.start_span("late", "late", SpanKind.TOOL, parent_correlation_id="root")

    assert root is not None and left is not None and right is not None and late is not None
    assert left.parent_span_id == right.parent_span_id == late.parent_span_id == root.span_id
    assert left.trace_id == right.trace_id == late.trace_id == root.trace_id
    assert left.execution_lane == root.execution_lane
    assert right.execution_lane != root.execution_lane
    assert right.execution_lane is not None and right.execution_lane.sort_index == 1
    assert store.completed[0].span_id == root.span_id


def test_recorder_creates_and_reuses_neutral_trace_metadata() -> None:
    store = FakeStore()
    recorder = TraceRecorder(
        store,
        clock=DeterministicClock(unix_time_ns=1_000, monotonic_time_ns=100),
        id_generator=DeterministicIdGenerator(),
        service_name="agent-service",
        resource_attributes={"environment": "test"},
    )
    recorder.start_span(
        "root",
        "root span",
        SpanKind.GRAPH,
        trace_name="agent run",
        trace_attributes={"thread.id": "thread-1"},
    )
    recorder.end_span("root")
    recorder.start_span("late", "late child", SpanKind.NODE, parent_correlation_id="root")

    assert len(store.traces) == 2
    assert store.traces[0] is store.traces[1]
    assert store.traces[0].name == "agent run"
    assert store.traces[0].service_name == "agent-service"
    assert store.traces[0].attributes["thread.id"] == "thread-1"
    assert store.traces[0].resource_attributes["environment"] == "test"


def test_events_metrics_terminal_states_and_cumulative_values() -> None:
    recorder, store = make_recorder()
    recorder.start_span("root", "graph", SpanKind.GRAPH)
    event = recorder.add_event("root", "checkpoint", attributes={"step": 1})
    first = recorder.record_metric("root", "tokens", {"input": 2}, cumulative=True)
    second = recorder.record_metric("root", "tokens", {"input": 3, "output": 1}, cumulative=True)
    completed = recorder.end_span(
        "root", status=SpanStatus.ERROR, status_description="boom", attributes={"failed": True}
    )

    assert event is not None and completed is not None and first is not None and second is not None
    assert completed.events == (event,)
    assert completed.status is SpanStatus.ERROR
    assert completed.status_description == "boom"
    assert completed.attributes["failed"] is True
    assert dict(first.values) == {"input": 2}
    assert dict(second.values) == {"input": 5, "output": 1}
    assert store.events == [event]
    assert store.completed == [completed]


@pytest.mark.parametrize("status", list(SpanStatus)[1:])
def test_all_terminal_states(status: SpanStatus) -> None:
    recorder, _ = make_recorder()
    recorder.start_span("span", "operation", SpanKind.CUSTOM)
    assert recorder.end_span("span", status=status).status is status  # type: ignore[union-attr]


def test_duplicate_unknown_and_closed_operations_are_nonfatal_or_strict() -> None:
    recorder, _ = make_recorder()
    recorder.start_span("span", "operation", SpanKind.CUSTOM)
    assert recorder.start_span("span", "duplicate", SpanKind.CUSTOM) is None
    assert recorder.add_event("missing", "event") is None
    assert recorder.end_span("missing") is None
    assert (
        recorder.start_span("orphan", "orphan", SpanKind.CUSTOM, parent_correlation_id="x") is None
    )
    recorder.close()
    assert recorder.start_span("late", "late", SpanKind.CUSTOM) is None

    strict, _ = make_recorder(strict=True)
    strict.start_span("span", "operation", SpanKind.CUSTOM)
    with pytest.raises(RecorderError, match="duplicate"):
        strict.start_span("span", "duplicate", SpanKind.CUSTOM)
    with pytest.raises(RecorderError, match="unknown"):
        strict.end_span("missing")


def test_processor_runs_outside_lock() -> None:
    processor = FakeProcessor()
    recorder, _ = make_recorder(processors=(processor,))
    processor.recorder_lock = recorder._condition
    recorder.start_span("span", "operation", SpanKind.CUSTOM)
    recorder.end_span("span")
    assert len(processor.spans) == 1


class FailingStore(FakeStore):
    def complete_span(self, span):
        raise OSError("store unavailable")


class FailingProcessor(FakeProcessor):
    def on_end(self, span):
        raise OSError("processor unavailable")


def test_store_completion_failures_do_not_commit_or_dispatch_processors() -> None:
    good = FakeProcessor()
    recorder = TraceRecorder(FailingStore(), processors=(good,))
    recorder.start_span("span", "operation", SpanKind.CUSTOM)
    completed = recorder.end_span("span")
    assert completed is None
    assert good.spans == []
    assert "span" in recorder._active

    strict_good = FakeProcessor()
    strict = TraceRecorder(FailingStore(), processors=(strict_good,), strict=True)
    strict.start_span("span", "operation", SpanKind.CUSTOM)
    with pytest.raises(RecorderError, match="complete span"):
        strict.end_span("span")
    assert strict_good.spans == []


def test_processor_failures_are_nonfatal_by_default_and_strict_after_all_dispatch() -> None:
    good = FakeProcessor()
    recorder = TraceRecorder(FakeStore(), processors=(FailingProcessor(), good))
    recorder.start_span("span", "operation", SpanKind.CUSTOM)
    completed = recorder.end_span("span")
    assert completed is not None
    assert good.spans == [completed]

    strict_good = FakeProcessor()
    strict = TraceRecorder(FakeStore(), processors=(FailingProcessor(), strict_good), strict=True)
    strict.start_span("span", "operation", SpanKind.CUSTOM)
    with pytest.raises(RecorderError, match="process completed span"):
        strict.end_span("span")
    assert len(strict_good.spans) == 1


def test_close_abandons_children_first_and_close_and_flush_are_idempotent() -> None:
    processor = FakeProcessor()
    recorder, store = make_recorder(processors=(processor,))
    root = recorder.start_span("root", "root", SpanKind.GRAPH)
    child = recorder.start_span("child", "child", SpanKind.NODE, parent_correlation_id="root")
    recorder.close()
    recorder.close()
    recorder.force_flush()

    assert root is not None and child is not None
    assert [span.span_id for span in store.completed] == [child.span_id, root.span_id]
    assert all(span.status is SpanStatus.ABANDONED for span in store.completed)
    assert store.flushes == processor.flushes == 1
    assert store.closes == processor.shutdowns == 1


@pytest.mark.parametrize("strict", [False, True])
def test_close_retries_only_abandoned_spans_rejected_by_store(strict: bool) -> None:
    class FailOnceMemoryStore(MemoryTraceStore):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 1
            self.close_calls = 0
            self.bundle_at_close = None
            self.trace_id = None

        def start_span(self, trace, span):
            self.trace_id = trace.trace_id
            super().start_span(trace, span)

        def complete_span(self, span):
            if self.failures:
                self.failures -= 1
                raise OSError("temporary completion failure")
            super().complete_span(span)

        def close(self, timeout=None):
            assert self.trace_id is not None
            self.bundle_at_close = self.get_trace(self.trace_id)
            self.close_calls += 1
            return super().close(timeout)

    store = FailOnceMemoryStore()
    processor = FakeProcessor()
    recorder = TraceRecorder(
        store,
        processors=(processor,),
        strict=strict,
        clock=DeterministicClock(unix_time_ns=1_000, monotonic_time_ns=100),
        id_generator=DeterministicIdGenerator(),
    )
    root = recorder.start_span("root", "root", SpanKind.GRAPH)
    child = recorder.start_span("child", "child", SpanKind.NODE, parent_correlation_id="root")
    assert root is not None and child is not None

    if strict:
        with pytest.raises(RecorderError, match="persist abandoned spans"):
            recorder.close()
    else:
        assert recorder.close() is False

    assert store.close_calls == 0
    assert list(recorder._active) == ["child"]
    assert [span.span_id for span in processor.spans] == [root.span_id]

    assert recorder.close() is True
    assert store.close_calls == 1
    assert store.bundle_at_close is not None
    assert [span.span_id for span in store.bundle_at_close.spans] == [root.span_id, child.span_id]
    assert all(span.status is SpanStatus.ABANDONED for span in store.bundle_at_close.spans)
    assert [span.span_id for span in processor.spans] == [root.span_id, child.span_id]

    assert recorder.close() is True
    assert store.close_calls == 1


def test_concurrent_close_callers_share_retryable_attempt_result() -> None:
    class BlockingFailOnceStore(MemoryTraceStore):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.failures = 1
            self.close_calls = 0

        def complete_span(self, span):
            if self.failures:
                self.failures -= 1
                self.entered.set()
                assert self.release.wait(1)
                raise OSError("temporary completion failure")
            super().complete_span(span)

        def close(self, timeout=None):
            self.close_calls += 1
            return super().close(timeout)

    class TrackingRecorder(TraceRecorder):
        def __init__(self, store) -> None:
            super().__init__(store)
            self.waiters = 0
            self.waiters_lock = threading.Lock()
            self.two_waiters = threading.Event()

        def _wait_for_close(self, attempt, deadline, operation):
            with self.waiters_lock:
                self.waiters += 1
                if self.waiters == 2:
                    self.two_waiters.set()
            return super()._wait_for_close(attempt, deadline, operation)

    store = BlockingFailOnceStore()
    recorder = TrackingRecorder(store)
    recorder.start_span("span", "span", SpanKind.CUSTOM)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(recorder.close, 1)
        assert store.entered.wait(1)
        second = executor.submit(recorder.close, 1)
        assert recorder.two_waiters.wait(1)
        store.release.set()
        assert first.result() is False
        assert second.result() is False

    assert store.close_calls == 0
    assert recorder.close() is True
    assert store.close_calls == 1


def test_context_manager_preserves_application_exception_when_strict_close_fails() -> None:
    class CloseFailingStore(FakeStore):
        def close(self, timeout=None):
            raise OSError("close failed")

    with (
        pytest.raises(ValueError, match="application"),
        TraceRecorder(CloseFailingStore(), strict=True),
    ):
        raise ValueError("application")

    with (
        pytest.raises(RecorderError, match="close"),
        TraceRecorder(CloseFailingStore(), strict=True),
    ):
        pass


def test_threaded_cumulative_metrics_have_no_lost_updates() -> None:
    recorder, store = make_recorder()
    recorder.start_span("root", "root", SpanKind.GRAPH)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda _: recorder.record_metric("root", "tokens", {"total": 1}, cumulative=True),
                range(200),
            )
        )

    assert store.metrics[-1].values["total"] == 200
    assert len(store.metrics) == 200


@pytest.mark.asyncio
async def test_async_tasks_can_finish_spans_started_elsewhere() -> None:
    recorder, store = make_recorder()
    recorder.start_span("root", "root", SpanKind.GRAPH)
    for index in range(50):
        recorder.start_span(
            f"child-{index}",
            "child",
            SpanKind.NODE,
            parent_correlation_id="root",
            lane_id=f"lane-{index % 4}",
        )

    await asyncio.gather(*(asyncio.to_thread(recorder.end_span, f"child-{i}") for i in range(50)))

    assert len(store.completed) == 50
    assert len({span.span_id for span in store.completed}) == 50
    assert all(span.parent_span_id == store.started[0].span_id for span in store.completed)


@pytest.mark.asyncio
async def test_async_cumulative_metrics_have_no_lost_updates() -> None:
    recorder, store = make_recorder()
    recorder.start_span("root", "root", SpanKind.GRAPH)

    await asyncio.gather(
        *(
            asyncio.to_thread(
                recorder.record_metric,
                "root",
                "tokens",
                {"total": 1},
                cumulative=True,
            )
            for _ in range(100)
        )
    )

    assert store.metrics[-1].values["total"] == 100


def test_threaded_starts_and_ends_preserve_unique_ids_and_parentage() -> None:
    recorder, store = make_recorder()
    root = recorder.start_span("root", "root", SpanKind.GRAPH)
    barrier = threading.Barrier(16)

    def lifecycle(index: int) -> None:
        barrier.wait()
        recorder.start_span(
            f"span-{index}", "worker", SpanKind.CUSTOM, parent_correlation_id="root"
        )
        recorder.end_span(f"span-{index}")

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lifecycle, range(16)))

    assert root is not None
    assert len(store.completed) == 16
    assert len({span.span_id for span in store.completed}) == 16
    assert all(span.parent_span_id == root.span_id for span in store.completed)


def test_store_operations_remain_ordered_when_start_is_blocked() -> None:
    class BlockingStore(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.operations: list[str] = []
            self.start_entered = threading.Event()
            self.release_start = threading.Event()

        def start_span(self, trace, span):
            self.operations.append("start-enter")
            self.start_entered.set()
            assert self.release_start.wait(1)
            super().start_span(trace, span)
            self.operations.append("start-exit")

        def record_event(self, event):
            self.operations.append("event")
            super().record_event(event)

        def record_metric(self, metric):
            self.operations.append("metric")
            super().record_metric(metric)

        def complete_span(self, span):
            self.operations.append("complete")
            super().complete_span(span)

    store = BlockingStore()
    recorder = TraceRecorder(store)
    with ThreadPoolExecutor(max_workers=2) as executor:
        start = executor.submit(recorder.start_span, "span", "span", SpanKind.CUSTOM)
        assert store.start_entered.wait(1)
        event = executor.submit(recorder.add_event, "span", "event")
        assert store.operations == ["start-enter"]
        store.release_start.set()
        assert start.result() is not None
        assert event.result() is not None
    assert recorder.record_metric("span", "metric", {"value": 1}) is not None
    assert recorder.end_span("span") is not None

    assert store.operations == ["start-enter", "start-exit", "event", "metric", "complete"]


@pytest.mark.parametrize("strict", [False, True])
def test_store_failures_leave_each_transition_retryable_and_atomic(strict: bool) -> None:
    class FailOnceStore(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.failures = {"start", "event", "metric", "complete"}

        def _fail_once(self, operation):
            if operation in self.failures:
                self.failures.remove(operation)
                raise OSError(operation)

        def start_span(self, trace, span):
            self._fail_once("start")
            super().start_span(trace, span)

        def record_event(self, event):
            self._fail_once("event")
            super().record_event(event)

        def record_metric(self, metric):
            self._fail_once("metric")
            super().record_metric(metric)

        def complete_span(self, span):
            self._fail_once("complete")
            super().complete_span(span)

    store = FailOnceStore()
    recorder = TraceRecorder(store, strict=strict)

    def rejected(call, message):
        if strict:
            with pytest.raises(RecorderError, match=message):
                call()
        else:
            assert call() is None

    rejected(lambda: recorder.start_span("span", "span", SpanKind.CUSTOM), "start span")
    assert "span" not in recorder._active
    assert recorder.start_span("span", "span", SpanKind.CUSTOM) is not None

    rejected(lambda: recorder.add_event("span", "event"), "record event")
    assert recorder._active["span"].events == []
    event = recorder.add_event("span", "event")

    rejected(
        lambda: recorder.record_metric("span", "tokens", {"total": 1}, cumulative=True),
        "record metric",
    )
    assert recorder._metric_totals == {}
    metric = recorder.record_metric("span", "tokens", {"total": 1}, cumulative=True)

    rejected(lambda: recorder.end_span("span"), "complete span")
    assert "span" in recorder._active
    completed = recorder.end_span("span")

    assert completed is not None and event is not None and metric is not None
    assert completed.events == (event,)
    assert metric.values["total"] == 1
    assert store.completed == [completed]


def test_delayed_child_rejected_after_store_eviction_does_not_become_active() -> None:
    store = MemoryTraceStore(max_traces=1)
    recorder = TraceRecorder(
        store,
        clock=DeterministicClock(unix_time_ns=1_000, monotonic_time_ns=100),
        id_generator=DeterministicIdGenerator(),
    )
    root = recorder.start_span("root", "root", SpanKind.GRAPH)
    assert root is not None
    recorder.end_span("root")

    replacement = Trace(TraceId(99), "replacement", "test", 2_000)
    replacement_span = Span(
        replacement.trace_id,
        SpanId(99),
        "replacement",
        SpanKind.CUSTOM,
        2_000,
        2_000,
    )
    store.start_span(replacement, replacement_span)

    assert (
        recorder.start_span("delayed", "delayed", SpanKind.NODE, parent_correlation_id="root")
        is None
    )
    assert "delayed" not in recorder._active


def test_trace_context_and_totals_are_released_after_last_active_span() -> None:
    recorder, _ = make_recorder()
    root = recorder.start_span("root", "root", SpanKind.GRAPH)
    recorder.start_span("child", "child", SpanKind.NODE, parent_correlation_id="root")
    recorder.add_event("child", "payload", attributes={"body": "not retained"})
    recorder.record_metric("child", "tokens", {"total": 1}, cumulative=True)
    recorder.end_span("root")

    retained_root = recorder._completed_contexts["root"]
    assert not isinstance(retained_root, Span)
    assert not hasattr(retained_root, "attributes")
    assert not hasattr(retained_root, "events")

    late = recorder.start_span("late", "late", SpanKind.TOOL, parent_correlation_id="root")
    assert late is not None
    recorder.end_span("child")
    retained_child = recorder._completed_contexts["child"]
    assert not isinstance(retained_child, Span)
    assert not hasattr(retained_child, "attributes")
    assert not hasattr(retained_child, "events")
    recorder.end_span("late")

    assert root is not None
    assert recorder._metric_totals == {}
    delayed = recorder.start_span("delayed", "delayed", SpanKind.NODE, parent_correlation_id="root")
    assert delayed is not None
    assert delayed.trace_id == root.trace_id
    assert delayed.parent_span_id == root.span_id


def test_completed_context_cache_evicts_oldest_context_deterministically() -> None:
    recorder = TraceRecorder(completed_context_cache_size=2)
    completed = []
    for correlation_id in ("first", "second", "third"):
        recorder.start_span(correlation_id, correlation_id, SpanKind.CUSTOM)
        completed.append(recorder.end_span(correlation_id))

    assert list(recorder._completed_contexts) == ["second", "third"]
    assert (
        recorder.start_span("orphan", "orphan", SpanKind.NODE, parent_correlation_id="first")
        is None
    )
    child = recorder.start_span("child", "child", SpanKind.NODE, parent_correlation_id="second")
    assert child is not None and completed[1] is not None
    assert child.parent_span_id == completed[1].span_id


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_completed_context_cache_size_is_validated(value) -> None:
    with pytest.raises((TypeError, ValueError), match="completed_context_cache_size"):
        TraceRecorder(completed_context_cache_size=value)


def test_force_flush_waits_for_prior_store_operation() -> None:
    class BlockingEventStore(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.event_entered = threading.Event()
            self.release_event = threading.Event()
            self.flush_called = threading.Event()

        def record_event(self, event):
            self.event_entered.set()
            assert self.release_event.wait(1)
            super().record_event(event)

        def force_flush(self, timeout=None):
            self.flush_called.set()
            return super().force_flush(timeout)

    store = BlockingEventStore()
    recorder = TraceRecorder(store)
    recorder.start_span("span", "span", SpanKind.CUSTOM)
    with ThreadPoolExecutor(max_workers=2) as executor:
        event = executor.submit(recorder.add_event, "span", "event")
        assert store.event_entered.wait(1)
        flush = executor.submit(recorder.force_flush, 1)
        assert not store.flush_called.is_set()
        store.release_event.set()
        assert event.result() is not None
        assert flush.result() is True
    assert store.flush_called.is_set()


def test_force_flush_is_idempotent_without_new_records() -> None:
    processor = FakeProcessor()
    recorder, store = make_recorder(processors=(processor,))

    assert recorder.force_flush() is True
    assert recorder.force_flush() is True

    assert store.flushes == 1
    assert processor.flushes == 1


def test_concurrent_close_waits_until_shutdown_finishes() -> None:
    class BlockingCloseStore(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.close_entered = threading.Event()
            self.release_close = threading.Event()

        def close(self, timeout=None):
            self.close_entered.set()
            assert self.release_close.wait(1)
            return super().close(timeout)

    store = BlockingCloseStore()
    recorder = TraceRecorder(store)
    second_started = threading.Event()

    def second_close():
        second_started.set()
        return recorder.close(1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(recorder.close, 1)
        assert store.close_entered.wait(1)
        second = executor.submit(second_close)
        assert second_started.wait(1)
        assert not second.done()
        store.release_close.set()
        assert first.result() is True
        assert second.result() is True
    assert store.closes == 1


def test_close_deadline_finishes_without_shutting_down_active_processor() -> None:
    class BlockingProcessor(FakeProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def on_end(self, span):
            self.entered.set()
            self.release.wait()
            super().on_end(span)

    store = FakeStore()
    processor = BlockingProcessor()
    recorder = TraceRecorder(store, processors=(processor,), shutdown_timeout=0.02)
    recorder.start_span("span", "span", SpanKind.CUSTOM)

    with ThreadPoolExecutor(max_workers=1) as executor:
        ending = executor.submit(recorder.end_span, "span")
        assert processor.entered.wait(1)
        assert recorder.close(1) is False
        assert recorder._close_done.is_set()
        assert recorder.close() is False
        assert processor.shutdowns == 0
        assert store.closes == 1
        processor.release.set()
        assert ending.result() is not None


def test_strict_close_deadline_is_stable_when_processor_remains_active() -> None:
    class BlockingProcessor(FakeProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def on_end(self, span):
            self.entered.set()
            self.release.wait()
            super().on_end(span)

    store = FakeStore()
    processor = BlockingProcessor()
    recorder = TraceRecorder(
        store,
        processors=(processor,),
        strict=True,
        shutdown_timeout=0.02,
    )
    recorder.start_span("span", "span", SpanKind.CUSTOM)

    with ThreadPoolExecutor(max_workers=1) as executor:
        ending = executor.submit(recorder.end_span, "span")
        assert processor.entered.wait(1)
        with pytest.raises(RecorderError, match="processor callbacks"):
            recorder.close(1)
        assert recorder._close_done.is_set()
        with pytest.raises(RecorderError, match="processor callbacks"):
            recorder.close()
        assert processor.shutdowns == 0
        assert store.closes == 1
        processor.release.set()
        assert ending.result() is not None


def test_timeouts_are_reported_and_pass_remaining_budget_cooperatively() -> None:
    class TimeoutStore(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.received_timeout = None

        def force_flush(self, timeout=None):
            self.received_timeout = timeout
            return False

    store = TimeoutStore()
    recorder = TraceRecorder(store)
    assert recorder.force_flush(0.5) is False
    assert store.received_timeout is not None
    assert 0 < store.received_timeout <= 0.5

    strict = TraceRecorder(TimeoutStore(), strict=True)
    with pytest.raises(RecorderError, match="force flush"):
        strict.force_flush(0.5)


def test_force_flush_times_out_while_prior_processor_is_active() -> None:
    class BlockingProcessor(FakeProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def on_end(self, span):
            self.entered.set()
            assert self.release.wait(1)
            super().on_end(span)

    processor = BlockingProcessor()
    recorder, _ = make_recorder(processors=(processor,))
    recorder.start_span("span", "span", SpanKind.CUSTOM)
    with ThreadPoolExecutor(max_workers=1) as executor:
        ending = executor.submit(recorder.end_span, "span")
        assert processor.entered.wait(1)
        assert recorder.force_flush(0) is False
        processor.release.set()
        assert ending.result() is not None


def test_timed_out_close_continues_shutdown_in_single_background_owner() -> None:
    class BlockingCloseStore(FakeStore):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.received_timeout = None

        def close(self, timeout=None):
            self.received_timeout = timeout
            self.entered.set()
            assert self.release.wait(1)
            return super().close(timeout)

    store = BlockingCloseStore()
    recorder = TraceRecorder(store, shutdown_timeout=0.5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(recorder.close, 0.01)
        assert store.entered.wait(1)
        assert first.result() is False
        assert not recorder._close_done.is_set()
        assert store.received_timeout is not None
        assert store.received_timeout > 0.1
        store.release.set()
    assert recorder.close() is True
    assert store.closes == 1


def test_force_flush_releases_flush_lock_before_waiting_for_close() -> None:
    class TrackingLock:
        def __init__(self) -> None:
            self.inner = threading.Lock()
            self.acquired = threading.Event()

        def acquire(self, blocking=True, timeout=-1):
            acquired = self.inner.acquire(blocking, timeout)
            if acquired:
                self.acquired.set()
            return acquired

        def release(self):
            self.inner.release()

    class BlockingProcessor(FakeProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def on_end(self, span):
            self.entered.set()
            assert self.release.wait(1)
            super().on_end(span)

    processor = BlockingProcessor()
    recorder, _ = make_recorder(processors=(processor,))
    tracking_lock = TrackingLock()
    recorder._flush_lock = cast(Any, tracking_lock)
    recorder.start_span("span", "span", SpanKind.CUSTOM)

    with ThreadPoolExecutor(max_workers=2) as executor:
        closing = executor.submit(recorder.close, 1)
        assert processor.entered.wait(1)
        flushing = executor.submit(recorder.force_flush, 1)
        assert tracking_lock.acquired.wait(1)
        processor.release.set()
        assert closing.result() is True
        assert flushing.result() is True


def test_processor_reentry_into_flush_and_close_is_rejected_without_deadlock() -> None:
    class ReentrantProcessor(FakeProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.recorder: TraceRecorder
            self.results = []

        def on_end(self, span):
            self.results.append(self.recorder.force_flush())
            self.results.append(self.recorder.close())
            super().on_end(span)

    processor = ReentrantProcessor()
    recorder, _ = make_recorder(processors=(processor,))
    processor.recorder = recorder
    recorder.start_span("span", "span", SpanKind.CUSTOM)

    assert recorder.end_span("span") is not None
    assert processor.results == [False, False]
    assert recorder.close() is True


def test_strict_processor_reentry_fails_promptly() -> None:
    class ReentrantProcessor(FakeProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.recorder: TraceRecorder
            self.errors = []

        def on_end(self, span):
            for callback in (self.recorder.force_flush, self.recorder.close):
                with pytest.raises(RecorderError, match="processor callbacks") as captured:
                    callback()
                self.errors.append(captured.value)
            super().on_end(span)

    processor = ReentrantProcessor()
    store = FakeStore()
    recorder = TraceRecorder(store, processors=(processor,), strict=True)
    processor.recorder = recorder
    recorder.start_span("span", "span", SpanKind.CUSTOM)

    assert recorder.end_span("span") is not None
    assert len(processor.errors) == 2
    assert recorder.close() is True


def test_processor_flush_and_shutdown_callbacks_also_reject_reentry() -> None:
    class ReentrantProcessor(FakeProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.recorder: TraceRecorder
            self.results = []

        def force_flush(self, timeout=None):
            self.results.append(self.recorder.close())
            return super().force_flush(timeout)

        def shutdown(self, timeout=None):
            self.results.append(self.recorder.force_flush())
            return super().shutdown(timeout)

    processor = ReentrantProcessor()
    recorder, _ = make_recorder(processors=(processor,))
    processor.recorder = recorder

    assert recorder.force_flush() is True
    assert recorder.close() is True
    assert processor.results == [False, False, False]


def test_close_clears_all_retained_state() -> None:
    recorder, _ = make_recorder()
    recorder.start_span("root", "root", SpanKind.GRAPH, lane_id="worker")
    recorder.record_metric("root", "tokens", {"total": 1}, cumulative=True)
    recorder.close()

    assert recorder._active == {}
    assert recorder._completed_contexts == {}
    assert recorder._metric_totals == {}
    assert recorder._lanes == {}
