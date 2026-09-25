"""Thread-safe in-memory trace storage."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace

from ..identifiers import SpanId, TraceId
from ..models import (
    MetricPoint,
    Span,
    SpanEvent,
    SpanStatus,
    Trace,
    TraceBundle,
    TraceQuery,
    TraceSummary,
)


@dataclass(slots=True)
class _TraceState:
    trace: Trace
    spans: dict[SpanId, Span] = field(default_factory=dict)
    metrics: list[MetricPoint] = field(default_factory=list)


class MemoryTraceStore:
    """Store complete traces in memory with deterministic trace-level eviction.

    When ``max_traces`` is reached, starting a new trace evicts the oldest
    finalized trace, ordered by start timestamp and then trace ID. Active
    traces are never evicted implicitly. If all retained traces are active,
    the new start raises ``RuntimeError`` and leaves the store unchanged.
    Explicit deletion may remove either an active or finalized trace.
    """

    def __init__(self, *, max_traces: int | None = None) -> None:
        if max_traces is not None:
            if isinstance(max_traces, bool) or not isinstance(max_traces, int):
                raise TypeError("max_traces must be an integer or None")
            if max_traces <= 0:
                raise ValueError("max_traces must be greater than zero")
        self._max_traces = max_traces
        self._traces: dict[TraceId, _TraceState] = {}
        self._lock = threading.RLock()
        self._closed = False

    def start_span(self, trace: Trace, span: Span) -> None:
        with self._lock:
            self._ensure_open()
            if trace.trace_id != span.trace_id:
                raise ValueError("trace and span must share the trace ID")
            if trace.end_time_unix_ns is not None:
                raise ValueError("start_span requires active trace metadata")
            if span.is_finished:
                raise ValueError("start_span requires an active span")
            if span.start_time_unix_ns < trace.start_time_unix_ns:
                raise ValueError("span cannot start before its trace")
            state = self._traces.get(trace.trace_id)
            if state is None:
                if span.parent_span_id is not None:
                    raise ValueError(f"parent span does not exist: {span.parent_span_id}")
                self._ensure_capacity()
                state = _TraceState(trace)
                self._traces[trace.trace_id] = state
            elif _active_trace(state.trace) != trace:
                raise ValueError(f"conflicting metadata for trace {trace.trace_id}")
            if span.span_id in state.spans:
                raise ValueError(f"span already exists: {span.span_id}")
            if span.parent_span_id is not None and span.parent_span_id not in state.spans:
                raise ValueError(f"parent span does not exist: {span.parent_span_id}")
            state.spans[span.span_id] = span
            if state.trace.end_time_unix_ns is not None:
                state.trace = replace(state.trace, end_time_unix_ns=None)

    def record_event(self, event: SpanEvent) -> None:
        with self._lock:
            self._ensure_open()
            state = self._require_trace(event.trace_id)
            span = state.spans.get(event.span_id)
            if span is None:
                raise ValueError(f"span does not exist: {event.span_id}")
            if span.is_finished:
                raise ValueError(f"span is already complete: {event.span_id}")
            state.spans[event.span_id] = replace(span, events=(*span.events, event))

    def record_metric(self, metric: MetricPoint) -> None:
        with self._lock:
            self._ensure_open()
            state = self._require_trace(metric.trace_id)
            if metric.span_id is not None and metric.span_id not in state.spans:
                raise ValueError(f"span does not exist: {metric.span_id}")
            state.metrics.append(metric)

    def complete_span(self, span: Span) -> None:
        with self._lock:
            self._ensure_open()
            if not span.is_finished:
                raise ValueError("complete_span requires a finished span")
            if span.status is SpanStatus.UNSET:
                raise ValueError("complete_span requires a terminal status")
            state = self._require_trace(span.trace_id)
            active = state.spans.get(span.span_id)
            if active is None:
                raise ValueError(f"span does not exist: {span.span_id}")
            if active.is_finished:
                raise ValueError(f"span is already complete: {span.span_id}")
            if not _same_started_span(active, span):
                raise ValueError(f"completed span conflicts with its start: {span.span_id}")
            if active.events != span.events:
                raise ValueError("completed span events differ from recorded events")
            state.spans[span.span_id] = span
            if all(item.is_finished for item in state.spans.values()):
                end_time = max(item.end_time_unix_ns or 0 for item in state.spans.values())
                state.trace = replace(state.trace, end_time_unix_ns=end_time)

    def get_trace(self, trace_id: TraceId) -> TraceBundle | None:
        with self._lock:
            self._ensure_open()
            state = self._traces.get(trace_id)
            if state is None:
                return None
            return TraceBundle(state.trace, tuple(state.spans.values()), tuple(state.metrics))

    def list_traces(self, query: TraceQuery | None = None) -> tuple[TraceSummary, ...]:
        with self._lock:
            self._ensure_open()
            query = query or TraceQuery()
            summaries = [self._summary(state) for state in self._traces.values()]
        matching = [summary for summary in summaries if _matches(summary, query)]
        matching.sort(key=lambda item: (item.start_time_unix_ns, item.trace_id.value), reverse=True)
        return tuple(matching[: query.limit])

    def delete_trace(self, trace_id: TraceId) -> bool:
        with self._lock:
            self._ensure_open()
            return self._traces.pop(trace_id, None) is not None

    def delete_traces(self, *, before_unix_ns: int) -> int:
        with self._lock:
            self._ensure_open()
            if isinstance(before_unix_ns, bool) or not isinstance(before_unix_ns, int):
                raise TypeError("before_unix_ns must be an integer")
            if before_unix_ns < 0:
                raise ValueError("before_unix_ns must be non-negative")
            trace_ids = [
                trace_id
                for trace_id, state in self._traces.items()
                if state.trace.end_time_unix_ns is not None
                and state.trace.end_time_unix_ns < before_unix_ns
            ]
            for trace_id in trace_ids:
                del self._traces[trace_id]
            return len(trace_ids)

    def close(self, timeout: float | None = None) -> bool:
        _validate_timeout(timeout)
        acquired = (
            self._lock.acquire() if timeout is None else self._lock.acquire(timeout=float(timeout))
        )
        if not acquired:
            return False
        try:
            self._closed = True
            return True
        finally:
            self._lock.release()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("trace store is closed")

    def _require_trace(self, trace_id: TraceId) -> _TraceState:
        state = self._traces.get(trace_id)
        if state is None:
            raise ValueError(f"trace does not exist: {trace_id}")
        return state

    def _ensure_capacity(self) -> None:
        if self._max_traces is None or len(self._traces) < self._max_traces:
            return
        candidates = [
            state for state in self._traces.values() if state.trace.end_time_unix_ns is not None
        ]
        if not candidates:
            raise RuntimeError("trace capacity is full with active traces")
        oldest = min(
            candidates, key=lambda item: (item.trace.start_time_unix_ns, item.trace.trace_id)
        )
        del self._traces[oldest.trace.trace_id]

    @staticmethod
    def _summary(state: _TraceState) -> TraceSummary:
        return TraceSummary(
            trace_id=state.trace.trace_id,
            name=state.trace.name,
            service_name=state.trace.service_name,
            start_time_unix_ns=state.trace.start_time_unix_ns,
            end_time_unix_ns=state.trace.end_time_unix_ns,
            span_count=len(state.spans),
            status=_trace_status(state),
        )


def _active_trace(trace: Trace) -> Trace:
    return trace if trace.end_time_unix_ns is None else replace(trace, end_time_unix_ns=None)


def _same_started_span(active: Span, completed: Span) -> bool:
    return (
        replace(
            completed,
            end_time_unix_ns=None,
            end_time_monotonic_ns=None,
            status=SpanStatus.UNSET,
            status_description=None,
            attributes=active.attributes,
            events=active.events,
        )
        == active
    )


def _trace_status(state: _TraceState) -> SpanStatus:
    if state.trace.end_time_unix_ns is None:
        return SpanStatus.UNSET
    statuses = {span.status for span in state.spans.values()}
    for status in (SpanStatus.ERROR, SpanStatus.CANCELLED, SpanStatus.ABANDONED):
        if status in statuses:
            return status
    return SpanStatus.SUCCESS


def _matches(summary: TraceSummary, query: TraceQuery) -> bool:
    if (
        query.start_time_unix_ns is not None
        and summary.start_time_unix_ns < query.start_time_unix_ns
    ):
        return False
    if query.end_time_unix_ns is not None and summary.start_time_unix_ns > query.end_time_unix_ns:
        return False
    if query.service_name is not None and summary.service_name != query.service_name:
        return False
    return query.status is None or summary.status is query.status


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a number or None")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be finite and non-negative")
