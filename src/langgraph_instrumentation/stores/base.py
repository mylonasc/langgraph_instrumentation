"""Synchronous persistence contracts for neutral trace records.

Each mutating call is atomic: it either commits the complete operation or
raises without changing visible state. Implementations raise ``ValueError``
for missing or conflicting trace lifecycle records. Once closed, every data
operation raises ``RuntimeError`` before validating its other arguments.
Query results are immutable snapshots and are ordered as documented by the
individual methods.

Stores may commit synchronously, in which case they need not implement
``FlushableTraceStore``. Durable or buffered implementations can expose that
optional protocol so the recorder can include them in a flush barrier.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..identifiers import TraceId
from ..models import MetricPoint, Span, SpanEvent, Trace, TraceBundle, TraceQuery, TraceSummary


class TraceStore(Protocol):
    """Canonical synchronous trace persistence extension point.

    ``start_span`` creates a trace when its ID is new and otherwise verifies
    the supplied metadata before appending to it. Existing finalized traces
    must be reopenable for delayed children. ``complete_span`` atomically
    replaces the active span and finalizes the trace when no spans remain
    active. Events and metrics require an existing trace; span-scoped records
    also require an existing span.

    ``list_traces`` returns newest starts first, with the trace ID as a stable
    tie breaker. Deletion and retention always remove complete traces,
    including all spans, events, and metrics. Retention only removes finalized
    traces whose end timestamp is strictly before its cutoff.

    ``close`` is idempotent. Its timeout bounds resource-lock acquisition; a
    timeout returns ``False`` and must not mark the store closed. Store methods
    do not apply recorder strictness or wrap errors in recorder exceptions.
    """

    def start_span(self, trace: Trace, span: Span) -> None: ...

    def record_event(self, event: SpanEvent) -> None: ...

    def record_metric(self, metric: MetricPoint) -> None: ...

    def complete_span(self, span: Span) -> None: ...

    def get_trace(self, trace_id: TraceId) -> TraceBundle | None: ...

    def list_traces(self, query: TraceQuery | None = None) -> tuple[TraceSummary, ...]: ...

    def delete_trace(self, trace_id: TraceId) -> bool: ...

    def delete_traces(self, *, before_unix_ns: int) -> int: ...

    def close(self, timeout: float | None = None) -> bool: ...


@runtime_checkable
class FlushableTraceStore(Protocol):
    """Optional capability for stores with buffered persistence work."""

    def force_flush(self, timeout: float | None = None) -> bool: ...
