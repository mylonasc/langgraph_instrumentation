"""Thread-safe trace span lifecycle coordination."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol, Self, TypeVar, cast

from .clock import Clock, SystemClock
from .identifiers import IdGenerator, RandomIdGenerator, SpanId, TraceId
from .models import (
    Attributes,
    ExecutionLane,
    MetricPoint,
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
    Trace,
    freeze_attributes,
)
from .stores.base import FlushableTraceStore, TraceStore

_LOGGER = logging.getLogger(__name__)
_Result = TypeVar("_Result")


class SpanProcessor(Protocol):
    """Consumes completed immutable spans using cooperative timeout budgets.

    Processor callbacks must not synchronously call ``force_flush`` or ``close``
    on their recorder. The recorder rejects such re-entry rather than waiting
    for the callback that is making the call.
    """

    def on_end(self, span: Span) -> None: ...

    def force_flush(self, timeout: float | None = None) -> bool: ...

    def shutdown(self, timeout: float | None = None) -> bool: ...


class RecorderError(RuntimeError):
    """Raised for lifecycle or instrumentation failures in strict mode."""


class _ClosingError(RuntimeError):
    pass


class _RetryableCloseError(RecorderError):
    pass


@dataclass(slots=True)
class _ActiveSpan:
    correlation_id: str
    trace: Trace
    span: Span
    events: list[SpanEvent]


@dataclass(frozen=True, slots=True)
class _SpanContext:
    trace: Trace
    span_id: SpanId
    execution_lane: ExecutionLane | None

    @property
    def trace_id(self) -> TraceId:
        return self.trace.trace_id


@dataclass(slots=True)
class _Operation:
    callback: Callable[[], object]
    done: bool = False
    cancelled: bool = False
    result: object = None
    error: BaseException | None = None


@dataclass(slots=True)
class _CloseAttempt:
    done: threading.Event
    result: bool = True
    error: BaseException | None = None
    retryable: bool = False


class TraceRecorder:
    """Coordinate span lifecycle independently of callback execution context.

    Store operations are globally ordered by recorder-call admission. A store
    failure never commits the corresponding in-memory transition, so the caller
    may retry it. Strict mode raises ``RecorderError``; non-strict mode returns
    ``None``. Processors run only after completion is accepted by the store and
    never while the recorder's state lock is held.

    ``force_flush`` and ``close`` accept seconds-based caller wait timeouts.
    Shutdown dependencies receive the recorder-wide ``shutdown_timeout`` budget
    configured at construction, so one impatient close caller cannot shorten
    shutdown for every caller. A dependency that ignores its cooperative budget
    can delay background shutdown; hard cancellation is deliberately avoided.
    Completed correlation contexts are retained without payloads in a bounded,
    completion-ordered cache so delayed children can reopen inactive traces.
    If persisting abandoned spans fails, shutdown dependencies remain open and
    only failed spans stay active for the next close attempt. Successfully
    persisted abandoned spans are dispatched once and are not retried.
    """

    def __init__(
        self,
        store: TraceStore | None = None,
        *,
        processors: Iterable[SpanProcessor] = (),
        clock: Clock | None = None,
        id_generator: IdGenerator | None = None,
        strict: bool = False,
        shutdown_timeout: float | None = 30.0,
        completed_context_cache_size: int = 4096,
        service_name: str = "langgraph-instrumentation",
        resource_attributes: Attributes | None = None,
    ) -> None:
        _validate_timeout(shutdown_timeout, "shutdown_timeout")
        _validate_cache_size(completed_context_cache_size)
        self._store = store
        self._processors = tuple(processors)
        self._clock = clock or SystemClock()
        self._id_generator = id_generator or RandomIdGenerator()
        self._strict = strict
        self._shutdown_timeout = shutdown_timeout
        self._completed_context_cache_size = completed_context_cache_size
        _validate_nonempty_string(service_name, "service_name")
        self._service_name = service_name
        self._resource_attributes = freeze_attributes(resource_attributes or {})
        self._condition = threading.Condition(threading.Lock())
        self._flush_lock = threading.Lock()
        self._processor_callback = threading.local()
        self._operations: deque[_Operation] = deque()
        self._dispatching = False
        self._dispatcher_thread_id: int | None = None
        self._active: dict[str, _ActiveSpan] = {}
        self._completed_contexts: OrderedDict[str, _SpanContext] = OrderedDict()
        self._lanes: dict[str, ExecutionLane] = {}
        self._metric_totals: dict[tuple[TraceId, str], dict[str, int | float]] = {}
        self._generation = 0
        self._flushed_generation = -1
        self._processor_sequence = 0
        self._active_processor_calls: set[int] = set()
        self._closing = False
        self._closed = False
        self._close_done = threading.Event()
        self._close_result = True
        self._close_error: BaseException | None = None
        self._close_attempt: _CloseAttempt | None = None

    def start_span(
        self,
        correlation_id: str,
        name: str,
        kind: SpanKind,
        *,
        parent_correlation_id: str | None = None,
        attributes: Attributes | None = None,
        lane_id: str | None = None,
        lane_name: str | None = None,
        trace_name: str | None = None,
        trace_attributes: Attributes | None = None,
    ) -> Span | None:
        """Start a root or child span and return its immutable active snapshot."""
        _validate_correlation_id(correlation_id)
        if parent_correlation_id is not None:
            _validate_correlation_id(parent_correlation_id, "parent_correlation_id")
            if trace_name is not None or trace_attributes is not None:
                raise ValueError("trace metadata may only be supplied for a root span")

        def operation() -> Span | None:
            with self._condition:
                lifecycle_error = None
                if correlation_id in self._active or correlation_id in self._completed_contexts:
                    lifecycle_error = f"duplicate span correlation ID: {correlation_id!r}"
                parent = None
                if lifecycle_error is None and parent_correlation_id is not None:
                    active_parent = self._active.get(parent_correlation_id)
                    parent = (
                        _context_from_span(active_parent.trace, active_parent.span)
                        if active_parent is not None
                        else self._completed_contexts.get(parent_correlation_id)
                    )
                    if parent is None:
                        lifecycle_error = (
                            f"unknown parent correlation ID: {parent_correlation_id!r}"
                        )
            if lifecycle_error is not None:
                return self._lifecycle_failure(lifecycle_error)
            with self._condition:
                reading = self._clock.now()
                trace_id = (
                    parent.trace_id if parent is not None else self._id_generator.new_trace_id()
                )
                trace = (
                    parent.trace
                    if parent is not None
                    else Trace(
                        trace_id=trace_id,
                        name=trace_name or name,
                        service_name=self._service_name,
                        start_time_unix_ns=reading.unix_time_ns,
                        attributes=trace_attributes or {},
                        resource_attributes=self._resource_attributes,
                    )
                )
                lane = self._resolve_lane(lane_id, lane_name, parent)
                span = Span(
                    trace_id=trace_id,
                    span_id=self._id_generator.new_span_id(),
                    parent_span_id=parent.span_id if parent is not None else None,
                    name=name,
                    kind=kind,
                    start_time_unix_ns=reading.unix_time_ns,
                    start_time_monotonic_ns=reading.monotonic_time_ns,
                    attributes=attributes or {},
                    execution_lane=lane,
                )
            succeeded = self._store_start_span(trace, span) if self._store else True
            if not succeeded:
                if self._strict:
                    raise RecorderError("failed to start span")
                return None
            with self._condition:
                self._lanes.setdefault(lane.lane_id, lane)
                self._active[correlation_id] = _ActiveSpan(correlation_id, trace, span, [])
                self._generation += 1
            return span

        return self._ordered_lifecycle(operation, "start a span")

    def add_event(
        self,
        correlation_id: str,
        name: str,
        *,
        attributes: Attributes | None = None,
    ) -> SpanEvent | None:
        """Append an event to an active span."""
        _validate_correlation_id(correlation_id)

        def operation() -> SpanEvent | None:
            with self._condition:
                active = self._active.get(correlation_id)
            if active is None:
                return self._lifecycle_failure(
                    f"cannot add event to unknown or completed span: {correlation_id!r}"
                )
            with self._condition:
                reading = self._clock.now()
                event = SpanEvent(
                    trace_id=active.span.trace_id,
                    span_id=active.span.span_id,
                    name=name,
                    time_unix_ns=reading.unix_time_ns,
                    time_monotonic_ns=reading.monotonic_time_ns,
                    attributes=attributes or {},
                )
            succeeded = (
                self._store_call("record event", self._store.record_event, event)
                if self._store
                else True
            )
            if not succeeded:
                if self._strict:
                    raise RecorderError("failed to record event")
                return None
            with self._condition:
                active.events.append(event)
                self._generation += 1
            return event

        return self._ordered_lifecycle(operation, "add an event")

    def record_metric(
        self,
        correlation_id: str,
        name: str,
        values: Mapping[str, int | float],
        *,
        attributes: Attributes | None = None,
        cumulative: bool = False,
    ) -> MetricPoint | None:
        """Record metric values, optionally as synchronized trace-wide totals."""
        _validate_correlation_id(correlation_id)

        def operation() -> MetricPoint | None:
            with self._condition:
                active = self._active.get(correlation_id)
            if active is None:
                return self._lifecycle_failure(
                    f"cannot record metric for unknown or completed span: {correlation_id!r}"
                )
            with self._condition:
                reading = self._clock.now()
                metric = MetricPoint(
                    trace_id=active.span.trace_id,
                    span_id=active.span.span_id,
                    name=name,
                    time_unix_ns=reading.unix_time_ns,
                    time_monotonic_ns=reading.monotonic_time_ns,
                    values=values,
                    attributes=attributes or {},
                )
                totals_key: tuple[TraceId, str] | None = None
                updated: dict[str, int | float] | None = None
                if cumulative:
                    totals_key = (active.span.trace_id, name)
                    updated = self._metric_totals.get(totals_key, {}).copy()
                    for value_name, value in metric.values.items():
                        total = updated.get(value_name, 0) + value
                        if isinstance(total, float) and not math.isfinite(total):
                            raise ValueError(
                                f"cumulative metric value {value_name!r} must be finite"
                            )
                        updated[value_name] = total
                    metric = replace(metric, values=updated)
            succeeded = (
                self._store_call("record metric", self._store.record_metric, metric)
                if self._store
                else True
            )
            if not succeeded:
                if self._strict:
                    raise RecorderError("failed to record metric")
                return None
            with self._condition:
                if totals_key is not None and updated is not None:
                    self._metric_totals[totals_key] = updated
                self._generation += 1
            return metric

        return self._ordered_lifecycle(operation, "record a metric")

    def end_span(
        self,
        correlation_id: str,
        *,
        status: SpanStatus = SpanStatus.SUCCESS,
        status_description: str | None = None,
        attributes: Attributes | None = None,
    ) -> Span | None:
        """Finish a span with a terminal status and dispatch its snapshot."""
        _validate_correlation_id(correlation_id)
        if not isinstance(status, SpanStatus):
            raise TypeError("status must be a SpanStatus")
        if status is SpanStatus.UNSET:
            raise ValueError("end_span status must be terminal")

        def operation() -> tuple[Span, int] | None:
            with self._condition:
                active = self._active.get(correlation_id)
            if active is None:
                return self._lifecycle_failure(
                    f"cannot end unknown or completed span: {correlation_id!r}"
                )
            with self._condition:
                completed = self._complete(active, status, status_description, attributes)
            succeeded = (
                self._store_call("complete span", self._store.complete_span, completed)
                if self._store
                else True
            )
            if not succeeded:
                if self._strict:
                    raise RecorderError("failed to complete span")
                return None
            with self._condition:
                del self._active[correlation_id]
                self._retain_completed_context(correlation_id, active.trace, completed)
                self._generation += 1
                token = self._begin_processor_call()
                self._release_trace_if_inactive(completed.trace_id)
            return completed, token

        result = self._ordered_lifecycle(operation, "end a span")
        if result is None:
            return None
        completed, token = result
        try:
            self._dispatch_processors(completed)
        finally:
            self._finish_processor_call(token)
        return completed

    def force_flush(self, timeout: float | None = None) -> bool:
        """Establish a quiescence barrier and cooperatively flush destinations."""
        if self._reject_processor_reentry("force flush"):
            return False
        deadline = _deadline(timeout)
        remaining = _remaining(deadline)
        acquired = (
            self._flush_lock.acquire()
            if remaining is None
            else self._flush_lock.acquire(timeout=remaining)
        )
        if not acquired:
            return self._timeout_failure("force flush")
        try:
            try:
                return self._force_flush(deadline)
            except _ClosingError:
                pass
        finally:
            self._flush_lock.release()
        with self._condition:
            attempt = self._close_attempt
            if attempt is None:
                if self._closed:
                    return self._closed_result()
                return self._timeout_failure("force flush")
        return self._wait_for_close(attempt, deadline, "force flush")

    def _force_flush(self, deadline: float | None) -> bool:
        """Flush while the caller owns the flush/shutdown serialization lock."""

        def store_barrier() -> tuple[bool, int, int, bool]:
            with self._condition:
                generation = self._generation
                processor_barrier = self._processor_sequence
                if self._flushed_generation == generation:
                    return True, processor_barrier, generation, False
            succeeded = self._flush_store(deadline)
            return succeeded, processor_barrier, generation, True

        try:
            store_ok, processor_barrier, generation, needs_flush = self._ordered(
                store_barrier, deadline
            )
        except TimeoutError:
            return self._timeout_failure("force flush")
        if not needs_flush:
            return True
        if not self._wait_for_processors(processor_barrier, deadline):
            return self._timeout_failure("force flush")
        processors_ok = self._flush_processors(deadline)
        succeeded = store_ok and processors_ok
        if succeeded:
            with self._condition:
                self._flushed_generation = max(self._flushed_generation, generation)
        if not succeeded and self._strict:
            raise RecorderError("failed to force flush")
        return succeeded

    def close(self, timeout: float | None = None) -> bool:
        """Abandon active spans child-first, then flush and shut down once."""
        if self._reject_processor_reentry("close"):
            return False
        deadline = _deadline(timeout)
        with self._condition:
            if self._closed:
                return self._closed_result()
            attempt = self._close_attempt
            if attempt is None:
                attempt = _CloseAttempt(threading.Event())
                self._close_attempt = attempt
                self._close_done = attempt.done
                self._close_result = True
                self._close_error = None
                self._closing = True
                threading.Thread(
                    target=self._close_worker,
                    args=(attempt,),
                    name="trace-recorder-close",
                    daemon=True,
                ).start()
        return self._wait_for_close(attempt, deadline, "close")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
            _LOGGER.exception("recorder close failed while preserving application exception")
        return False

    def _perform_close(self, deadline: float | None) -> tuple[bool, bool]:
        def abandon() -> tuple[list[Span], int, bool]:
            with self._condition:
                active = sorted(
                    self._active.values(),
                    key=lambda item: self._active_depth(item.span),
                    reverse=True,
                )
            completed: list[Span] = []
            completed_correlations: list[str] = []
            succeeded = True
            for item in active:
                span = self._complete(item, SpanStatus.ABANDONED, "recorder closed", None)
                store_ok = (
                    self._store_call("complete span", self._store.complete_span, span)
                    if self._store
                    else True
                )
                succeeded = store_ok and succeeded
                if store_ok:
                    completed.append(span)
                    completed_correlations.append(item.correlation_id)
            with self._condition:
                for correlation_id in completed_correlations:
                    del self._active[correlation_id]
                self._generation += len(completed)
                barrier = self._processor_sequence
            return completed, barrier, succeeded

        try:
            completed, barrier, stores_ok = self._ordered(abandon, allow_during_closing=True)
        except TimeoutError:
            return self._timeout_failure("close"), False
        if not self._wait_for_processors(barrier, deadline):
            return self._finish_close_after_processor_timeout(deadline), False
        if not self._acquire_flush_lock(deadline):
            return self._close_failure(), False
        try:
            processors_ok = True
            for span in completed:
                try:
                    processors_ok = self._dispatch_processors(span) and processors_ok
                except RecorderError:
                    processors_ok = False
            if not stores_ok:
                if self._strict:
                    raise _RetryableCloseError("failed to persist abandoned spans")
                return False, True
            flush_ok = self._flush_store(deadline) and self._flush_processors(deadline)
            shutdown_ok = self._shutdown_dependencies(deadline)
        finally:
            self._flush_lock.release()
        succeeded = stores_ok and processors_ok and flush_ok and shutdown_ok
        if not succeeded and self._strict:
            raise RecorderError("failed to close recorder")
        return succeeded, False

    def _close_worker(self, attempt: _CloseAttempt) -> None:
        try:
            attempt.result, attempt.retryable = self._perform_close(
                _deadline(self._shutdown_timeout)
            )
        except _RetryableCloseError as error:
            attempt.error = error
            attempt.retryable = True
        except BaseException as error:
            attempt.error = error
        finally:
            with self._condition:
                self._close_result = attempt.result
                self._close_error = attempt.error
                if not attempt.retryable:
                    self._active.clear()
                    self._completed_contexts.clear()
                    self._metric_totals.clear()
                    self._lanes.clear()
                    self._operations.clear()
                    self._closed = True
                self._closing = False
                self._close_attempt = None
                attempt.done.set()
                self._condition.notify_all()

    def _ordered(
        self,
        callback: Callable[[], _Result],
        deadline: float | None = None,
        *,
        allow_during_closing: bool = False,
    ) -> _Result:
        operation = _Operation(cast(Callable[[], object], callback))
        with self._condition:
            if (self._closing or self._closed) and not allow_during_closing:
                raise _ClosingError
            if self._dispatching and self._dispatcher_thread_id == threading.get_ident():
                raise RecorderError("a store callback must not re-enter its recorder")
            self._operations.append(operation)
            while not operation.done:
                if not self._dispatching and self._operations:
                    current = self._operations.popleft()
                    self._dispatching = True
                    self._dispatcher_thread_id = threading.get_ident()
                else:
                    remaining = _remaining(deadline)
                    if remaining is not None and remaining <= 0:
                        if operation in self._operations:
                            self._operations.remove(operation)
                            operation.cancelled = True
                        raise TimeoutError
                    self._condition.wait(remaining)
                    continue
                self._condition.release()
                try:
                    if not current.cancelled:
                        current.result = current.callback()
                except BaseException as error:
                    current.error = error
                finally:
                    self._condition.acquire()
                    current.done = True
                    self._dispatching = False
                    self._dispatcher_thread_id = None
                    self._condition.notify_all()
            if operation.error is not None:
                raise operation.error
            return cast(_Result, operation.result)

    def _ordered_lifecycle(
        self,
        callback: Callable[[], _Result],
        operation: str,
    ) -> _Result | None:
        try:
            return self._ordered(callback)
        except _ClosingError:
            return self._lifecycle_failure(f"cannot {operation} after recorder close has begun")

    def _resolve_lane(
        self,
        lane_id: str | None,
        lane_name: str | None,
        parent: _SpanContext | None,
    ) -> ExecutionLane:
        if lane_id is None and parent is not None and parent.execution_lane is not None:
            return parent.execution_lane
        resolved_id = lane_id or "main"
        lane = self._lanes.get(resolved_id)
        if lane is None:
            lane = ExecutionLane(resolved_id, lane_name, len(self._lanes))
        elif lane_name is not None and lane.name != lane_name:
            raise ValueError(f"execution lane {resolved_id!r} already has a different name")
        return lane

    def _complete(
        self,
        active: _ActiveSpan,
        status: SpanStatus,
        description: str | None,
        attributes: Attributes | None,
    ) -> Span:
        reading = self._clock.now()
        merged_attributes = dict(active.span.attributes)
        if attributes is not None:
            merged_attributes.update(attributes)
        return replace(
            active.span,
            end_time_unix_ns=max(reading.unix_time_ns, active.span.start_time_unix_ns),
            end_time_monotonic_ns=max(
                reading.monotonic_time_ns, active.span.start_time_monotonic_ns
            ),
            status=status,
            status_description=description,
            attributes=merged_attributes,
            events=tuple(active.events),
        )

    def _active_depth(self, span: Span) -> int:
        parents = {item.span.span_id: item.span.parent_span_id for item in self._active.values()}
        depth = 0
        parent_id = span.parent_span_id
        while parent_id in parents:
            depth += 1
            parent_id = parents[parent_id]
        return depth

    def _release_trace_if_inactive(self, trace_id: TraceId) -> None:
        if any(item.span.trace_id == trace_id for item in self._active.values()):
            return
        self._metric_totals = {
            key: totals for key, totals in self._metric_totals.items() if key[0] != trace_id
        }

    def _retain_completed_context(self, correlation_id: str, trace: Trace, span: Span) -> None:
        self._completed_contexts[correlation_id] = _context_from_span(trace, span)
        while len(self._completed_contexts) > self._completed_context_cache_size:
            self._completed_contexts.popitem(last=False)

    def _begin_processor_call(self) -> int:
        self._processor_sequence += 1
        token = self._processor_sequence
        self._active_processor_calls.add(token)
        return token

    def _finish_processor_call(self, token: int) -> None:
        with self._condition:
            self._active_processor_calls.discard(token)
            self._condition.notify_all()

    def _wait_for_processors(self, barrier: int, deadline: float | None) -> bool:
        with self._condition:
            while any(token <= barrier for token in self._active_processor_calls):
                remaining = _remaining(deadline)
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _dispatch_processors(self, span: Span) -> bool:
        succeeded = True
        errors: list[Exception] = []
        for processor in self._processors:
            try:
                self._invoke_processor(processor.on_end, span)
            except Exception as error:
                succeeded = False
                errors.append(error)
                _LOGGER.exception("trace recorder failed to process completed span")
        if errors and self._strict:
            raise RecorderError("failed to process completed span") from errors[0]
        return succeeded

    def _store_call[Value](
        self,
        operation: str,
        callback: Callable[[Value], None],
        value: Value,
    ) -> bool:
        try:
            callback(value)
            return True
        except Exception:
            _LOGGER.exception("trace recorder failed to %s", operation)
            return False

    def _store_start_span(self, trace: Trace, span: Span) -> bool:
        try:
            if self._store is not None:
                self._store.start_span(trace, span)
            return True
        except Exception:
            _LOGGER.exception("trace recorder failed to start span")
            return False

    def _flush_store(self, deadline: float | None) -> bool:
        if self._store is None:
            return True
        if not isinstance(self._store, FlushableTraceStore):
            return True
        return self._timed_call("flush store", self._store.force_flush, deadline)

    def _flush_processors(self, deadline: float | None) -> bool:
        succeeded = True
        for processor in self._processors:
            succeeded = (
                self._timed_processor_call("flush processor", processor.force_flush, deadline)
                and succeeded
            )
        return succeeded

    def _shutdown_dependencies(self, deadline: float | None) -> bool:
        succeeded = True
        for processor in self._processors:
            succeeded = (
                self._timed_processor_call("shut down processor", processor.shutdown, deadline)
                and succeeded
            )
        succeeded = self._close_store(deadline) and succeeded
        return succeeded

    def _finish_close_after_processor_timeout(
        self,
        deadline: float | None,
    ) -> bool:
        if self._acquire_flush_lock(deadline):
            try:
                self._close_store(deadline)
            finally:
                self._flush_lock.release()
        if self._strict:
            raise RecorderError("timed out waiting for processor callbacks during close")
        return False

    def _acquire_flush_lock(self, deadline: float | None) -> bool:
        remaining = _remaining(deadline)
        if remaining is None:
            self._flush_lock.acquire()
            return True
        return self._flush_lock.acquire(timeout=remaining)

    def _close_store(self, deadline: float | None) -> bool:
        if self._store is None:
            return True
        return self._timed_call("close store", self._store.close, deadline)

    def _close_failure(self) -> bool:
        if self._strict:
            raise RecorderError("failed to close recorder")
        return False

    def _timed_call(
        self,
        operation: str,
        callback: Callable[[float | None], bool],
        deadline: float | None,
    ) -> bool:
        remaining = _remaining(deadline)
        try:
            return callback(remaining)
        except Exception:
            _LOGGER.exception("trace recorder failed to %s", operation)
            return False

    def _timed_processor_call(
        self,
        operation: str,
        callback: Callable[[float | None], bool],
        deadline: float | None,
    ) -> bool:
        try:
            return self._invoke_processor(callback, _remaining(deadline))
        except Exception:
            _LOGGER.exception("trace recorder failed to %s", operation)
            return False

    def _invoke_processor[Value, Result](
        self,
        callback: Callable[[Value], Result],
        value: Value,
    ) -> Result:
        previous = getattr(self._processor_callback, "active", False)
        self._processor_callback.active = True
        try:
            return callback(value)
        finally:
            self._processor_callback.active = previous

    def _reject_processor_reentry(self, operation: str) -> bool:
        if not getattr(self._processor_callback, "active", False):
            return False
        message = f"processor callbacks must not call recorder {operation}"
        if self._strict:
            raise RecorderError(message)
        _LOGGER.warning(message)
        return True

    def _wait_for_close(
        self,
        attempt: _CloseAttempt,
        deadline: float | None,
        operation: str,
    ) -> bool:
        if not attempt.done.wait(_remaining(deadline)):
            return self._timeout_failure(operation)
        if attempt.error is not None:
            raise attempt.error
        return attempt.result

    def _closed_result(self) -> bool:
        if self._close_error is not None:
            raise self._close_error
        return self._close_result

    def _timeout_failure(self, operation: str) -> bool:
        if self._strict:
            raise RecorderError(f"timed out during {operation}")
        return False

    def _lifecycle_failure(self, message: str) -> None:
        if self._strict:
            raise RecorderError(message)
        _LOGGER.warning(message)
        return None


def _validate_correlation_id(value: str, field_name: str = "correlation_id") -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _context_from_span(trace: Trace, span: Span) -> _SpanContext:
    return _SpanContext(trace, span.span_id, span.execution_lane)


def _validate_nonempty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _validate_timeout(timeout: float | None, field_name: str) -> None:
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError(f"{field_name} must be a number or None")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _validate_cache_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("completed_context_cache_size must be an integer")
    if value <= 0:
        raise ValueError("completed_context_cache_size must be greater than zero")


def _deadline(timeout: float | None) -> float | None:
    _validate_timeout(timeout, "timeout")
    if timeout is None:
        return None
    return time.monotonic() + timeout


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
