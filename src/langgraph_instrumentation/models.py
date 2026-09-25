"""Exporter-independent trace domain records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import cast

from .identifiers import SpanId, TraceId

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]
type FrozenJSONValue = JSONScalar | tuple[FrozenJSONValue, ...] | Mapping[str, FrozenJSONValue]
type Attributes = Mapping[str, object]

_EMPTY_ATTRIBUTES: Attributes = MappingProxyType({})


class SpanKind(StrEnum):
    """The operation represented by a span."""

    GRAPH = "graph"
    NODE = "node"
    CHAIN = "chain"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    CUSTOM = "custom"


class SpanStatus(StrEnum):
    """The terminal state of a span."""

    UNSET = "unset"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


@dataclass(frozen=True, slots=True)
class ExecutionLane:
    """A neutral logical lane used to visualize concurrent execution."""

    lane_id: str
    name: str | None = None
    sort_index: int | None = None

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.lane_id, "lane_id")
        if self.name is not None:
            _validate_nonempty_string(self.name, "lane name")
        if self.sort_index is not None:
            _validate_integer(self.sort_index, "sort_index")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "lane_id": self.lane_id,
            "name": self.name,
            "sort_index": self.sort_index,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ExecutionLane:
        return cls(
            lane_id=_required_string(data, "lane_id"),
            name=_optional_string(data, "name"),
            sort_index=_optional_integer(data, "sort_index"),
        )


@dataclass(frozen=True, slots=True)
class Trace:
    """Trace-level metadata shared by related spans."""

    trace_id: TraceId
    name: str
    service_name: str
    start_time_unix_ns: int
    end_time_unix_ns: int | None = None
    attributes: Attributes = field(default_factory=lambda: _EMPTY_ATTRIBUTES)
    resource_attributes: Attributes = field(default_factory=lambda: _EMPTY_ATTRIBUTES)

    def __post_init__(self) -> None:
        _validate_instance(self.trace_id, TraceId, "trace_id")
        _validate_nonempty_string(self.name, "trace name")
        _validate_nonempty_string(self.service_name, "service_name")
        _validate_timestamp(self.start_time_unix_ns, "start_time_unix_ns")
        _validate_end_timestamp(
            self.start_time_unix_ns,
            self.end_time_unix_ns,
            "end_time_unix_ns",
        )
        object.__setattr__(self, "attributes", freeze_attributes(self.attributes))
        object.__setattr__(
            self,
            "resource_attributes",
            freeze_attributes(self.resource_attributes),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "trace_id": str(self.trace_id),
            "name": self.name,
            "service_name": self.service_name,
            "start_time_unix_ns": self.start_time_unix_ns,
            "end_time_unix_ns": self.end_time_unix_ns,
            "attributes": attributes_to_dict(self.attributes),
            "resource_attributes": attributes_to_dict(self.resource_attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Trace:
        return cls(
            trace_id=TraceId.from_hex(_required_string(data, "trace_id")),
            name=_required_string(data, "name"),
            service_name=_required_string(data, "service_name"),
            start_time_unix_ns=_required_integer(data, "start_time_unix_ns"),
            end_time_unix_ns=_optional_integer(data, "end_time_unix_ns"),
            attributes=_attributes_from_data(data, "attributes"),
            resource_attributes=_attributes_from_data(data, "resource_attributes"),
        )


@dataclass(frozen=True, slots=True)
class SpanEvent:
    """A timestamped event associated with a span."""

    trace_id: TraceId
    span_id: SpanId
    name: str
    time_unix_ns: int
    time_monotonic_ns: int
    attributes: Attributes = field(default_factory=lambda: _EMPTY_ATTRIBUTES)

    def __post_init__(self) -> None:
        _validate_instance(self.trace_id, TraceId, "trace_id")
        _validate_instance(self.span_id, SpanId, "span_id")
        _validate_nonempty_string(self.name, "event name")
        _validate_timestamp(self.time_unix_ns, "time_unix_ns")
        _validate_timestamp(self.time_monotonic_ns, "time_monotonic_ns")
        object.__setattr__(self, "attributes", freeze_attributes(self.attributes))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "trace_id": str(self.trace_id),
            "span_id": str(self.span_id),
            "name": self.name,
            "time_unix_ns": self.time_unix_ns,
            "time_monotonic_ns": self.time_monotonic_ns,
            "attributes": attributes_to_dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SpanEvent:
        return cls(
            trace_id=TraceId.from_hex(_required_string(data, "trace_id")),
            span_id=SpanId.from_hex(_required_string(data, "span_id")),
            name=_required_string(data, "name"),
            time_unix_ns=_required_integer(data, "time_unix_ns"),
            time_monotonic_ns=_required_integer(data, "time_monotonic_ns"),
            attributes=_attributes_from_data(data, "attributes"),
        )


@dataclass(frozen=True, slots=True)
class Span:
    """An operation within a trace, either active or completed."""

    trace_id: TraceId
    span_id: SpanId
    name: str
    kind: SpanKind
    start_time_unix_ns: int
    start_time_monotonic_ns: int
    parent_span_id: SpanId | None = None
    end_time_unix_ns: int | None = None
    end_time_monotonic_ns: int | None = None
    status: SpanStatus = SpanStatus.UNSET
    status_description: str | None = None
    attributes: Attributes = field(default_factory=lambda: _EMPTY_ATTRIBUTES)
    events: tuple[SpanEvent, ...] = ()
    execution_lane: ExecutionLane | None = None

    def __post_init__(self) -> None:
        _validate_instance(self.trace_id, TraceId, "trace_id")
        _validate_instance(self.span_id, SpanId, "span_id")
        if self.parent_span_id is not None:
            _validate_instance(self.parent_span_id, SpanId, "parent_span_id")
        _validate_instance(self.kind, SpanKind, "kind")
        _validate_instance(self.status, SpanStatus, "status")
        if self.execution_lane is not None:
            _validate_instance(self.execution_lane, ExecutionLane, "execution_lane")
        _validate_nonempty_string(self.name, "span name")
        _validate_timestamp(self.start_time_unix_ns, "start_time_unix_ns")
        _validate_timestamp(self.start_time_monotonic_ns, "start_time_monotonic_ns")
        _validate_end_timestamp(
            self.start_time_unix_ns,
            self.end_time_unix_ns,
            "end_time_unix_ns",
        )
        _validate_end_timestamp(
            self.start_time_monotonic_ns,
            self.end_time_monotonic_ns,
            "end_time_monotonic_ns",
        )
        if (self.end_time_unix_ns is None) != (self.end_time_monotonic_ns is None):
            raise ValueError("span end timestamps must either both be set or both be absent")
        if self.parent_span_id == self.span_id:
            raise ValueError("a span cannot be its own parent")
        if self.status_description is not None:
            _validate_nonempty_string(self.status_description, "status_description")
        if self.status is not SpanStatus.UNSET and self.end_time_unix_ns is None:
            raise ValueError("an active span must have unset status")

        events = tuple(self.events)
        for event in events:
            _validate_instance(event, SpanEvent, "event")
            if event.trace_id != self.trace_id or event.span_id != self.span_id:
                raise ValueError("span event IDs must match their containing span")
            if event.time_monotonic_ns < self.start_time_monotonic_ns:
                raise ValueError("span event cannot precede its containing span")
            if (
                self.end_time_monotonic_ns is not None
                and event.time_monotonic_ns > self.end_time_monotonic_ns
            ):
                raise ValueError("span event cannot follow its containing span")
        frozen_events = tuple(sorted(events, key=_event_sort_key))
        object.__setattr__(self, "attributes", freeze_attributes(self.attributes))
        object.__setattr__(self, "events", frozen_events)

    @property
    def is_finished(self) -> bool:
        return self.end_time_monotonic_ns is not None

    @property
    def duration_ns(self) -> int | None:
        if self.end_time_monotonic_ns is None:
            return None
        return self.end_time_monotonic_ns - self.start_time_monotonic_ns

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "trace_id": str(self.trace_id),
            "span_id": str(self.span_id),
            "parent_span_id": str(self.parent_span_id) if self.parent_span_id else None,
            "name": self.name,
            "kind": self.kind.value,
            "start_time_unix_ns": self.start_time_unix_ns,
            "start_time_monotonic_ns": self.start_time_monotonic_ns,
            "end_time_unix_ns": self.end_time_unix_ns,
            "end_time_monotonic_ns": self.end_time_monotonic_ns,
            "status": self.status.value,
            "status_description": self.status_description,
            "attributes": attributes_to_dict(self.attributes),
            "events": [event.to_dict() for event in self.events],
            "execution_lane": self.execution_lane.to_dict() if self.execution_lane else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Span:
        parent_span_id = _optional_string(data, "parent_span_id")
        lane_data = data.get("execution_lane")
        events_data = _required_list(data, "events", default=[])
        return cls(
            trace_id=TraceId.from_hex(_required_string(data, "trace_id")),
            span_id=SpanId.from_hex(_required_string(data, "span_id")),
            parent_span_id=(
                SpanId.from_hex(parent_span_id) if parent_span_id is not None else None
            ),
            name=_required_string(data, "name"),
            kind=_enum_from_data(SpanKind, data, "kind"),
            start_time_unix_ns=_required_integer(data, "start_time_unix_ns"),
            start_time_monotonic_ns=_required_integer(data, "start_time_monotonic_ns"),
            end_time_unix_ns=_optional_integer(data, "end_time_unix_ns"),
            end_time_monotonic_ns=_optional_integer(data, "end_time_monotonic_ns"),
            status=_enum_from_data(SpanStatus, data, "status"),
            status_description=_optional_string(data, "status_description"),
            attributes=_attributes_from_data(data, "attributes"),
            events=tuple(
                SpanEvent.from_dict(_expect_mapping(item, "event")) for item in events_data
            ),
            execution_lane=(
                ExecutionLane.from_dict(_expect_mapping(lane_data, "execution_lane"))
                if lane_data is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """A timestamped set of numeric metric values."""

    trace_id: TraceId
    name: str
    time_unix_ns: int
    time_monotonic_ns: int
    values: Mapping[str, int | float]
    span_id: SpanId | None = None
    attributes: Attributes = field(default_factory=lambda: _EMPTY_ATTRIBUTES)

    def __post_init__(self) -> None:
        _validate_instance(self.trace_id, TraceId, "trace_id")
        if self.span_id is not None:
            _validate_instance(self.span_id, SpanId, "span_id")
        _validate_nonempty_string(self.name, "metric name")
        _validate_timestamp(self.time_unix_ns, "time_unix_ns")
        _validate_timestamp(self.time_monotonic_ns, "time_monotonic_ns")
        object.__setattr__(self, "values", _freeze_metric_values(self.values))
        object.__setattr__(self, "attributes", freeze_attributes(self.attributes))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "trace_id": str(self.trace_id),
            "span_id": str(self.span_id) if self.span_id else None,
            "name": self.name,
            "time_unix_ns": self.time_unix_ns,
            "time_monotonic_ns": self.time_monotonic_ns,
            "values": cast(JSONValue, dict(self.values)),
            "attributes": attributes_to_dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MetricPoint:
        span_id = _optional_string(data, "span_id")
        values = _expect_mapping(data.get("values"), "values")
        return cls(
            trace_id=TraceId.from_hex(_required_string(data, "trace_id")),
            span_id=SpanId.from_hex(span_id) if span_id is not None else None,
            name=_required_string(data, "name"),
            time_unix_ns=_required_integer(data, "time_unix_ns"),
            time_monotonic_ns=_required_integer(data, "time_monotonic_ns"),
            values=cast(Mapping[str, int | float], values),
            attributes=_attributes_from_data(data, "attributes"),
        )


@dataclass(frozen=True, slots=True)
class TraceBundle:
    """A deterministic snapshot of a trace and its collected records."""

    trace: Trace
    spans: tuple[Span, ...] = ()
    metrics: tuple[MetricPoint, ...] = ()

    def __post_init__(self) -> None:
        _validate_instance(self.trace, Trace, "trace")
        spans = tuple(self.spans)
        metrics = tuple(self.metrics)
        span_ids: set[SpanId] = set()
        for span in spans:
            _validate_instance(span, Span, "span")
            if span.trace_id != self.trace.trace_id:
                raise ValueError("all spans in a bundle must share the trace ID")
            if span.span_id in span_ids:
                raise ValueError(f"duplicate span ID in bundle: {span.span_id}")
            span_ids.add(span.span_id)
        for metric in metrics:
            _validate_instance(metric, MetricPoint, "metric")
            if metric.trace_id != self.trace.trace_id:
                raise ValueError("all metrics in a bundle must share the trace ID")
        object.__setattr__(self, "spans", tuple(sorted(spans, key=_span_sort_key)))
        object.__setattr__(self, "metrics", tuple(sorted(metrics, key=_metric_sort_key)))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "trace": self.trace.to_dict(),
            "spans": [span.to_dict() for span in self.spans],
            "metrics": [metric.to_dict() for metric in self.metrics],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TraceBundle:
        trace = Trace.from_dict(_expect_mapping(data.get("trace"), "trace"))
        spans = _required_list(data, "spans", default=[])
        metrics = _required_list(data, "metrics", default=[])
        return cls(
            trace=trace,
            spans=tuple(Span.from_dict(_expect_mapping(item, "span")) for item in spans),
            metrics=tuple(
                MetricPoint.from_dict(_expect_mapping(item, "metric")) for item in metrics
            ),
        )


@dataclass(frozen=True, slots=True)
class TraceSummary:
    """Compact trace information returned by listing operations."""

    trace_id: TraceId
    name: str
    service_name: str
    start_time_unix_ns: int
    end_time_unix_ns: int | None
    span_count: int
    status: SpanStatus = SpanStatus.UNSET

    def __post_init__(self) -> None:
        _validate_instance(self.trace_id, TraceId, "trace_id")
        _validate_instance(self.status, SpanStatus, "status")
        _validate_nonempty_string(self.name, "trace name")
        _validate_nonempty_string(self.service_name, "service_name")
        _validate_timestamp(self.start_time_unix_ns, "start_time_unix_ns")
        _validate_end_timestamp(
            self.start_time_unix_ns,
            self.end_time_unix_ns,
            "end_time_unix_ns",
        )
        if isinstance(self.span_count, bool) or not isinstance(self.span_count, int):
            raise TypeError("span_count must be an integer")
        if self.span_count < 0:
            raise ValueError("span_count must be non-negative")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "trace_id": str(self.trace_id),
            "name": self.name,
            "service_name": self.service_name,
            "start_time_unix_ns": self.start_time_unix_ns,
            "end_time_unix_ns": self.end_time_unix_ns,
            "span_count": self.span_count,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TraceSummary:
        return cls(
            trace_id=TraceId.from_hex(_required_string(data, "trace_id")),
            name=_required_string(data, "name"),
            service_name=_required_string(data, "service_name"),
            start_time_unix_ns=_required_integer(data, "start_time_unix_ns"),
            end_time_unix_ns=_optional_integer(data, "end_time_unix_ns"),
            span_count=_required_integer(data, "span_count"),
            status=_enum_from_data(SpanStatus, data, "status"),
        )


@dataclass(frozen=True, slots=True)
class TraceQuery:
    """Store-independent filters for listing traces."""

    start_time_unix_ns: int | None = None
    end_time_unix_ns: int | None = None
    service_name: str | None = None
    status: SpanStatus | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if self.status is not None:
            _validate_instance(self.status, SpanStatus, "status")
        if self.start_time_unix_ns is not None:
            _validate_timestamp(self.start_time_unix_ns, "start_time_unix_ns")
        if self.end_time_unix_ns is not None:
            _validate_timestamp(self.end_time_unix_ns, "end_time_unix_ns")
        if (
            self.start_time_unix_ns is not None
            and self.end_time_unix_ns is not None
            and self.end_time_unix_ns < self.start_time_unix_ns
        ):
            raise ValueError("end_time_unix_ns cannot precede start_time_unix_ns")
        if self.service_name is not None:
            _validate_nonempty_string(self.service_name, "service_name")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("limit must be an integer")
        if self.limit <= 0:
            raise ValueError("limit must be greater than zero")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "start_time_unix_ns": self.start_time_unix_ns,
            "end_time_unix_ns": self.end_time_unix_ns,
            "service_name": self.service_name,
            "status": self.status.value if self.status else None,
            "limit": self.limit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TraceQuery:
        status = _optional_string(data, "status")
        return cls(
            start_time_unix_ns=_optional_integer(data, "start_time_unix_ns"),
            end_time_unix_ns=_optional_integer(data, "end_time_unix_ns"),
            service_name=_optional_string(data, "service_name"),
            status=(_enum_from_value(SpanStatus, status, "status") if status is not None else None),
            limit=_required_integer(data, "limit"),
        )


def freeze_attributes(attributes: Mapping[str, object]) -> Attributes:
    """Validate and deeply freeze JSON-compatible attributes."""
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping")
    frozen: dict[str, FrozenJSONValue] = {}
    for key, value in attributes.items():
        if not isinstance(key, str):
            raise TypeError("attribute keys must be strings")
        frozen[key] = _freeze_json(value, f"attributes.{key}")
    return MappingProxyType(frozen)


def attributes_to_dict(attributes: Attributes) -> dict[str, JSONValue]:
    """Return a mutable JSON-compatible copy of frozen attributes."""
    return {key: _thaw_json(cast(FrozenJSONValue, value)) for key, value in attributes.items()}


def _freeze_json(value: object, path: str) -> FrozenJSONValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _thaw_json(value: FrozenJSONValue) -> JSONValue:
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    return value


def _freeze_metric_values(values: Mapping[str, int | float]) -> Mapping[str, int | float]:
    if not isinstance(values, Mapping):
        raise TypeError("metric values must be a mapping")
    if not values:
        raise ValueError("metric values must not be empty")
    frozen: dict[str, int | float] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError("metric value keys must be strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metric value {key!r} must be numeric")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric value {key!r} must be finite")
        frozen[key] = value
    return MappingProxyType(frozen)


def _validate_nonempty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _validate_timestamp(value: int, field_name: str) -> None:
    _validate_integer(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")


def _validate_instance(value: object, expected_type: type[object], field_name: str) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{field_name} must be a {expected_type.__name__}")


def _validate_end_timestamp(start: int, end: int | None, field_name: str) -> None:
    if end is None:
        return
    _validate_timestamp(end, field_name)
    if end < start:
        raise ValueError(f"{field_name} cannot precede its start timestamp")


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _required_integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_integer(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer or null")
    return value


def _required_list(
    data: Mapping[str, object],
    key: str,
    *,
    default: list[object] | None = None,
) -> list[object]:
    value = data.get(key, default)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value


def _expect_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return cast(Mapping[str, object], value)


def _attributes_from_data(data: Mapping[str, object], key: str) -> Attributes:
    value = data.get(key, {})
    return freeze_attributes(_expect_mapping(value, key))


def _enum_from_data[EnumType: StrEnum](
    enum_type: type[EnumType],
    data: Mapping[str, object],
    key: str,
) -> EnumType:
    return _enum_from_value(enum_type, _required_string(data, key), key)


def _enum_from_value[EnumType: StrEnum](
    enum_type: type[EnumType],
    value: str,
    field_name: str,
) -> EnumType:
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from error


def _event_sort_key(event: SpanEvent) -> tuple[int, int, str]:
    return (event.time_monotonic_ns, event.time_unix_ns, event.name)


def _span_sort_key(span: Span) -> tuple[int, int]:
    return (span.start_time_monotonic_ns, span.span_id.value)


def _metric_sort_key(metric: MetricPoint) -> tuple[int, int, str]:
    return (metric.time_monotonic_ns, metric.time_unix_ns, metric.name)
