"""Canonical trace and span identifiers."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import ClassVar, Protocol, Self


@dataclass(frozen=True, slots=True, order=True)
class TraceId:
    """A non-zero 128-bit trace identifier."""

    value: int

    bit_width: ClassVar[int] = 128
    hex_width: ClassVar[int] = 32

    def __post_init__(self) -> None:
        _validate_identifier(self.value, self.bit_width, "trace ID")

    def __str__(self) -> str:
        return f"{self.value:0{self.hex_width}x}"

    @classmethod
    def from_hex(cls, value: str) -> Self:
        return cls(_parse_hex_identifier(value, cls.hex_width, "trace ID"))


@dataclass(frozen=True, slots=True, order=True)
class SpanId:
    """A non-zero 64-bit span identifier."""

    value: int

    bit_width: ClassVar[int] = 64
    hex_width: ClassVar[int] = 16

    def __post_init__(self) -> None:
        _validate_identifier(self.value, self.bit_width, "span ID")

    def __str__(self) -> str:
        return f"{self.value:0{self.hex_width}x}"

    @classmethod
    def from_hex(cls, value: str) -> Self:
        return cls(_parse_hex_identifier(value, cls.hex_width, "span ID"))


class IdGenerator(Protocol):
    """Generates canonical trace and span identifiers."""

    def new_trace_id(self) -> TraceId:
        """Return a new trace identifier."""
        ...

    def new_span_id(self) -> SpanId:
        """Return a new span identifier."""
        ...


class RandomIdGenerator:
    """Cryptographically random identifier generator."""

    def new_trace_id(self) -> TraceId:
        return TraceId(_random_nonzero_bits(TraceId.bit_width))

    def new_span_id(self) -> SpanId:
        return SpanId(_random_nonzero_bits(SpanId.bit_width))


class DeterministicIdGenerator:
    """Sequential identifier generator for deterministic tests."""

    def __init__(self, *, trace_start: int = 1, span_start: int = 1) -> None:
        _validate_identifier(trace_start, TraceId.bit_width, "trace_start")
        _validate_identifier(span_start, SpanId.bit_width, "span_start")
        self._next_trace_id = trace_start
        self._next_span_id = span_start
        self._lock = threading.Lock()

    def new_trace_id(self) -> TraceId:
        with self._lock:
            identifier = TraceId(self._next_trace_id)
            self._next_trace_id += 1
            return identifier

    def new_span_id(self) -> SpanId:
        with self._lock:
            identifier = SpanId(self._next_span_id)
            self._next_span_id += 1
            return identifier


def _validate_identifier(value: int, bit_width: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be non-zero")
    if value >= 1 << bit_width:
        raise ValueError(f"{label} must fit in {bit_width} bits")


def _parse_hex_identifier(value: str, width: int, label: str) -> int:
    if not isinstance(value, str):
        raise TypeError(f"{label} hexadecimal value must be a string")
    if len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must use lowercase hexadecimal characters")
    return int(value, 16)


def _random_nonzero_bits(bit_width: int) -> int:
    while (value := secrets.randbits(bit_width)) == 0:
        pass
    return value
