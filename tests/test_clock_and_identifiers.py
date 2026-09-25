from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest

from langgraph_instrumentation import (
    ClockReading,
    DeterministicClock,
    DeterministicIdGenerator,
    RandomIdGenerator,
    SpanId,
    SystemClock,
    TraceId,
)


def test_identifier_hex_representation_is_fixed_width_and_round_trips() -> None:
    trace_id = TraceId(0xAB)
    span_id = SpanId(0xCD)

    assert str(trace_id) == "000000000000000000000000000000ab"
    assert str(span_id) == "00000000000000cd"
    assert TraceId.from_hex(str(trace_id)) == trace_id
    assert SpanId.from_hex(str(span_id)) == span_id


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: TraceId(0), "non-zero"),
        (lambda: TraceId(1 << 128), "128 bits"),
        (lambda: SpanId(0), "non-zero"),
        (lambda: SpanId(1 << 64), "64 bits"),
        (lambda: TraceId.from_hex("1"), "exactly 32"),
        (lambda: SpanId.from_hex("z" * 16), "lowercase hexadecimal"),
        (lambda: SpanId.from_hex("A" * 16), "lowercase hexadecimal"),
        (lambda: SpanId.from_hex("+" + "1" * 15), "lowercase hexadecimal"),
    ],
)
def test_invalid_identifiers_fail_clearly(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_deterministic_id_generator_is_repeatable_and_thread_safe() -> None:
    first = DeterministicIdGenerator(trace_start=10, span_start=20)
    second = DeterministicIdGenerator(trace_start=10, span_start=20)

    assert [first.new_trace_id() for _ in range(2)] == [second.new_trace_id() for _ in range(2)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        span_ids = list(executor.map(lambda _: first.new_span_id(), range(100)))

    assert len(set(span_ids)) == 100
    assert min(identifier.value for identifier in span_ids) == 20
    assert max(identifier.value for identifier in span_ids) == 119


def test_random_id_generator_produces_valid_nonzero_ids() -> None:
    generator = RandomIdGenerator()

    assert 0 < generator.new_trace_id().value < 1 << 128
    assert 0 < generator.new_span_id().value < 1 << 64


def test_deterministic_clock_advances_wall_and_monotonic_time() -> None:
    clock = DeterministicClock(unix_time_ns=1_000, monotonic_time_ns=100)

    assert clock.now() == ClockReading(1_000, 100)
    assert clock.advance(25) == ClockReading(1_025, 125)
    assert clock.advance(10, unix_nanoseconds=50) == ClockReading(1_075, 135)


def test_clocks_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ClockReading(-1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        DeterministicClock().advance(-1)
    with pytest.raises(TypeError, match="must be an integer"):
        ClockReading(cast(int, 1.5), 0)
    with pytest.raises(TypeError, match="must be an integer"):
        DeterministicClock().advance(cast(int, True))


def test_system_clock_returns_valid_reading() -> None:
    reading = SystemClock().now()

    assert reading.unix_time_ns > 0
    assert reading.monotonic_time_ns > 0
