"""Clock abstractions used by trace collection."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ClockReading:
    """A paired wall-clock and monotonic timestamp."""

    unix_time_ns: int
    monotonic_time_ns: int

    def __post_init__(self) -> None:
        _validate_nanoseconds(self.unix_time_ns, "unix_time_ns")
        _validate_nanoseconds(self.monotonic_time_ns, "monotonic_time_ns")


class Clock(Protocol):
    """Produces paired timestamps for trace lifecycle operations."""

    def now(self) -> ClockReading:
        """Return the current wall-clock and monotonic timestamps."""
        ...


class SystemClock:
    """Clock backed by the system wall and monotonic clocks."""

    def now(self) -> ClockReading:
        return ClockReading(
            unix_time_ns=time.time_ns(),
            monotonic_time_ns=time.perf_counter_ns(),
        )


class DeterministicClock:
    """Manually advanced clock for deterministic tests and simulations."""

    def __init__(self, *, unix_time_ns: int = 0, monotonic_time_ns: int = 0) -> None:
        self._reading = ClockReading(unix_time_ns, monotonic_time_ns)
        self._lock = threading.Lock()

    def now(self) -> ClockReading:
        with self._lock:
            return self._reading

    def advance(
        self,
        nanoseconds: int,
        *,
        unix_nanoseconds: int | None = None,
    ) -> ClockReading:
        """Advance both clocks, optionally using a different wall-clock delta."""
        _validate_nanoseconds(nanoseconds, "nanoseconds")
        if unix_nanoseconds is not None:
            _validate_nanoseconds(unix_nanoseconds, "unix_nanoseconds")

        wall_delta = nanoseconds if unix_nanoseconds is None else unix_nanoseconds
        with self._lock:
            self._reading = ClockReading(
                unix_time_ns=self._reading.unix_time_ns + wall_delta,
                monotonic_time_ns=self._reading.monotonic_time_ns + nanoseconds,
            )
            return self._reading


def _validate_nanoseconds(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
