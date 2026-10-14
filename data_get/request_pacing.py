"""Thread-shared, non-bursting request starts and sustained-success pacing.

This controls requests per second independently of the worker count.  A grant
does not wait for any earlier HTTP request to finish.  Call a shared cooldown
gate before ``wait`` and place ``wait`` as close to the actual HTTP start as
possible; a consumed grant may be abandoned if a new cooldown intervenes.

The optional adaptive policy scores only useful successes.  Every changed rate
discards at least 30 seconds, then pools at least 180 seconds of actual elapsed
time before comparing rates.  Counts, status codes and failure ratios do not
enter its objective.  These measurements are a control heuristic, not a claim
of statistical significance or a proven platform-wide optimal rate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

from data_get.collector import StopRequested


@dataclass(frozen=True, slots=True)
class PacingAdjustment:
    from_rate: float
    to_rate: float
    reason: str
    trend: str
    stable_score: float | None = None
    baseline_score: float | None = None
    sample_windows: tuple[int, ...] = ()
    sample_elapsed_seconds: tuple[float, ...] = ()


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


class RequestPacer:
    """Allow one immediate start, then equally spaced starts without catch-up.

    ``wait_fn(event, seconds)`` is injectable for tests and follows Event.wait:
    a truthy return means cancellation.  The normal wait is cancellation-aware
    and periodically rechecks the current rate.  Concurrent waiters share one
    grant timestamp, so idle time never accumulates a bucket of permits.
    """

    min_rate = 0.1

    def __init__(
        self,
        max_rate: float,
        start_rate: float | None = None,
        adaptive: bool = True,
        *,
        window_seconds: float = 30.0,
        measurement_seconds: float = 180.0,
        time_fn: Callable[[], float] = time.monotonic,
        wait_fn: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        self.max_rate = _finite_number(max_rate, "max_rate")
        if self.max_rate < self.min_rate:
            raise ValueError("max_rate must be at least 0.1 requests/second")
        self.current_rate = self.max_rate if start_rate is None else _finite_number(start_rate, "start_rate")
        if not self.min_rate <= self.current_rate <= self.max_rate:
            raise ValueError("start_rate must be between 0.1 and max_rate")
        self.window_seconds = _finite_number(window_seconds, "window_seconds")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.measurement_seconds = _finite_number(measurement_seconds, "measurement_seconds")
        if self.measurement_seconds < 180:
            raise ValueError("measurement_seconds must be at least 180 seconds")
        self.adaptive = bool(adaptive)
        self._time_fn = time_fn
        self._wait_fn = wait_fn if wait_fn is not None else lambda event, seconds: event.wait(seconds)
        self._lock = threading.Lock()
        self._last_grant_at: float | None = None
        self._next_grant_at: float | None = None
        self._schedule_grants = 0
        self._started_requests = 0
        self._finished_requests = 0
        self._active_requests = 0
        self._peak_active_requests = 0
        self._last_request_started_at: float | None = None
        self._start_gap_min: float | None = None
        self._start_gap_max: float | None = None
        self._start_gap_total = 0.0
        self._start_gap_count = 0
        self._window_started_at: float | None = None
        self._window_success = 0
        self._last_window_success: int | None = None
        self._last_window_elapsed: float | None = None
        self._settle_until: float | None = None
        self._settle_pending = self.adaptive
        self._settle_reason = "startup_warmup_discarded"
        self._sample_successes: list[int] = []
        self._sample_seconds: list[float] = []
        self._stable_evaluations = 0
        self._last_stable_success: int | None = None
        self._last_stable_seconds: float | None = None
        self._phase = "baseline" if self.adaptive else "fixed"
        self._trend = "warming_up" if self.adaptive else "fixed"
        self._baseline_rate: float | None = None
        self._baseline_score: float | None = None
        self._quality_reference: float | None = None
        self._step = max(self.min_rate / 2, self.max_rate / 2)
        self._direction = -1

    def wait(self, stop_event: threading.Event) -> None:
        while True:
            if stop_event.is_set():
                raise StopRequested()
            with self._lock:
                if stop_event.is_set():
                    raise StopRequested()
                now = self._time_fn()
                if self._next_grant_at is None or now >= self._next_grant_at:
                    self._last_grant_at = now
                    self._next_grant_at = now + 1.0 / self.current_rate
                    self._schedule_grants += 1
                    if self._window_started_at is None:
                        self._window_started_at = now
                        self._settle_until = now + 30.0
                    return
                delay = self._next_grant_at - now
            # The short interruptible wait also notices a rate change without
            # keeping the lock or waiting an entire old low-rate interval.
            if self._wait_fn(stop_event, min(delay, 0.1)):
                raise StopRequested()

    def record_started(
        self,
        stop_event: threading.Event | None = None,
        ready_in: Callable[[], float] | None = None,
        commit_start: Callable[[], bool] | None = None,
    ) -> None:
        """Recheck spacing/cooldown at the final application-level start mark.

        A worker can be paused after ``wait`` while another worker starts first.
        Therefore a scheduling grant alone is not enough: this second gate
        spaces actual start marks and rechecks a newly arriving cooldown.  The
        caller should invoke the HTTP function immediately after this returns;
        operating-system scheduling or packet transmission is not atomic here.
        """

        event = stop_event if stop_event is not None else threading.Event()
        while True:
            if event.is_set():
                raise StopRequested()
            with self._lock:
                if event.is_set():
                    raise StopRequested()
                cooldown_remaining = max(0.0, _finite_number(ready_in(), "ready_in result")) if ready_in else 0.0
                now = self._time_fn()
                interval = 1.0 / self.current_rate
                start_remaining = (self._last_request_started_at + interval - now
                                   if self._last_request_started_at is not None else 0.0)
                delay = max(cooldown_remaining, start_remaining)
                # Optional second gate (for a proxy pool) atomically claims its
                # real start only after this global gate is also ready.  A new
                # cooldown may have arrived since ready_in() was evaluated.
                if delay <= 0 and commit_start is not None and not commit_start():
                    delay = 0.01
                if delay <= 0:
                    if self._last_request_started_at is not None:
                        gap = max(0.0, now - self._last_request_started_at)
                        self._start_gap_min = gap if self._start_gap_min is None else min(self._start_gap_min, gap)
                        self._start_gap_max = gap if self._start_gap_max is None else max(self._start_gap_max, gap)
                        self._start_gap_total += gap
                        self._start_gap_count += 1
                    self._last_request_started_at = now
                    self._next_grant_at = max(self._next_grant_at or now, now + interval)
                    self._started_requests += 1
                    self._active_requests += 1
                    self._peak_active_requests = max(self._peak_active_requests, self._active_requests)
                    return
            if self._wait_fn(event, min(delay, 0.1)):
                raise StopRequested()

    def record_finished(self) -> None:
        """Call from the HTTP operation's finally block, including failures."""

        with self._lock:
            if self._active_requests <= 0:
                raise RuntimeError("request pacing finish recorded without an active request")
            self._active_requests -= 1
            self._finished_requests += 1

    def record_success(self) -> PacingAdjustment | None:
        return self._record(success=True)

    def record_failure(self) -> PacingAdjustment | None:
        return self._record(success=False)

    def _record(self, *, success: bool) -> PacingAdjustment | None:
        with self._lock:
            # Construction, checkpoint replay and snapshots are not crawl time.
            if self._window_started_at is None:
                return None
            now = self._time_fn()
            decision = self._advance_locked(now)
            if success:
                self._window_success += 1
            return decision

    def _advance_locked(self, now: float) -> PacingAdjustment | None:
        assert self._window_started_at is not None
        elapsed = max(0.0, now - self._window_started_at)
        if elapsed < self.window_seconds:
            return None
        successes = self._window_success
        self._last_window_success = successes
        self._last_window_elapsed = elapsed
        self._window_success = 0
        self._window_started_at = now
        if not self.adaptive:
            return None
        if self._settle_pending:
            if self._settle_until is not None and now < self._settle_until:
                return None
            self._settle_pending = False
            self._sample_successes.clear()
            self._sample_seconds.clear()
            self._trend = "stabilizing"
            return PacingAdjustment(
                self.current_rate, self.current_rate, self._settle_reason, self._trend,
                sample_windows=(successes,), sample_elapsed_seconds=(elapsed,),
            )
        self._sample_successes.append(successes)
        self._sample_seconds.append(elapsed)
        sample_seconds = sum(self._sample_seconds)
        if sample_seconds < self.measurement_seconds:
            return None
        samples = tuple(self._sample_successes)
        durations = tuple(self._sample_seconds)
        self._last_stable_success = sum(samples)
        self._last_stable_seconds = sample_seconds
        # Scores always use successes per 30 seconds, even when test windows or
        # delayed completions have a different actual duration.
        score = self._last_stable_success * 30.0 / sample_seconds
        self._sample_successes.clear()
        self._sample_seconds.clear()
        self._stable_evaluations += 1
        return self._choose_locked(score, samples, durations, now)

    def _candidate_locked(self) -> tuple[float, int]:
        for direction in (self._direction, -self._direction):
            bound = max(self.min_rate / 2, self.current_rate / 2) if direction < 0 else self.current_rate
            step = min(self._step, bound)
            target = max(self.min_rate, min(self.max_rate, self.current_rate + direction * step))
            if not math.isclose(target, self.current_rate, rel_tol=1e-10, abs_tol=1e-10):
                return target, direction
        return self.current_rate, self._direction

    def _choose_locked(self, score: float, samples: tuple[int, ...],
                       durations: tuple[float, ...], now: float) -> PacingAdjustment:
        baseline = self._baseline_score
        reason = "baseline_established"
        if self._phase == "probe":
            assert baseline is not None and self._baseline_rate is not None
            reference = self._quality_reference if self._quality_reference is not None else baseline
            better = score > baseline * 1.05
            lower_and_similar = self.current_rate < self._baseline_rate and score >= reference * 0.95
            if not (better or lower_and_similar):
                self._step = max(self.min_rate / 2, self._step / 2)
                self._direction = -self._direction
                self._phase = "refresh"
                return self._change_locked(
                    self._baseline_rate, "probe_rejected", score, baseline, samples, durations, now,
                )
            reason = "probe_improved" if better else "similar_score_lower_rate"
        elif self._phase == "refresh":
            reason = "holding_refreshed"
            if baseline is not None and baseline > 0 and self._quality_reference is not None:
                self._quality_reference *= score / baseline
            else:
                self._quality_reference = score
            if baseline is not None and baseline > 0 and (score < baseline * 0.75 or score > baseline * 1.4):
                self._step = max(self.min_rate / 2, self.max_rate / 2)
                self._direction = -1
                reason = "baseline_shift_reexplore"
        self._baseline_rate = self.current_rate
        self._baseline_score = score
        self._quality_reference = max(score, self._quality_reference or 0.0)
        target, self._direction = self._candidate_locked()
        self._phase = "probe" if target != self.current_rate else "refresh"
        return self._change_locked(target, reason, score, baseline, samples, durations, now)

    def _change_locked(self, target: float, reason: str, score: float,
                       baseline: float | None, samples: tuple[int, ...],
                       durations: tuple[float, ...], now: float) -> PacingAdjustment:
        old_rate = self.current_rate
        self.current_rate = max(self.min_rate, min(self.max_rate, target))
        if self.current_rate != old_rate:
            self._settle_pending = True
            self._settle_until = now + 30.0
            self._settle_reason = "rate_warmup_discarded"
            self._trend = "probing_up" if self.current_rate > old_rate else "probing_down"
            if self._last_grant_at is not None:
                self._next_grant_at = self._last_grant_at + 1.0 / self.current_rate
        else:
            self._trend = "holding"
        return PacingAdjustment(old_rate, self.current_rate, reason, self._trend,
                                score, baseline, samples, durations)

    def _snapshot_locked(self, now: float) -> dict[str, object]:
        return {
            "current_rate": self.current_rate,
            "max_rate": self.max_rate,
            "min_rate": self.min_rate,
            "adaptive": self.adaptive,
            "window_success": self._window_success,
            "last_window_success": self._last_window_success,
            "window_elapsed_seconds": (max(0.0, now - self._window_started_at)
                                       if self._window_started_at is not None else 0.0),
            "last_window_elapsed_seconds": self._last_window_elapsed,
            "trend": self._trend,
            "phase": self._phase,
            "settling": self._settle_pending,
            "measurement_seconds": self.measurement_seconds,
            "sample_successes": tuple(self._sample_successes),
            "sample_elapsed_seconds": tuple(self._sample_seconds),
            "stable_evaluations": self._stable_evaluations,
            "last_stable_successes": self._last_stable_success,
            "last_stable_elapsed_seconds": self._last_stable_seconds,
            "baseline_rate": self._baseline_rate,
            "baseline_score": self._baseline_score,
            "quality_reference": self._quality_reference,
            "step": self._step,
            "score_method": "pooled_successes_per_30_seconds",
            "schedule_grants": self._schedule_grants,
            "started_requests": self._started_requests,
            "finished_requests": self._finished_requests,
            "active_requests": self._active_requests,
            "peak_active_requests": self._peak_active_requests,
            "start_gap_min": self._start_gap_min,
            "start_gap_max": self._start_gap_max,
            "start_gap_mean": (self._start_gap_total / self._start_gap_count
                               if self._start_gap_count else None),
        }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_locked(self._time_fn())

    def stats(self) -> dict[str, object]:
        return self.snapshot()
