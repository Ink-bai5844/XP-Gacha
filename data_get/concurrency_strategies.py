"""Success-throughput controllers with bounded, request-level concurrency.

The original ``stable`` controller remains available unchanged.  New policies
share a settling window and a bounded stable sampler, so they compare sustained
useful completions rather than the initial burst or the number of failed calls.
Stable pairs can finish early; low or variable rates take a third window.  The
score pools actual successful completions over actual elapsed time, preserving
periodic success bursts instead of dropping them as median outliers.
No policy sends requests itself or changes a site's access controls.
"""

from __future__ import annotations

import math
import time
from typing import Callable

from data_get.collector import (
    AdaptiveConcurrency,
    ConcurrencyAdjustment,
    ThroughputSnapshot,
)


class _SettledThroughputController(AdaptiveConcurrency):
    """Reuse the existing thread-safe request gate, with settled rate samples."""

    strategy = "settled"

    def __init__(
        self,
        max_concurrency: int,
        *,
        window_seconds: float = 30.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(max_concurrency, window_seconds=window_seconds, time_fn=time_fn)
        self._settle_pending = True
        self._settle_reason = "startup_warmup_discarded"
        self._rate_samples: list[float] = []
        self._sample_successes: list[int] = []
        self._sample_seconds: list[float] = []
        self._last_actual_success: int | None = None
        self._last_actual_elapsed: float | None = None
        self._last_stable_actual_success: int | None = None
        self._last_stable_elapsed: float | None = None
        self._stable_evaluations = 0
        self._phase = "baseline"

    def _clear_samples_locked(self) -> None:
        self._rate_samples.clear()
        self._sample_successes.clear()
        self._sample_seconds.clear()

    def _advance_locked(self, now: float) -> ConcurrencyAdjustment | None:
        if self._window_started_at is None:
            self._window_started_at = now
            return None
        elapsed = max(0.0, now - self._window_started_at)
        if elapsed < self.window_seconds:
            return None
        # A delayed completion can close a 60- or 90-second window.  Normalize
        # by the actual duration; do not pretend its total arrived in 30 seconds
        # and do not manufacture decisions for concurrency levels never used.
        rate = self._window_success * self.window_seconds / elapsed
        self._last_actual_success = self._window_success
        self._last_actual_elapsed = elapsed
        previous = self._last_window_success
        self._previous_window_success = previous
        self._last_window_success = round(rate)
        self._window_success = 0
        self._window_started_at = now

        if self._settle_pending:
            self._settle_pending = False
            self._clear_samples_locked()
            self._trend = "stabilizing"
            return self._decision_locked(
                previous_window_success=previous,
                last_window_success=round(rate),
                reason=self._settle_reason,
                sample_windows=(round(rate),),
            )

        self._rate_samples.append(rate)
        self._sample_successes.append(self._last_actual_success)
        self._sample_seconds.append(elapsed)
        samples = tuple(self._rate_samples)
        visible_samples = tuple(round(value) for value in samples)
        if len(samples) == 1:
            return None
        if len(samples) == 2:
            larger = max(samples)
            if larger < 10 or abs(samples[0] - samples[1]) > max(2.0, larger * 0.15):
                return self._decision_locked(
                    previous_window_success=previous,
                    last_window_success=round(rate),
                    reason="third_pooled_sample_needed",
                    sample_windows=visible_samples,
                )
        # Pool successes and wall time rather than taking a median (which loses
        # a periodic 0/20/0 burst) or an unweighted mean of rates (which gives a
        # delayed 60-second window the same weight as a 30-second window).
        self._last_stable_actual_success = sum(self._sample_successes)
        self._last_stable_elapsed = sum(self._sample_seconds)
        score = self._last_stable_actual_success * self.window_seconds / self._last_stable_elapsed
        self._clear_samples_locked()
        self._stable_evaluations += 1
        return self._apply_stable_score_locked(
            float(score),
            visible_samples,
            previous_window_success=previous,
            last_window_success=round(rate),
        )

    def _change_locked(self, target: int, *, reason: str, score: float,
                       baseline: float | None, samples: tuple[int, ...],
                       previous: int | None, last: int) -> ConcurrencyAdjustment:
        old = self.current
        self.current = max(1, min(self.max_concurrency, target))
        if self.current != old:
            self._settle_pending = True
            self._settle_reason = "level_warmup_discarded"
            self._clear_samples_locked()
            self._trend = "probing_up" if self.current > old else "probing_down"
            self._condition.notify_all()
        else:
            self._trend = "holding"
        return self._decision_locked(
            previous_window_success=previous,
            last_window_success=last,
            reason=reason,
            stable_score=score,
            baseline_score=baseline,
            sample_windows=samples,
            from_concurrency=old,
            to_concurrency=self.current,
        )

    def snapshot(self) -> ThroughputSnapshot:
        # Snapshot is observational: UI polling must never move a window or
        # change the request limit.  Report actual elapsed time, even after idle.
        now = self._time_fn()
        with self._lock:
            elapsed = (max(0.0, now - self._window_started_at)
                       if self._window_started_at is not None else 0.0)
            return ThroughputSnapshot(
                current=self.current,
                window_success=self._window_success,
                last_window_success=self._last_window_success,
                previous_window_success=self._previous_window_success,
                window_elapsed_seconds=elapsed,
                trend=self._trend,
            )

    def _stats_locked(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "phase": self._phase,
            "settling": self._settle_pending,
            "sample_rates": tuple(self._rate_samples),
            "sample_actual_successes": tuple(self._sample_successes),
            "sample_elapsed_seconds": tuple(self._sample_seconds),
            "score_method": "pooled_success_rate",
            "stable_evaluations": self._stable_evaluations,
            "last_actual_success": self._last_actual_success,
            "last_actual_elapsed_seconds": self._last_actual_elapsed,
            "last_stable_actual_success": self._last_stable_actual_success,
            "last_stable_elapsed_seconds": self._last_stable_elapsed,
            "baseline_concurrency": self._baseline_concurrency,
            "baseline_score": self._baseline_success,
        }

    def stats(self) -> dict[str, object]:
        with self._lock:
            return self._stats_locked()


class AdaptiveStepConcurrency(_SettledThroughputController):
    """Coarse-to-fine search with bounded moves and fresh rollback baselines."""

    strategy = "adaptive-step"

    def __init__(self, max_concurrency: int, **kwargs) -> None:
        super().__init__(max_concurrency, **kwargs)
        self._step = max(1, math.ceil(self.max_concurrency / 2))
        self._direction = -1
        self._relative_tolerance = 0.05
        self._quality_reference: float | None = None

    def _candidate_locked(self, direction: int) -> tuple[int, int]:
        # Keep the coarse-to-fine step budget, but bound its effect relative to
        # the current level.  A successful 30 -> 15 probe should next try 8,
        # not jump past that region to 1; a boundary reversal from 1 should
        # similarly try 2 rather than immediately returning to 16.
        for candidate_direction in (direction, -direction):
            bound = max(1, self.current // 2) if candidate_direction < 0 else self.current
            effective_step = min(self._step, bound)
            target = max(1, min(
                self.max_concurrency, self.current + candidate_direction * effective_step,
            ))
            if target != self.current:
                return target, candidate_direction
        return self.current, direction

    def _apply_stable_score_locked(self, score: float, samples: tuple[int, ...], *,
                                   previous_window_success: int | None,
                                   last_window_success: int) -> ConcurrencyAdjustment:
        baseline = self._baseline_success
        reason = "baseline_established"
        if self._phase == "probe":
            assert baseline is not None and self._baseline_concurrency is not None
            tolerance = baseline * self._relative_tolerance
            better = score > baseline + tolerance
            reference = self._quality_reference if self._quality_reference is not None else baseline
            lower_with_similar_score = (
                self.current < self._baseline_concurrency
                and score >= reference * (1 - self._relative_tolerance)
            )
            if not (better or lower_with_similar_score):
                self._step = max(1, math.ceil(self._step / 2))
                self._direction = -self._direction
                self._phase = "refresh"
                return self._change_locked(
                    self._baseline_concurrency, reason="probe_rejected", score=score,
                    baseline=baseline, samples=samples,
                    previous=previous_window_success, last=last_window_success,
                )
            reason = "probe_improved" if better else "similar_score_lower_concurrency"
        elif self._phase == "refresh":
            reason = "holding_refreshed"
            # Re-scale the quality reference only on a fresh measurement at
            # the same concurrency.  Successive "within 5%" lower probes must
            # not compound into an arbitrarily large throughput regression.
            if baseline is not None and baseline > 0 and self._quality_reference is not None:
                self._quality_reference *= score / baseline
            else:
                self._quality_reference = score
            # A substantial change in the freshly remeasured baseline signals
            # a changed operating regime.  Reopen coarse exploration instead of
            # staying indefinitely at a one-slot step inherited from yesterday.
            if baseline is not None and baseline > 0 and (
                score < baseline * 0.75 or score > baseline * 1.4
            ):
                self._step = max(1, math.ceil(self.max_concurrency / 2))
                self._direction = -1
                reason = "baseline_shift_reexplore"

        self._baseline_concurrency = self.current
        self._baseline_success = score
        self._quality_reference = max(score, self._quality_reference or 0.0)
        target, direction = self._candidate_locked(self._direction)
        self._direction = direction
        self._phase = "probe" if target != self.current else "refresh"
        return self._change_locked(
            target, reason=reason, score=score, baseline=baseline, samples=samples,
            previous=previous_window_success, last=last_window_success,
        )

    def _stats_locked(self) -> dict[str, object]:
        return {**super()._stats_locked(), "step": self._step, "direction": self._direction,
                "quality_reference": self._quality_reference}


class BanditConcurrency(_SettledThroughputController):
    """Discounted UCB on geometric levels, with bounded exploration frequency.

    Each candidate first gets one stable evaluation.  Thereafter two of every
    three evaluations exploit the best measured throughput; only the third may
    use UCB to revisit an uncertain or stale candidate.  Discounted evidence
    makes old observations easier to replace as network conditions change.
    """

    strategy = "bandit"

    def __init__(self, max_concurrency: int, **kwargs) -> None:
        super().__init__(max_concurrency, **kwargs)
        candidates = {1, self.max_concurrency}
        level = self.max_concurrency
        while level > 1:
            level = max(1, round(level / 2))
            candidates.add(level)
        self.candidates = tuple(sorted(candidates))
        self._weights = {level: 0.0 for level in self.candidates}
        self._totals = {level: 0.0 for level in self.candidates}
        self._visits = {level: 0 for level in self.candidates}
        self._discount = 0.85
        self._post_sweep_evaluations = 0

    def _mean(self, level: int) -> float:
        return self._totals[level] / self._weights[level] if self._weights[level] else 0.0

    def _best_locked(self) -> int:
        best_score = max(self._mean(level) for level in self.candidates)
        # Within five percent, prefer fewer in-flight requests.  This also
        # resolves the all-zero case deterministically at one slot.
        return min(level for level in self.candidates if self._mean(level) >= best_score * 0.95)

    def _apply_stable_score_locked(self, score: float, samples: tuple[int, ...], *,
                                   previous_window_success: int | None,
                                   last_window_success: int) -> ConcurrencyAdjustment:
        previous_baseline = self._baseline_success
        for level in self.candidates:
            self._weights[level] *= self._discount
            self._totals[level] *= self._discount
        self._weights[self.current] += 1.0
        self._totals[self.current] += score
        self._visits[self.current] += 1

        unvisited = [level for level in reversed(self.candidates) if not self._visits[level]]
        if unvisited:
            target = unvisited[0]
            self._phase = "sweep"
            reason = "bandit_initial_sweep"
        else:
            best = self._best_locked()
            self._baseline_concurrency = best
            self._baseline_success = self._mean(best)
            self._post_sweep_evaluations += 1
            target = best
            self._phase = "exploit"
            reason = "bandit_exploit"
            if self._post_sweep_evaluations % 3 == 0 and len(self.candidates) > 1:
                scale = max(1.0, max(self._mean(level) for level in self.candidates))
                total_weight = sum(self._weights.values())
                def ucb(level: int) -> tuple[float, int]:
                    bonus = 0.35 * math.sqrt(
                        math.log1p(total_weight) / max(self._weights[level], 1e-12)
                    )
                    return self._mean(level) / scale + bonus, -level
                target = max(self.candidates, key=ucb)
                if target != best:
                    self._phase = "explore"
                    reason = "bandit_ucb_explore"
        return self._change_locked(
            target, reason=reason, score=score, baseline=previous_baseline, samples=samples,
            previous=previous_window_success, last=last_window_success,
        )

    def _stats_locked(self) -> dict[str, object]:
        return {
            **super()._stats_locked(),
            "candidates": self.candidates,
            "arms": {level: {"score": self._mean(level), "weight": self._weights[level],
                              "visits": self._visits[level]} for level in self.candidates},
        }


class PacedConcurrency(AdaptiveConcurrency):
    """Separate a fixed in-flight ceiling from a smooth global launch rate."""

    strategy = "paced"

    def __init__(self, max_concurrency: int, *, request_rate_limit: float = 1.0,
                 request_start_rate: float | None = None, adaptive_pacing: bool = True,
                 request_concurrency: int | None = None, **kwargs) -> None:
        super().__init__(max_concurrency, **kwargs)
        from data_get.request_pacing import RequestPacer
        if request_concurrency is not None:
            if isinstance(request_concurrency, bool) or not 1 <= request_concurrency <= self.max_concurrency:
                raise ValueError("request_concurrency must be within the worker ceiling")
            self.current = int(request_concurrency)
        self.pacer = RequestPacer(request_rate_limit, request_start_rate, adaptive_pacing,
                                  window_seconds=self.window_seconds, time_fn=self._time_fn)

    def _advance_locked(self, now: float) -> ConcurrencyAdjustment | None:
        if self._window_started_at is None:
            self._window_started_at = now
        elapsed = max(0.0, now - self._window_started_at)
        if elapsed >= self.window_seconds:
            self._previous_window_success = self._last_window_success
            self._last_window_success = round(self._window_success * self.window_seconds / elapsed)
            self._window_success = 0
            self._window_started_at = now
        self._trend = "pacing"
        return None

    def _record(self, *, success: bool) -> None:
        super()._record(success=success)
        decision = self.pacer.record_success() if success else self.pacer.record_failure()
        if decision is not None:
            print(f"[采集] 逐发频率 {decision.from_rate:.3g}→{decision.to_rate:.3g} 请求/秒；"
                  f"{decision.reason}（仅比较稳定成功吞吐）", flush=True)
        return None

    def stats(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {"strategy": self.strategy, "current_concurrency": snapshot.current,
                "max_concurrency": self.max_concurrency, "pacing": self.pacer.stats()}


def create_controller(
    strategy: str,
    max_concurrency: int,
    *,
    window_seconds: float = 30.0,
    time_fn: Callable[[], float] = time.monotonic,
    request_rate_limit: float = 1.0,
    request_start_rate: float | None = None,
    adaptive_pacing: bool = True,
    request_concurrency: int | None = None,
) -> AdaptiveConcurrency:
    """Build a controller; ``stable`` retains the original implementation."""

    if strategy == "paced":
        return PacedConcurrency(max_concurrency, window_seconds=window_seconds, time_fn=time_fn,
                                request_rate_limit=request_rate_limit, request_start_rate=request_start_rate,
                                adaptive_pacing=adaptive_pacing, request_concurrency=request_concurrency)
    controllers = {
        "stable": AdaptiveConcurrency,
        "adaptive-step": AdaptiveStepConcurrency,
        "bandit": BanditConcurrency,
    }
    try:
        controller = controllers[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown concurrency strategy: {strategy!r}") from exc
    return controller(max_concurrency, window_seconds=window_seconds, time_fn=time_fn)
