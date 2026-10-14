"""No-network request pacing checks with deterministic clocks and real threads."""

import math
import threading
import unittest

from data_get.collector import StopRequested
from data_get.request_pacing import RequestPacer


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def wait(self, event, seconds):
        self.now += seconds
        return event.is_set()


class ManualClock:
    def __init__(self):
        self.now = 0.0
        self.condition = threading.Condition()

    def __call__(self):
        with self.condition:
            return self.now

    def advance(self, seconds):
        with self.condition:
            self.now += seconds
            self.condition.notify_all()

    def wait(self, event, seconds):
        with self.condition:
            self.condition.wait(timeout=min(seconds, 0.01))
        return event.is_set()


class PacingDriver:
    def __init__(self, **kwargs):
        self.clock = FakeClock()
        self.pacer = RequestPacer(kwargs.pop("max_rate", 4), time_fn=self.clock,
                                  wait_fn=self.clock.wait, **kwargs)
        self.pacer.wait(threading.Event())

    def finish_window(self, successes, elapsed=30, failures=0):
        start = self.clock.now
        self.clock.now = start + 0.1
        for _ in range(successes):
            self.pacer.record_success()
        for _ in range(failures):
            self.pacer.record_failure()
        self.clock.now = start + elapsed
        return self.pacer.record_failure()


class RequestPacingTests(unittest.TestCase):
    def test_finite_validated_rates_and_minimum_measurement(self):
        for invalid in (None, True, 0, -1, 0.099, math.nan, math.inf, -math.inf, "nan"):
            with self.subTest(max_rate=invalid), self.assertRaises(ValueError):
                RequestPacer(invalid)
        for invalid in (0, -1, 3, math.nan, math.inf, False):
            with self.subTest(start_rate=invalid), self.assertRaises(ValueError):
                RequestPacer(2, start_rate=invalid)
        for name, invalid in (("window_seconds", 0), ("window_seconds", math.inf),
                              ("measurement_seconds", 179.9), ("measurement_seconds", math.nan)):
            with self.subTest(parameter=name, value=invalid), self.assertRaises(ValueError):
                RequestPacer(2, **{name: invalid})
        self.assertEqual(RequestPacer(2).current_rate, 2)
        self.assertEqual(RequestPacer(2, start_rate=0.1).current_rate, 0.1)

    def test_construction_and_snapshot_do_not_start_measurement(self):
        clock = FakeClock()
        pacer = RequestPacer(2, time_fn=clock, wait_fn=clock.wait)
        clock.now = 500
        self.assertIsNone(pacer.record_success())
        self.assertEqual(pacer.snapshot()["window_elapsed_seconds"], 0)
        self.assertEqual(pacer.snapshot()["window_success"], 0)
        self.assertEqual(pacer.snapshot()["schedule_grants"], 0)
        pacer.wait(threading.Event())
        self.assertEqual(pacer.snapshot()["window_elapsed_seconds"], 0)
        clock.now = 1000
        before = pacer.stats()
        self.assertEqual(pacer.snapshot(), before)
        self.assertEqual(pacer.current_rate, 2)
        self.assertEqual(pacer.stats()["stable_evaluations"], 0)

    def test_fixed_pacing_has_no_startup_burst_and_does_not_catch_up_after_idle(self):
        clock = FakeClock()
        pacer = RequestPacer(2, adaptive=False, time_fn=clock, wait_fn=clock.wait)
        times = []
        for _ in range(3):
            pacer.wait(threading.Event())
            times.append(clock.now)
        self.assertEqual(times, [0, 0.5, 1.0])
        clock.now = 100  # A long idle / Retry-After interval creates no tokens.
        resumed = []
        for _ in range(3):
            pacer.wait(threading.Event())
            resumed.append(clock.now)
        self.assertEqual(resumed, [100, 100.5, 101])
        self.assertEqual(pacer.stats()["schedule_grants"], 6)
        self.assertEqual(pacer.stats()["started_requests"], 0)

    def test_cancellation_before_or_during_wait_never_grants_an_extra_slot(self):
        clock = FakeClock()
        stopped = threading.Event()
        stopped.set()
        pacer = RequestPacer(1, time_fn=clock, wait_fn=clock.wait)
        with self.assertRaises(StopRequested):
            pacer.wait(stopped)
        self.assertEqual(pacer.stats()["schedule_grants"], 0)
        stopped.clear()
        pacer.wait(stopped)

        def cancel_wait(event, seconds):
            event.set()
            return True
        pacer._wait_fn = cancel_wait
        with self.assertRaises(StopRequested):
            pacer.wait(stopped)
        self.assertEqual(pacer.stats()["schedule_grants"], 1)

    def test_real_threads_start_smoothly_without_waiting_for_prior_request_to_finish(self):
        clock = ManualClock()
        pacer = RequestPacer(2, adaptive=False, time_fn=clock, wait_fn=clock.wait)
        stop = threading.Event()
        finish_requests = threading.Event()
        barrier = threading.Barrier(6)
        observed = []
        errors = []
        observed_condition = threading.Condition()

        def worker():
            try:
                barrier.wait(timeout=2)
                pacer.wait(stop)
                pacer.record_started()
                with observed_condition:
                    observed.append(clock())
                    observed_condition.notify_all()
                try:
                    finish_requests.wait(timeout=3)
                finally:
                    pacer.record_finished()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        try:
            barrier.wait(timeout=2)
            for target in range(1, 6):
                with observed_condition:
                    self.assertTrue(observed_condition.wait_for(lambda: len(observed) >= target, timeout=2))
                self.assertEqual(len(observed), target)
                self.assertEqual(pacer.stats()["active_requests"], target)
                self.assertEqual(pacer.stats()["peak_active_requests"], target)
                if target < 5:
                    clock.advance(0.5)
            self.assertEqual(sorted(observed), [0, 0.5, 1.0, 1.5, 2.0])
            self.assertEqual(pacer.stats()["start_gap_min"], 0.5)
            self.assertEqual(pacer.stats()["start_gap_max"], 0.5)
            self.assertEqual(pacer.stats()["start_gap_mean"], 0.5)
        finally:
            stop.set()
            finish_requests.set()
            clock.advance(0)
            for thread in threads:
                thread.join(timeout=3)
        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(pacer.stats()["started_requests"], 5)
        self.assertEqual(pacer.stats()["finished_requests"], 5)
        self.assertEqual(pacer.stats()["active_requests"], 0)
        self.assertEqual(pacer.stats()["peak_active_requests"], 5)

    def test_actual_starts_are_separate_from_abandoned_grants_and_counters_cannot_underflow(self):
        driver = PacingDriver(adaptive=False)
        driver.clock.now = 5
        driver.pacer.wait(threading.Event())  # First grant was abandoned.
        driver.pacer.record_started()
        driver.pacer.record_finished()
        driver.clock.now = 6
        driver.pacer.wait(threading.Event())
        driver.pacer.record_started()
        driver.pacer.record_finished()
        state = driver.pacer.stats()
        self.assertEqual((state["schedule_grants"], state["started_requests"], state["finished_requests"]), (3, 2, 2))
        self.assertEqual((state["start_gap_min"], state["start_gap_max"], state["start_gap_mean"]), (1, 1, 1))
        self.assertEqual(state["peak_active_requests"], 1)
        with self.assertRaisesRegex(RuntimeError, "without an active request"):
            driver.pacer.record_finished()
        self.assertEqual(driver.pacer.stats()["active_requests"], 0)
        self.assertEqual(driver.pacer.stats()["finished_requests"], 2)

    def test_late_ticket_rechecks_actual_start_spacing_and_defers_future_grants(self):
        clock = FakeClock()
        pacer = RequestPacer(2, adaptive=False, time_fn=clock, wait_fn=clock.wait)
        stop = threading.Event()
        pacer.wait(stop)  # A gets a scheduling grant at 0, then is paused.
        clock.now = 0.5
        pacer.wait(stop)  # B gets the next grant and starts before A.
        pacer.record_started(stop)
        clock.now = 0.5001
        pacer.record_started(stop)  # A must not reuse its stale grant immediately.
        self.assertEqual(clock.now, 1.0)
        self.assertEqual(pacer.stats()["start_gap_min"], 0.5)
        self.assertEqual(pacer.stats()["active_requests"], 2)
        pacer.wait(stop)
        self.assertEqual(clock.now, 1.5)
        pacer.record_finished()
        pacer.record_finished()

    def test_cooldown_arriving_after_grant_is_rechecked_before_actual_start(self):
        clock = FakeClock()
        pacer = RequestPacer(2, adaptive=False, time_fn=clock, wait_fn=clock.wait)
        stop = threading.Event()
        pacer.wait(stop)
        cooldown_until = [2.0]

        def wait_with_extension(event, seconds):
            clock.now += seconds
            if clock.now >= 1.0:
                cooldown_until[0] = 3.0
            return event.is_set()
        pacer._wait_fn = wait_with_extension
        pacer.record_started(stop, lambda: cooldown_until[0] - clock.now)
        self.assertEqual(clock.now, 3.0)
        self.assertEqual(pacer.stats()["started_requests"], 1)
        self.assertEqual(pacer.stats()["schedule_grants"], 1)
        pacer.wait(stop)
        self.assertEqual(clock.now, 3.5)
        pacer.record_finished()

    def test_cancellation_while_waiting_to_mark_actual_start_leaves_counters_unchanged(self):
        clock = FakeClock()
        pacer = RequestPacer(2, adaptive=False, time_fn=clock, wait_fn=clock.wait)
        stop = threading.Event()
        pacer.wait(stop)
        pacer.record_started(stop)
        before = pacer.stats()

        def cancel_wait(event, seconds):
            event.set()
            return True
        pacer._wait_fn = cancel_wait
        with self.assertRaises(StopRequested):
            pacer.record_started(stop, lambda: 5)
        after = pacer.stats()
        for key in ("started_requests", "active_requests", "peak_active_requests", "start_gap_min"):
            self.assertEqual(before[key], after[key])
        pacer.record_finished()

    def test_invalid_cooldown_remaining_cannot_bypass_actual_start_gate(self):
        for value in (math.nan, math.inf, None):
            clock = FakeClock()
            pacer = RequestPacer(2, time_fn=clock, wait_fn=clock.wait)
            pacer.wait(threading.Event())
            with self.assertRaises(ValueError):
                pacer.record_started(ready_in=lambda: value)
            self.assertEqual(pacer.stats()["started_requests"], 0)

    def test_startup_burst_is_discarded_and_at_least_180_seconds_are_measured(self):
        driver = PacingDriver()
        self.assertEqual(driver.finish_window(9999).reason, "startup_warmup_discarded")
        for _ in range(5):
            self.assertIsNone(driver.finish_window(20))
            self.assertEqual(driver.pacer.current_rate, 4)
        result = driver.finish_window(20)
        self.assertEqual((result.from_rate, result.to_rate), (4, 2))
        self.assertEqual(result.stable_score, 20)
        self.assertEqual(sum(result.sample_elapsed_seconds), 180)
        self.assertEqual(driver.pacer.stats()["last_stable_successes"], 120)
        self.assertEqual(driver.finish_window(9999).reason, "rate_warmup_discarded")
        for _ in range(5):
            self.assertIsNone(driver.finish_window(30))
        improved = driver.finish_window(30)
        self.assertEqual(improved.stable_score, 30)
        self.assertEqual(improved.to_rate, 1)

    def test_short_test_windows_still_discard_a_full_30_seconds(self):
        driver = PacingDriver(window_seconds=10)
        self.assertIsNone(driver.finish_window(999, elapsed=10))
        self.assertIsNone(driver.finish_window(999, elapsed=10))
        self.assertEqual(driver.finish_window(999, elapsed=10).reason, "startup_warmup_discarded")
        self.assertEqual(driver.pacer.stats()["sample_successes"], ())
        for _ in range(17):
            self.assertIsNone(driver.finish_window(10, elapsed=10))
        result = driver.finish_window(10, elapsed=10)
        self.assertEqual(result.stable_score, 30)
        self.assertEqual(sum(result.sample_elapsed_seconds), 180)

    def test_delayed_windows_are_weighted_by_actual_elapsed_and_not_synthesized(self):
        driver = PacingDriver()
        driver.finish_window(999)
        self.assertIsNone(driver.finish_window(60, elapsed=120))
        result = driver.finish_window(60, elapsed=60)
        self.assertEqual(result.stable_score, 20)  # 120 successes / 180 seconds * 30.
        self.assertEqual(result.sample_elapsed_seconds, (120, 60))
        self.assertEqual(driver.pacer.stats()["stable_evaluations"], 1)

    def test_failure_volume_never_changes_objective_or_decisions(self):
        clean, noisy = PacingDriver(), PacingDriver()
        for successes in [500] + [20] * 6 + [500] + [25] * 6:
            self.assertEqual(clean.finish_window(successes), noisy.finish_window(successes, failures=1000))
            self.assertEqual(clean.pacer.snapshot(), noisy.pacer.snapshot())

    def test_periodic_successes_are_pooled_instead_of_median_discarded(self):
        driver = PacingDriver()
        driver.finish_window(999)
        results = [driver.finish_window(count) for count in (0, 20, 0, 0, 20, 0)]
        self.assertTrue(all(result is None for result in results[:-1]))
        self.assertAlmostEqual(results[-1].stable_score, 40 / 6)

    def test_fixed_mode_does_not_adapt_after_failures_or_arbitrary_successes(self):
        driver = PacingDriver(max_rate=4, start_rate=0.7, adaptive=False)
        for count in (0, 999, 5) * 15:
            self.assertIsNone(driver.finish_window(count))
        self.assertEqual(driver.pacer.current_rate, 0.7)
        self.assertEqual(driver.pacer.stats()["stable_evaluations"], 0)
        self.assertEqual(driver.pacer.stats()["sample_successes"], ())

    def test_rate_change_recomputes_next_grant_without_exceeding_new_rate(self):
        driver = PacingDriver()
        driver.finish_window(999)
        for _ in range(5):
            driver.finish_window(20)
        # Grant immediately before the score closes, then lower 4 -> 2 rps.
        driver.clock.now = 209.9
        driver.pacer.wait(threading.Event())
        driver.clock.now = 210
        self.assertEqual(driver.pacer.record_failure().to_rate, 2)
        driver.pacer.wait(threading.Event())
        self.assertAlmostEqual(driver.clock.now, 210.4)

    def test_many_updates_respect_rate_floor_and_independent_rate_cap(self):
        for cap in (0.1, 0.3, 1.3, 4):
            driver = PacingDriver(max_rate=cap)
            for index in range(120):
                driver.finish_window(index % 4)
                self.assertGreaterEqual(driver.pacer.current_rate, 0.1)
                self.assertLessEqual(driver.pacer.current_rate, cap)


if __name__ == "__main__":
    unittest.main()
