"""Deterministic controller tests: no network requests or real waiting windows."""

import threading
import unittest

from data_get.collector import AdaptiveConcurrency, StopRequested
from data_get.concurrency_strategies import create_controller


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class WindowDriver:
    def __init__(self, strategy, cap=30, window_seconds=30):
        self.clock = FakeClock()
        self.controller = create_controller(
            strategy, cap, time_fn=self.clock, window_seconds=window_seconds,
        )
        self.controller.acquire(threading.Event())
        self.controller.release()

    def finish(self, successes, *, elapsed=None, failures=0):
        duration = elapsed if elapsed is not None else self.controller.window_seconds
        start = self.clock.now
        self.clock.now = start + min(0.1, duration / 2)
        for _ in range(successes):
            self.controller.record_success()
        for _ in range(failures):
            self.controller.record_failure()
        self.clock.now = start + duration
        return self.controller.record_failure()


class ConcurrencyStrategyTests(unittest.TestCase):
    def test_stable_factory_returns_original_controller(self):
        self.assertIs(type(create_controller("stable", 30)), AdaptiveConcurrency)
        with self.assertRaisesRegex(ValueError, "unknown concurrency strategy"):
            create_controller("missing", 30)

    def test_initial_cap_and_factory_injected_window(self):
        for strategy in ("adaptive-step", "bandit"):
            with self.subTest(strategy=strategy):
                driver = WindowDriver(strategy, cap=7, window_seconds=10)
                self.assertEqual(driver.controller.current, 7)
                self.assertEqual(driver.controller.max_concurrency, 7)
                self.assertEqual(driver.finish(100).reason, "startup_warmup_discarded")
                self.assertIsNone(driver.finish(20))
                change = driver.finish(20)
                self.assertEqual(change.stable_score, 20)
                self.assertLess(change.to_concurrency, 7)

    def test_startup_and_each_changed_level_discard_a_complete_window(self):
        driver = WindowDriver("adaptive-step")
        self.assertEqual(driver.finish(3000).reason, "startup_warmup_discarded")
        self.assertIsNone(driver.finish(80))
        first = driver.finish(80)
        self.assertEqual((first.from_concurrency, first.to_concurrency), (30, 15))
        self.assertEqual(first.stable_score, 80)
        self.assertEqual(driver.finish(9000).reason, "level_warmup_discarded")
        self.assertIsNone(driver.finish(100))
        second = driver.finish(100)
        self.assertEqual(second.stable_score, 100)
        self.assertEqual(second.baseline_score, 80)
        self.assertEqual(second.to_concurrency, 8)

    def test_improving_downward_probes_preserve_intermediate_levels_and_reverse_safely(self):
        driver = WindowDriver("adaptive-step")
        visited = []
        for score in (100, 120, 140, 160, 180, 200):
            driver.finish(999)  # Each newly selected level settles first.
            self.assertIsNone(driver.finish(score))
            result = driver.finish(score)
            visited.append(result.to_concurrency)
            if result.to_concurrency < result.from_concurrency:
                self.assertLessEqual(
                    result.from_concurrency - result.to_concurrency,
                    max(1, result.from_concurrency // 2),
                )
            else:
                self.assertLessEqual(result.to_concurrency, result.from_concurrency * 2)
            self.assertLessEqual(result.to_concurrency, driver.controller.max_concurrency)
        self.assertEqual(visited, [15, 8, 4, 2, 1, 2])

    def test_long_idle_window_uses_actual_elapsed_without_synthetic_probes(self):
        for strategy in ("adaptive-step", "bandit"):
            with self.subTest(strategy=strategy):
                driver = WindowDriver(strategy)
                driver.finish(1000)
                driver.finish(120, elapsed=60)
                decision = driver.finish(60)
                self.assertEqual(decision.stable_score, 60)
                self.assertEqual(decision.sample_windows, (60, 60))
                self.assertEqual(driver.controller.stats()["stable_evaluations"], 1)
                self.assertLess(driver.controller.current, 30)

    def test_fractional_rate_normalization_is_not_rounded_before_scoring(self):
        driver = WindowDriver("adaptive-step")
        driver.finish(1000)
        driver.finish(31, elapsed=60)
        result = driver.finish(31, elapsed=60)
        self.assertEqual(result.stable_score, 15.5)
        self.assertEqual(driver.controller.stats()["last_actual_elapsed_seconds"], 60)

    def test_low_or_unstable_counts_take_three_windows_and_pool_successes(self):
        for samples, expected in (((2, 8, 4), 14 / 3), ((20, 200, 22), 242 / 3)):
            driver = WindowDriver("adaptive-step")
            driver.finish(1000)
            self.assertIsNone(driver.finish(samples[0]))
            second = driver.finish(samples[1])
            self.assertEqual(second.reason, "third_pooled_sample_needed")
            self.assertEqual(driver.controller.current, 30)
            final = driver.finish(samples[2])
            self.assertEqual(final.stable_score, expected)

    def test_periodic_burst_is_not_discarded_as_a_median_outlier(self):
        for strategy in ("adaptive-step", "bandit"):
            with self.subTest(strategy=strategy):
                driver = WindowDriver(strategy)
                driver.finish(1000)
                self.assertIsNone(driver.finish(0))
                self.assertEqual(driver.finish(20).reason, "third_pooled_sample_needed")
                result = driver.finish(0)
                self.assertAlmostEqual(result.stable_score, 20 / 3)
                self.assertEqual(result.sample_windows, (0, 20, 0))
                self.assertEqual(driver.controller.stats()["last_stable_actual_success"], 20)
                self.assertEqual(driver.controller.stats()["last_stable_elapsed_seconds"], 90)

    def test_three_window_pooled_rate_weights_actual_elapsed_time(self):
        for strategy in ("adaptive-step", "bandit"):
            driver = WindowDriver(strategy)
            driver.finish(1000)
            driver.finish(30, elapsed=60)
            self.assertEqual(driver.finish(60).reason, "third_pooled_sample_needed")
            result = driver.finish(0)
            self.assertEqual(result.sample_windows, (15, 60, 0))
            self.assertEqual(result.stable_score, 22.5)  # 90 successes / 120 seconds * 30

    def test_early_stable_pair_also_pools_actual_elapsed_time(self):
        for strategy in ("adaptive-step", "bandit"):
            driver = WindowDriver(strategy)
            driver.finish(1000)
            driver.finish(30, elapsed=60)
            result = driver.finish(16)
            self.assertAlmostEqual(result.stable_score, 46 / 3)
            self.assertEqual(result.sample_windows, (15, 16))

    def test_failures_do_not_contribute_to_score_or_direction(self):
        for strategy in ("adaptive-step", "bandit"):
            clean, noisy = WindowDriver(strategy), WindowDriver(strategy)
            for successes in (1000, 40, 40, 1000, 50, 50, 1000, 80, 80):
                left = clean.finish(successes)
                right = noisy.finish(successes, failures=500)
                self.assertEqual(left, right)
                self.assertEqual(clean.controller.snapshot(), noisy.controller.snapshot())

    def test_snapshot_does_not_advance_or_change_concurrency(self):
        for strategy in ("adaptive-step", "bandit"):
            driver = WindowDriver(strategy)
            driver.clock.now = 300
            before = driver.controller.stats()
            for _ in range(20):
                snapshot = driver.controller.snapshot()
                self.assertEqual(snapshot.current, 30)
                self.assertEqual(snapshot.window_elapsed_seconds, 300)
            self.assertEqual(driver.controller.stats(), before)
            self.assertEqual(driver.controller.record_failure().reason, "startup_warmup_discarded")
            self.assertEqual(driver.controller.stats()["stable_evaluations"], 0)

    def test_rejected_step_rolls_back_shrinks_and_remeasures_baseline(self):
        driver = WindowDriver("adaptive-step")
        for count in (1000, 100, 100, 1000, 20):
            driver.finish(count)
        rejection = driver.finish(20)
        self.assertEqual(rejection.reason, "probe_rejected")
        self.assertEqual(driver.controller.current, 30)
        self.assertEqual(driver.controller.stats()["step"], 8)
        self.assertEqual(driver.finish(999).reason, "level_warmup_discarded")
        self.assertIsNone(driver.finish(90))
        refreshed = driver.finish(90)
        self.assertEqual(refreshed.reason, "holding_refreshed")
        self.assertEqual(refreshed.stable_score, 90)
        self.assertEqual(refreshed.to_concurrency, 22)

    def test_similar_throughput_prefers_lower_concurrency(self):
        driver = WindowDriver("adaptive-step")
        for count in (1000, 100, 100, 1000, 98):
            driver.finish(count)
        decision = driver.finish(98)
        self.assertEqual(decision.reason, "similar_score_lower_concurrency")
        self.assertEqual(driver.controller.stats()["baseline_concurrency"], 15)

    def test_adaptive_step_finds_lower_productive_region_quickly(self):
        driver = WindowDriver("adaptive-step")
        seen_baselines = []
        decisions = []
        for _ in range(65):
            level = driver.controller.current
            count = round(120 * min(level / 8, 8 / level))
            result = driver.finish(count)
            if result and result.stable_score is not None:
                decisions.append(result)
                seen_baselines.append(driver.controller.stats()["baseline_concurrency"])
            self.assertLessEqual(driver.controller.current, 30)
            self.assertGreaterEqual(driver.controller.current, 1)
        self.assertEqual(decisions[0].to_concurrency, 15)
        self.assertIn(8, seen_baselines)
        self.assertEqual(driver.controller.stats()["baseline_concurrency"], 8)

    def test_similar_score_tolerance_does_not_accumulate_into_large_regression(self):
        driver = WindowDriver("adaptive-step")
        for _ in range(200):
            driver.finish(driver.controller.current * 4)
        # Linear throughput peaks at 30.  One slot less is within 5%, but
        # repeatedly accepting a 3-5% loss per step would fall far below 95%.
        self.assertGreaterEqual(driver.controller.stats()["baseline_concurrency"], 29)
        self.assertEqual(driver.controller.stats()["quality_reference"], 120)

    def test_bandit_sweep_is_finite_and_revisits_stale_levels(self):
        driver = WindowDriver("bandit")
        post_sweep = []
        for _ in range(150):
            level = driver.controller.current
            result = driver.finish(round(120 * min(level / 8, 8 / level)))
            if result and result.reason.startswith("bandit_") and result.reason != "bandit_initial_sweep":
                post_sweep.append(result)
        stats = driver.controller.stats()
        self.assertEqual(stats["candidates"][0], 1)
        self.assertEqual(stats["candidates"][-1], 30)
        self.assertTrue(all(arm["visits"] > 0 for arm in stats["arms"].values()))
        self.assertEqual(stats["baseline_concurrency"], 8)
        self.assertTrue(any(result.reason == "bandit_ucb_explore" for result in post_sweep))
        for index, result in enumerate(post_sweep, start=1):
            if result.reason == "bandit_ucb_explore":
                self.assertEqual(index % 3, 0)

    def test_bandit_adapts_after_success_curve_changes(self):
        driver = WindowDriver("bandit")
        for peak in (8, 2):
            for _ in range(180):
                level = driver.controller.current
                driver.finish(round(120 * min(level / peak, peak / level)))
            self.assertEqual(driver.controller.stats()["baseline_concurrency"], peak)

    def test_one_slot_and_zero_successes_stay_bounded(self):
        for strategy in ("adaptive-step", "bandit"):
            for cap in (1, 2, 30):
                driver = WindowDriver(strategy, cap)
                for _ in range(40):
                    driver.finish(0)
                    self.assertGreaterEqual(driver.controller.current, 1)
                    self.assertLessEqual(driver.controller.current, cap)
                self.assertGreater(driver.controller.stats()["stable_evaluations"], 0)

    def test_concurrency_gate_still_honors_stop_after_reduction(self):
        for strategy in ("adaptive-step", "bandit"):
            driver = WindowDriver(strategy, cap=4)
            controller = driver.controller
            for _ in range(4):
                controller.acquire(threading.Event())
            for count in (1000, 50, 50):
                driver.finish(count)
            self.assertEqual(controller.current, 2)
            stopped = threading.Event()
            stopped.set()
            with self.assertRaises(StopRequested):
                controller.acquire(stopped)
            for _ in range(4):
                controller.release()
            controller.acquire(threading.Event())
            controller.release()
            with self.assertRaises(RuntimeError):
                controller.release()


if __name__ == "__main__":
    unittest.main()
