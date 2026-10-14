from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data_get.collection_lock import CollectionOutputBusyError, collection_output_lock
from data_get.collector import CollectionConfig, CollectionSummary
from tools import benchmark_collection as benchmark


class BenchmarkTests(unittest.TestCase):
    def config(self, root: Path, **kwargs) -> CollectionConfig:
        return CollectionConfig(mode="nh-online", output_csv=root / "data.csv", image_dir=root / "images",
                                state_file=root / "state.jsonl", error_log=root / "errors.jsonl", **kwargs)

    def test_dry_run_does_not_construct_network_adapter_or_write_report(self):
        with patch.object(benchmark, "run_benchmark", side_effect=AssertionError("must not run")), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(benchmark.main([]), 0)
        plan = json.loads(output.getvalue())
        self.assertFalse(plan["requestsEnabled"])
        self.assertEqual(plan["maxPages"], 700)
        self.assertEqual(plan["workers"], 30)
        self.assertTrue(plan["resume"])

    def test_source_hashes_include_only_relative_names_and_hashes(self):
        hashes = benchmark.algorithm_source_hashes()
        self.assertEqual(set(hashes), {"data_get/collector.py", "data_get/concurrency_strategies.py", "data_get/request_pacing.py", "tools/benchmark_collection.py"})
        for name, digest in hashes.items():
            self.assertFalse(Path(name).is_absolute())
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_fixed_plan_keeps_original_identity_and_reports_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), workers=30, strategy="bandit")
            normal = benchmark.benchmark_plan(config, 240, 30)
            fixed = benchmark.benchmark_plan(config, 240, 30, fixed_concurrency=15)
            self.assertEqual(fixed["strategy"], "fixed")
            self.assertEqual(fixed["fixedConcurrency"], 15)
            self.assertEqual(fixed["workers"], 30)
            self.assertEqual(fixed["identity"], normal["identity"])
            self.assertEqual(config.strategy, "bandit")

    def test_fixed_rate_selects_pacing_without_changing_worker_ceiling_or_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), workers=30, strategy="stable", request_rate_limit=1.0)
            original = benchmark.benchmark_plan(config, 240, 30)
            fixed = benchmark.benchmark_plan(config, 240, 30, fixed_concurrency=1, fixed_rate=0.5)
            self.assertEqual(fixed["strategy"], "paced")
            self.assertEqual(fixed["fixedConcurrency"], 1)
            self.assertEqual(fixed["workers"], 30)
            self.assertEqual(fixed["fixedRate"], 0.5)
            self.assertEqual(fixed["requestStartRate"], 0.5)
            self.assertFalse(fixed["adaptivePacing"])
            self.assertEqual(fixed["identity"], original["identity"])
            self.assertEqual(config.strategy, "stable")

    def test_invalid_fixed_rates_reject_before_creating_output_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), workers=30, request_rate_limit=1.0)
            for rate in (0.09, -1, 1.1, float("nan"), float("inf"), True):
                with self.subTest(rate=rate), self.assertRaises(ValueError):
                    benchmark.run_benchmark(config, report_dir=Path(temporary) / "report",
                                            fixed_rate=rate, adapter=object())
            self.assertFalse((Path(temporary) / "report").exists())

    def test_pacing_cli_remains_offline_without_run(self):
        with patch.object(benchmark, "run_benchmark", side_effect=AssertionError("must not run")), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(benchmark.main(["--fixed-rate", "0.5", "--rate-limit", "1",
                                             "--start-rate", "0.3", "--fixed-concurrency", "1"]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["strategy"], "paced")
        self.assertFalse(plan["requestsEnabled"])
        self.assertEqual(plan["requestStartRate"], 0.5)

    def test_fixed_limit_rejects_values_outside_declared_upper_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), workers=30)
            for value in (0, -1, 31, 1.5, True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    benchmark.run_benchmark(config, report_dir=Path(temporary) / "report",
                                            fixed_concurrency=value, adapter=object())
            self.assertFalse((Path(temporary) / "report").exists())

    def test_fixed_controller_never_changes_level_after_success_and_failure_windows(self):
        clock = [0.0]
        controller = benchmark.FixedConcurrency(30, 15, time_fn=lambda: clock[0])
        for window in range(12):
            clock[0] = window * 30
            self.assertIsNone(controller.record_failure())
            for _ in range(window * 5):
                self.assertIsNone(controller.record_success())
            snapshot = controller.snapshot()
            self.assertEqual(snapshot.current, 15)
            self.assertEqual(snapshot.trend, "fixed")
            self.assertEqual(snapshot.window_success, window * 5)
        self.assertIsNotNone(controller.snapshot().last_window_success)

    def test_fixed_controller_enforces_request_gate_below_worker_ceiling(self):
        controller = benchmark.FixedConcurrency(30, 2)
        stop = threading.Event()
        acquired = threading.Event()
        attempting = threading.Event()
        controller.acquire(stop)
        controller.acquire(stop)

        def third_request():
            attempting.set()
            controller.acquire(stop)
            acquired.set()
            controller.release()

        worker = threading.Thread(target=third_request)
        worker.start()
        try:
            self.assertTrue(attempting.wait(1))
            self.assertFalse(acquired.wait(0.03))
            controller.release()
            self.assertTrue(acquired.wait(1))
        finally:
            controller.release()
            stop.set()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())

    def test_output_lock_blocks_shared_csv_even_when_state_differs_and_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.config(root).resolved()
            second = replace(first, state_file=root / "other-state.jsonl")
            with collection_output_lock(first):
                with self.assertRaises(CollectionOutputBusyError):
                    with collection_output_lock(second):
                        self.fail("second writer entered")
            with collection_output_lock(second):
                pass
            self.assertTrue((root / "data.csv.collection.lock").exists())

    def test_output_lock_blocks_shared_state_even_when_csv_differs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.config(root).resolved()
            second = replace(first, output_csv=root / "other.csv")
            with collection_output_lock(first):
                with self.assertRaises(CollectionOutputBusyError):
                    with collection_output_lock(second):
                        self.fail("second checkpoint writer entered")
            # A failed multi-target acquisition must release earlier locks.
            with collection_output_lock(second):
                pass

    def test_measured_adapter_forwards_cancellation_hooks(self):
        class Adapter:
            cooldown = object()

        adapter = Adapter()
        measured = benchmark.MeasuredAdapter(adapter)
        self.assertTrue(hasattr(measured, "cooldown"))
        self.assertIs(measured.cooldown, adapter.cooldown)

    def test_stable_rate_excludes_warmup_and_shutdown_completions(self):
        with tempfile.TemporaryDirectory() as temporary:
            clock = [10.0]
            recorder = benchmark.ProgressRecorder(Path(temporary) / "telemetry.jsonl", 30,
                                                   benchmark.MeasuredAdapter(object()), clock=lambda: clock[0])
            recorder.update({"csvWrites": 0, "complete": 100, "missingImage": 5})
            recorder.start()
            clock[0] = 40
            recorder.update({"csvWrites": 60, "complete": 160, "missingImage": 5})
            recorder.sample()
            clock[0] = 100
            recorder.update({"csvWrites": 80, "complete": 180, "missingImage": 5})
            recorder.sample(stop=True)
            clock[0] = 120
            recorder.update({"csvWrites": 100, "complete": 200, "missingImage": 5})
            final = recorder.sample()
            throughput = recorder.throughput(final)
            self.assertEqual(throughput["newInformation"], 20)
            self.assertEqual(throughput["newInformationPerMinute"], 20)
            self.assertEqual(final["informationCompleted"], 205)

    def test_live_csv_writes_override_stale_progress_at_warmup_and_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            clock = [10.0]
            store = SimpleNamespace(write_count=0)
            recorder = benchmark.ProgressRecorder(Path(temporary) / "telemetry.jsonl", 30,
                                                   benchmark.MeasuredAdapter(object()), clock=lambda: clock[0])
            recorder.bind_runner(SimpleNamespace(csv_store=store))
            recorder.update({"csvWrites": 0, "complete": 100})
            recorder.start()
            # The detail write happened, but thumbnail retries keep the public
            # progress payload at zero throughout the entire trial.
            clock[0] = 40
            store.write_count = 60
            baseline = recorder.sample()
            self.assertEqual(baseline["csvWrites"], 60)
            self.assertEqual(baseline["csvWritesSource"], "csv_store")
            clock[0] = 100
            store.write_count = 80
            recorder.sample(stop=True)
            # These writes occur while draining and must remain in the CSV,
            # but cannot improve the measured throughput of the active trial.
            clock[0] = 120
            store.write_count = 100
            final = recorder.sample()
            self.assertEqual(final["csvWrites"], 100)
            self.assertEqual(recorder.throughput(final)["newInformation"], 20)
            self.assertEqual(recorder.throughput(final)["newInformationPerMinute"], 20)

    def test_multiple_observed_intervals_preserve_periodic_bursts_and_exclude_drain(self):
        with tempfile.TemporaryDirectory() as temporary:
            clock = [0.0]
            store = SimpleNamespace(write_count=0)
            recorder = benchmark.ProgressRecorder(Path(temporary) / "telemetry.jsonl", 30,
                                                   benchmark.MeasuredAdapter(object()), clock=lambda: clock[0])
            recorder.bind_runner(SimpleNamespace(csv_store=store))
            recorder.start()
            # Alternating quiet/burst windows must not disappear through a
            # median or gaps between successive 30-second boundaries.
            for seconds, count in ((30, 0), (60, 30), (90, 30), (120, 60)):
                clock[0], store.write_count = seconds, count
                recorder.sample(stop=seconds == 120)
            clock[0], store.write_count = 150, 100
            final = recorder.sample()
            intervals = recorder.throughput_intervals(final)
            self.assertEqual(intervals["wholeActiveRun"]["newInformation"], 60)
            self.assertEqual(intervals["wholeActiveRun"]["newInformationPerMinute"], 30)
            self.assertEqual(intervals["secondHalf"]["newInformation"], 30)
            self.assertEqual(intervals["middleHalf"]["newInformation"], 30)
            self.assertEqual([window["newInformation"] for window in intervals["windows"]], [0, 30, 0, 30])
            self.assertEqual(sum(window["newInformation"] for window in intervals["windows"]), 60)
            self.assertTrue(intervals["shutdownExcluded"])

    def test_interval_reports_actual_sampling_times_without_interpolated_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            clock = [0.0]
            store = SimpleNamespace(write_count=0)
            recorder = benchmark.ProgressRecorder(Path(temporary) / "telemetry.jsonl", 30,
                                                   benchmark.MeasuredAdapter(object()), clock=lambda: clock[0])
            recorder.bind_runner(SimpleNamespace(csv_store=store))
            recorder.start()
            for seconds, count in ((29, 10), (34, 11), (60, 20)):
                clock[0], store.write_count = seconds, count
                recorder.sample(stop=seconds == 60)
            intervals = recorder.throughput_intervals(recorder.sample())
            first, second = intervals["windows"]
            self.assertEqual(first["actualEndSeconds"], 29)
            self.assertEqual(second["actualStartSeconds"], 29)
            self.assertEqual(first["newInformation"] + second["newInformation"], 20)

    def test_pacing_telemetry_comes_from_current_controller(self):
        with tempfile.TemporaryDirectory() as temporary:
            measured = benchmark.MeasuredAdapter(object())
            recorder = benchmark.ProgressRecorder(Path(temporary) / "telemetry.jsonl", 0, measured,
                                                   fixed_concurrency=1, paced=True, fixed_rate=0.5)
            pacer = SimpleNamespace(stats=lambda: {"started_requests": 3, "active_requests": 2,
                                                   "current_rate": 0.5, "max_rate": 1, "start_gap_mean": 2.0})
            controller = SimpleNamespace(pacer=pacer, stats=lambda: {"strategy": "paced"})
            recorder.bind_runner(SimpleNamespace(csv_store=SimpleNamespace(write_count=1), concurrency=controller))
            recorder.update({"concurrencyStrategy": "paced", "activeRequests": 1, "actualRequestStarts": 0})
            recorder.start()
            sample = recorder.sample()
            self.assertEqual(sample["concurrencyStrategy"], "paced")
            self.assertEqual(sample["fixedRate"], 0.5)
            self.assertEqual(sample["currentConcurrency"], 1)
            self.assertTrue(sample["requestPacing"])
            self.assertEqual(sample["pacingStats"]["started_requests"], 3)
            self.assertEqual(sample["actualRequestStarts"], 3)
            self.assertEqual(sample["activeRequests"], 2)
            self.assertEqual(sample["requestRate"], 0.5)

    def test_timer_drift_does_not_append_zero_duration_tail_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            clock = [0.0]
            store = SimpleNamespace(write_count=0)
            recorder = benchmark.ProgressRecorder(Path(temporary) / "telemetry.jsonl", 30,
                                                   benchmark.MeasuredAdapter(object()), clock=lambda: clock[0])
            recorder.bind_runner(SimpleNamespace(csv_store=store))
            recorder.start()
            for seconds, count in ((30, 10), (60.0023, 20)):
                clock[0], store.write_count = seconds, count
                recorder.sample(stop=seconds > 60)
            result = recorder.throughput_intervals(recorder.sample())
            self.assertEqual(len(result["windows"]), 2)
            self.assertTrue(all(window["measuredSeconds"] > 0 for window in result["windows"]))
            self.assertEqual(sum(window["newInformation"] for window in result["windows"]), 20)
            self.assertEqual(result["wholeActiveRun"]["newInformation"], 20)
            self.assertEqual(result["wholeActiveRun"]["actualEndSeconds"], 60.0023)

    def test_fixed_rate_and_concurrency_keep_paced_wrapper_bound_to_real_adapter(self):
        from data_get.concurrency_strategies import PacedConcurrency

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root, workers=30, strategy="paced", request_rate_limit=1.0,
                                 request_start_rate=0.3)

            class Adapter:
                pacer = None

                def set_request_pacer(self, pacer):
                    self.pacer = pacer

            class Runner:
                def __init__(self, resolved, adapter, *, stop_event, progress_callback):
                    self.adapter = adapter
                    self.stop = stop_event
                    self.csv_store = SimpleNamespace(write_count=0)
                    progress_callback({"concurrencyStrategy": "paced"})

                def run(self):
                    if not isinstance(self.concurrency, PacedConcurrency):
                        raise AssertionError("fixed gate must not replace pacing controller")
                    if self.adapter.adapter.pacer is not self.concurrency.pacer:
                        raise AssertionError("pacer must be bound to the real adapter")
                    self.concurrency.pacer.record_started()
                    self.concurrency.pacer.record_finished()
                    if not self.stop.wait(1):
                        raise AssertionError("duration did not safely stop")
                    return CollectionSummary(mode="nh-online", interrupted=True)

            for fixed_concurrency, rate in ((1, 0.5), (30, 0.5), (1, None)):
                with self.subTest(concurrency=fixed_concurrency, rate=rate), patch.object(benchmark, "OnlineCollectionRunner", Runner):
                    result = benchmark.run_benchmark(config, report_dir=root / f"trial-{fixed_concurrency}-{rate}",
                                                     duration=0.025, warmup=0, sample_interval=0.01,
                                                     adapter=Adapter(), fixed_concurrency=fixed_concurrency, fixed_rate=rate)
                self.assertEqual(result["controllerStats"]["strategy"], "paced")
                self.assertEqual(result["controllerStats"]["current_concurrency"], fixed_concurrency)
                self.assertEqual(result["controllerStats"]["max_concurrency"], 30)
                self.assertEqual(result["controllerStats"]["pacing"]["adaptive"], rate is None)
                self.assertEqual(result["final"]["actualRequestStarts"], 1)
                self.assertEqual(result["throughputIntervals"]["wholeActiveRun"]["actualRequestStarts"], 1)

    def test_duration_starts_after_initialization_backs_up_before_mutation_and_closes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)
            Path(config.output_csv).write_bytes(b"original csv")
            Path(config.state_file).write_bytes(b"original checkpoint")
            calls = []

            class Adapter:
                def close(self):
                    calls.append("closed")

            class Runner:
                def __init__(self, resolved, adapter, *, stop_event, progress_callback):
                    self.stop = stop_event
                    self.callback = progress_callback
                    self.config = resolved
                    self.csv_store = SimpleNamespace(write_count=0)
                    self.callback({"csvWrites": 0, "complete": 10})
                    time.sleep(0.07)
                    self.assert_backup()
                    Path(resolved.output_csv).write_bytes(b"new csv")
                    self.callback({"csvWrites": 0, "complete": 10})

                def assert_backup(self):
                    if (root / "report" / "before" / "output.csv").read_bytes() != b"original csv":
                        raise AssertionError("backup must precede runner initialization")

                def run(self):
                    if self.concurrency.snapshot().current != 7:
                        raise AssertionError("fixed gate must be installed before run starts")
                    started = time.monotonic()
                    self.stop.wait(2)
                    calls.append(time.monotonic() - started)
                    self.csv_store.write_count = 4
                    self.callback({"csvWrites": 4, "complete": 14})
                    return CollectionSummary(mode="nh-online", interrupted=True)

            with patch.object(benchmark, "OnlineCollectionRunner", Runner):
                result = benchmark.run_benchmark(config, report_dir=root / "report", duration=0.05,
                                                 warmup=0, sample_interval=0.01, adapter=Adapter(),
                                                 fixed_concurrency=7)
            self.assertGreaterEqual(calls[0], 0.035)
            self.assertEqual(calls[1], "closed")
            self.assertTrue(result["expectedInterruption"])
            self.assertEqual(result["exitCode"], 0)
            self.assertEqual(result["stopReason"], "duration")
            self.assertEqual(result["metricVersion"], 3)
            self.assertEqual(set(result["sourceSha256"]), {"data_get/collector.py", "data_get/concurrency_strategies.py", "data_get/request_pacing.py", "tools/benchmark_collection.py"})
            self.assertIsNone(result["controllerStats"])
            self.assertEqual(result["config"]["strategy"], "fixed")
            self.assertEqual(result["config"]["fixedConcurrency"], 7)
            self.assertEqual(result["final"]["concurrencyStrategy"], "fixed")
            self.assertEqual(result["final"]["currentConcurrency"], 7)
            self.assertEqual(result["stableThroughput"]["newInformation"], 0)
            self.assertEqual(result["final"]["csvWrites"], 4)
            self.assertEqual(Path(config.output_csv).read_bytes(), b"new csv")
            self.assertEqual((root / "report" / "before" / "state.jsonl").read_bytes(), b"original checkpoint")
            self.assertTrue((root / "report" / "summary.json").exists())

    def test_cooperative_sigint_sets_stop_without_raising(self):
        reasons = []
        import signal

        original = signal.getsignal(signal.SIGINT)
        with benchmark.cooperative_sigint(reasons.append) as control:
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)
        self.assertEqual(reasons, ["keyboard_interrupt"])
        self.assertIs(signal.getsignal(signal.SIGINT), original)
        self.assertFalse(control.is_alive())

    def test_sigint_during_adapter_lock_is_deferred_to_control_thread(self):
        import signal

        class BoundedLock:
            """Fail instead of hanging the test on the former reentrant bug."""
            def __init__(self):
                self.lock = threading.Lock()

            def __enter__(self):
                if not self.lock.acquire(timeout=0.5):
                    raise AssertionError("synchronous SIGINT callback deadlocked on adapter lock")
                return self

            def __exit__(self, *_args):
                self.lock.release()

        with tempfile.TemporaryDirectory() as temporary:
            measured = benchmark.MeasuredAdapter(object())
            measured._lock = BoundedLock()
            recorder = benchmark.ProgressRecorder(Path(temporary) / "telemetry.jsonl", 0, measured)
            recorder.start()
            stopped = threading.Event()
            callback_threads = []

            def stop(_reason):
                callback_threads.append(threading.get_ident())
                recorder.sample(stop=True)
                stopped.set()

            with benchmark.cooperative_sigint(stop) as control:
                with measured._lock:
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                    # Handler returns while the interrupted main-thread lock
                    # remains held; control can finish after this block exits.
                    self.assertFalse(stopped.is_set())
                self.assertTrue(stopped.wait(1))
            self.assertFalse(control.is_alive())
            self.assertNotEqual(callback_threads, [threading.get_ident()])
            self.assertIsNotNone(recorder.active_end)

    def test_sigint_during_initialization_safely_stops_before_crawl_and_drain_not_counted(self):
        import signal

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)

            class Runner:
                def __init__(self, resolved, adapter, *, stop_event, progress_callback):
                    self.stop = stop_event
                    self.csv_store = SimpleNamespace(write_count=0)
                    self.concurrency = SimpleNamespace(stats=lambda: {"strategy": "fake", "stable_evaluations": 1})
                    progress_callback({"csvWrites": 0})
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                    if not stop_event.wait(1):
                        raise AssertionError("initialization cancellation was not delivered")

                def run(self):
                    if not self.stop.is_set():
                        raise AssertionError("crawl started without initialization stop")
                    return CollectionSummary(mode="nh-online", interrupted=True)

            with patch.object(benchmark, "OnlineCollectionRunner", Runner):
                result = benchmark.run_benchmark(config, report_dir=root / "report", duration=10,
                                                 warmup=0, adapter=object())
            self.assertEqual(result["stopReason"], "keyboard_interrupt")
            self.assertEqual(result["exitCode"], 130)
            self.assertEqual(result["stableThroughput"]["newInformation"], 0)
            self.assertEqual(result["controllerStats"], {"strategy": "fake", "stable_evaluations": 1})

    def test_interrupt_during_backup_preserves_sources_and_releases_output_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)
            Path(config.output_csv).write_bytes(b"untouched source csv")
            Path(config.state_file).write_bytes(b"untouched state")
            with patch.object(benchmark, "file_evidence", side_effect=KeyboardInterrupt), \
                 patch.object(benchmark, "OnlineCollectionRunner") as runner:
                with self.assertRaises(KeyboardInterrupt):
                    benchmark.run_benchmark(config, report_dir=root / "report", adapter=object())
            runner.assert_not_called()
            self.assertEqual(Path(config.output_csv).read_bytes(), b"untouched source csv")
            self.assertEqual(Path(config.state_file).read_bytes(), b"untouched state")
            with collection_output_lock(config.resolved()):
                pass

    def test_request_metadata_does_not_retain_error_messages_or_urls(self):
        class Adapter:
            def fetch_detail(self, item):
                raise TimeoutError("secret-url-or-credential")

        measured = benchmark.MeasuredAdapter(Adapter())
        with self.assertRaises(TimeoutError):
            measured.fetch_detail("secret-url")
        output = json.dumps(measured.snapshot())
        self.assertNotIn("secret", output)
        self.assertEqual(measured.snapshot()["fetch_detail"]["status"], {"TimeoutError": 1})

    def test_adapter_return_does_not_claim_valid_image_success(self):
        class Adapter:
            def fetch_thumbnail(self, url):
                return b"not an image"

        measured = benchmark.MeasuredAdapter(Adapter())
        measured.fetch_thumbnail("https://example.test/image")
        result = measured.snapshot()["fetch_thumbnail"]
        self.assertEqual(result["operationsReturned"], 1)
        self.assertNotIn("success", result)


if __name__ == "__main__":
    unittest.main()
