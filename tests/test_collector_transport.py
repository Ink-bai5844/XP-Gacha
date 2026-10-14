from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data_get.collector import (
    CollectionConfig, CollectionRequestError, HostCooldown, NHAdapter, JMAdapter,
    OnlineCollectionRunner, StopRequested, _parse_retry_after, _wait_for_request_start,
)


class CollectorTransportTests(unittest.TestCase):
    def test_rate_settings_do_not_change_checkpoint_and_are_validated(self):
        original = CollectionConfig(mode="nh-online", max_pages=700).resolved()
        paced = CollectionConfig(mode="nh-online", max_pages=700, strategy="paced",
                                 request_rate_limit=0.5, request_start_rate=0.25).resolved()
        self.assertEqual(original.identity, paced.identity)
        self.assertEqual(original.state_file, paced.state_file)
        for value in (0, -1, float("nan"), float("inf"), 101, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CollectionConfig(mode="nh-online", request_rate_limit=value).resolved()
        with self.assertRaises(ValueError):
            CollectionConfig(mode="nh-online", request_rate_limit=0.5, request_start_rate=1).resolved()

    def test_pacing_is_after_cooldown_and_before_actual_http_for_nh_and_jm(self):
        for adapter_type in (NHAdapter, JMAdapter):
            with self.subTest(adapter=adapter_type.__name__):
                events = []
                adapter = adapter_type(CollectionConfig(mode="nh-online", reuse_connections=False).resolved())
                adapter.cooldown.wait = lambda _url: events.append("cooldown")
                pacer = SimpleNamespace(wait=lambda _stop: events.append("pace"),
                                        record_started=lambda *_args: events.append("started"),
                                        record_finished=lambda: events.append("finished"))
                adapter.set_request_pacer(pacer)
                def get(*_args, **_kwargs):
                    events.append("http")
                    return SimpleNamespace(status_code=200, headers={})
                if adapter_type is NHAdapter:
                    with patch("curl_cffi.requests.get", side_effect=get):
                        adapter._get("https://example.test/a")
                else:
                    adapter._scraper = lambda: SimpleNamespace(get=get)
                    adapter._get("https://example.test/a")
                self.assertEqual(events, ["cooldown", "pace", "started", "http", "finished"])

    def test_new_cooldown_during_pacing_discards_old_permit(self):
        events = []
        cooldown = SimpleNamespace(stop_event=threading.Event(),
                                   wait=lambda _url: events.append("cooldown"),
                                   remaining=Mock(side_effect=[10, 0]))
        pacer = SimpleNamespace(wait=lambda _stop: events.append("pace"))
        _wait_for_request_start(cooldown, pacer, "https://example.test")
        self.assertEqual(events, ["cooldown", "pace", "cooldown", "pace"])

    def test_cancelled_pacing_never_sends_http_and_transport_failure_releases_metric(self):
        adapter = NHAdapter(CollectionConfig(mode="nh-online", reuse_connections=False).resolved())
        pacer = Mock()
        adapter.set_request_pacer(pacer)
        pacer.wait.side_effect = StopRequested
        with patch("curl_cffi.requests.get") as get:
            with self.assertRaises(StopRequested):
                adapter._get("https://example.test")
            get.assert_not_called()
            pacer.record_started.assert_not_called()
        pacer.wait.side_effect = None
        with patch("curl_cffi.requests.get", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                adapter._get("https://example.test")
        pacer.record_started.assert_called_once()
        pacer.record_finished.assert_called_once()

    def test_sessions_are_reused_per_worker_and_closed_after_join(self):
        adapter = NHAdapter(CollectionConfig(mode="nh-online").resolved())
        response = SimpleNamespace(status_code=200, headers={})
        sessions = []

        def make_session(**kwargs):
            self.assertFalse(kwargs["use_thread_local_curl"])
            session = Mock()
            session.get.return_value = response
            sessions.append(session)
            return session

        with patch("curl_cffi.requests.Session", side_effect=make_session):
            adapter._get("https://example.test/a")
            adapter._get("https://example.test/b")
            worker = threading.Thread(target=lambda: adapter._get("https://example.test/c"))
            worker.start()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(sessions), 2)
            self.assertEqual(sessions[0].get.call_count, 2)
            adapter.close()
            for session in sessions:
                session.close.assert_called_once()
            adapter._get("https://example.test/d")
            self.assertEqual(len(sessions), 3)
            adapter.close()

    def test_connection_reuse_control_is_outside_checkpoint_identity(self):
        original = CollectionConfig(mode="nh-online", max_pages=700).resolved()
        compared = CollectionConfig(mode="nh-online", max_pages=700, strategy="bandit", reuse_connections=False).resolved()
        self.assertEqual(original.identity, compared.identity)
        self.assertEqual(original.state_file, compared.state_file)
        adapter = NHAdapter(compared)
        with patch("curl_cffi.requests.get", return_value=SimpleNamespace(status_code=200)) as get:
            adapter._get("https://example.test/a")
            get.assert_called_once()

    def test_retry_after_seconds_dates_and_malformed_values(self):
        self.assertEqual(_parse_retry_after("17"), 17)
        self.assertEqual(_parse_retry_after("Thu, 01 Jan 1970 00:01:00 GMT", now=30), 30)
        for value in (None, "invalid", "-10", "NaN", "inf", "Thu, 01 Jan 1970 00:00:00 GMT"):
            self.assertEqual(_parse_retry_after(value), 0)

    def test_http_retry_after_is_shared_and_cancellable(self):
        adapter = NHAdapter(CollectionConfig(mode="nh-online", reuse_connections=False).resolved())
        with patch("curl_cffi.requests.get", return_value=SimpleNamespace(status_code=429, headers={"Retry-After": "60"})):
            with self.assertRaises(CollectionRequestError) as raised:
                adapter._get("https://example.test/a")
        self.assertEqual(raised.exception.retry_after, 60)
        adapter.cooldown.stop_event.set()
        with self.assertRaises(StopRequested):
            adapter._get("https://example.test/b")

    def test_server_delay_is_not_capped_by_request_backoff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = CollectionConfig(mode="nh-online", output_csv=root / "info.csv", image_dir=root / "images",
                                      state_file=root / "state.jsonl", error_log=root / "errors.jsonl",
                                      request_attempts=2, retry_backoff=0).resolved()
            waits = []
            runner = OnlineCollectionRunner(config, object(), sleep_fn=waits.append)
            operation = Mock(side_effect=[CollectionRequestError("busy", retry_after=75), "ok"])
            self.assertEqual(runner._attempt(operation), "ok")
            self.assertEqual(waits, [75])
            runner.csv_store.close()
            runner.checkpoint.close()
            runner.error_writer.close()

    def test_origin_cooldown_does_not_block_other_origins(self):
        cooldown = HostCooldown()
        cooldown.defer("https://limited.test/a", 60)
        cooldown.wait("https://cdn.test/b")
        cooldown.stop_event.set()
        with self.assertRaises(StopRequested):
            cooldown.wait("https://limited.test/b")

    def test_round_cleanup_reaches_wrapped_adapters_after_workers_exit(self):
        from tests.test_collectors import OneItemAdapter
        from tools.benchmark_collection import MeasuredAdapter
        from data_get.collector import run_collection

        adapter = OneItemAdapter()
        original = adapter.fetch_detail
        attempts = 0

        def detail(item):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("retry next round")
            return original(item)

        adapter.fetch_detail = detail
        adapter.close_idle_sessions = Mock()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = CollectionConfig(mode="nh-online", output_csv=root / "info.csv", image_dir=root / "images",
                                      state_file=root / "state.jsonl", error_log=root / "errors.jsonl",
                                      request_attempts=1, retry_backoff=0, max_rounds=2)
            summary = run_collection(config, adapter=MeasuredAdapter(adapter))
        self.assertTrue(summary.success)
        self.assertEqual(adapter.close_idle_sessions.call_count, 2)


if __name__ == "__main__":
    unittest.main()
