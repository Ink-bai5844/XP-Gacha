from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data_get.proxy_pool import ProxyPool, ProxyPoolConfig, ProxyPoolStopped, load_proxy_pool_config


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def config(count=3, **kwargs):
    return {
        "proxies": [{"name": f"node{index + 1}", "url": f"http://127.0.0.1:{17891 + index}"} for index in range(count)],
        **kwargs,
    }


class ProxyPoolTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.stop = threading.Event()
        self.pool = ProxyPool(config(), time_fn=self.clock)
        self.url = "https://target.example.test/resource"

    def acquire_started(self, pool=None, url=None):
        pool = self.pool if pool is None else pool
        lease = pool.acquire(self.url if url is None else url, self.stop)
        self.assertTrue(pool.mark_started(lease))
        return lease

    def test_explicit_protocols_and_default_names(self):
        for scheme in ("http", "https", "socks5", "socks5h"):
            with self.subTest(scheme=scheme):
                pool = ProxyPool({"proxies": [{"url": f"{scheme}://localhost:1080"}]})
                lease = self.acquire_started(pool)
                self.assertEqual(lease.name, "node1")
                self.assertEqual(lease.proxies, {"http": f"{scheme}://localhost:1080", "https": f"{scheme}://localhost:1080"})
                pool.release(lease)

    def test_invalid_or_direct_endpoints_are_rejected_without_echoing_credentials(self):
        for value in (
            "", "direct", "127.0.0.1:7890", "ftp://host:21", "http://", "http://host:0",
            "http://host:65536", "http://host:abc", "http://host/path", "http://host/?token=secret",
            "http://user:password@host/#secret", "http://user:password@bad host:7890", None,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    ProxyPool({"proxies": [{"url": value}]})
                self.assertNotIn("password", str(caught.exception))
                self.assertNotIn("secret", str(caught.exception))

    def test_duplicate_endpoints_are_rejected_even_with_different_credentials(self):
        for urls in (
            ["http://LOCALHOST:80", "http://localhost/"],
            ["http://user:secret@host:7890", "http://other:secret@host:7890"],
            ["socks5://host:1080", "socks5h://host:1080"],
        ):
            with self.subTest(urls=urls), self.assertRaises(ValueError):
                ProxyPool({"proxies": [{"url": url} for url in urls]})

    def test_names_and_pool_size_are_validated(self):
        for payload in (
            {}, {"proxies": []}, config(33), {"proxies": "url"},
            {"proxies": [{"url": "http://host:1", "name": "http://user:secret@host"}]},
            {"proxies": [{"url": "http://host:1", "name": " "}]},
            {"proxies": [{"url": "http://host:1", "name": "same"}, {"url": "http://host:2", "name": "same"}]},
            {**config(), "unknown": True},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ProxyPool(payload)
        self.assertEqual(len(ProxyPool(config(32)).config.proxies), 32)

    def test_numeric_configuration_is_bounded_and_finite(self):
        for name, invalid in (
            ("per_proxy_interval", (0, -1, 3601, float("nan"), float("inf"), 10 ** 500, True, "3.2")),
            ("cooldown_seconds", (0, -1, 86401, float("nan"), float("inf"), False, "60")),
            ("connection_cooldown_seconds", (0, -1, 3601, float("nan"), float("inf"), True, "5")),
        ):
            for value in invalid:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ProxyPool(config(**{name: value}))

    def test_round_robin_and_per_endpoint_spacing(self):
        first = [self.acquire_started() for _ in range(3)]
        self.assertEqual([lease.index for lease in first], [0, 1, 2])
        self.assertEqual(self.pool.snapshot()["available"], 0)
        self.assertEqual(self.pool.snapshot()["inFlight"], 3)
        for lease in first:
            self.pool.record_result(lease, status_code=200)
        self.clock.advance(3.2)
        second = [self.acquire_started() for _ in range(3)]
        self.assertEqual([lease.index for lease in second], [0, 1, 2])
        for lease in second:
            self.pool.release(lease)
        self.assertEqual(self.pool.snapshot()["requests"], 6)
        self.assertEqual(self.pool.snapshot()["successes"], 3)

    def test_rate_limit_is_isolated_to_proxy_and_hostname(self):
        first = self.acquire_started()
        self.pool.record_result(first, status_code=429, retry_after=90)
        next_lease = self.acquire_started()
        self.assertEqual(next_lease.index, 1)
        self.pool.release(next_lease)
        self.clock.advance(3.2)
        self.assertEqual(self.pool.snapshot(self.url)["cooling"], 1)
        self.assertEqual(self.pool.snapshot("https://images.example.test/cover")["cooling"], 0)
        third = self.acquire_started()
        self.assertEqual(third.index, 2)
        self.pool.release(third)
        other_host = self.acquire_started(url="https://images.example.test/cover")
        self.assertEqual(other_host.index, 0)
        self.pool.record_result(other_host, status_code=200)
        # Success on a separate host cannot erase the first host's cooldown.
        self.assertEqual(self.pool.snapshot(self.url)["cooling"], 1)
        self.clock.advance(87)
        self.assertEqual(self.pool.snapshot(self.url)["cooling"], 0)

    def test_missing_retry_after_uses_cooldown_and_success_cannot_shorten_it(self):
        lease = self.acquire_started()
        self.pool.record_result(lease, status_code=503, retry_after=float("nan"))
        state = self.pool.snapshot(self.url)
        self.assertEqual(state["nodes"][0]["cooldownRemainingSeconds"], 60)
        self.assertEqual(state["failures"], 1)
        self.assertEqual(state["rateLimited"], 1)
        self.assertEqual(state["successes"], 0)

    def test_connection_failure_cools_only_that_proxy_across_hosts(self):
        lease = self.acquire_started()
        self.pool.record_result(lease, error=ConnectionError("should not be exposed: secret"))
        self.assertEqual(self.pool.snapshot("https://other.example.test")["cooling"], 1)
        self.assertNotIn("secret", json.dumps(self.pool.snapshot()))
        self.clock.advance(5)
        self.assertEqual(self.pool.snapshot()["available"], 3)

    def test_older_in_flight_success_does_not_clear_newer_host_cooldown(self):
        pool = ProxyPool(config(1), time_fn=self.clock)
        first = self.acquire_started(pool)
        self.clock.advance(3.2)
        second = self.acquire_started(pool)
        pool.record_result(first, status_code=429, retry_after=120)
        pool.record_result(second, status_code=200)
        self.assertEqual(pool.snapshot(self.url)["cooling"], 1)
        self.assertEqual(pool.snapshot(self.url)["nodes"][0]["cooldownRemainingSeconds"], 120)

    def test_release_is_idempotent_and_rejects_a_foreign_lease(self):
        lease = self.acquire_started()
        self.assertTrue(self.pool.record_result(lease, status_code=200))
        self.assertFalse(self.pool.release(lease))
        self.assertFalse(self.pool.record_result(lease, status_code=429))
        self.assertEqual(self.pool.snapshot()["inFlight"], 0)
        self.assertEqual(self.pool.snapshot()["successes"], 1)
        with self.assertRaises(ValueError):
            ProxyPool(config()).release(lease)

    def test_all_unavailable_waits_and_cancels_without_a_direct_fallback(self):
        pool = ProxyPool(config(1), time_fn=self.clock)
        lease = self.acquire_started(pool)
        pool.record_result(lease, status_code=429, retry_after=3600)
        result = []
        entered = threading.Event()

        def blocked_acquire():
            entered.set()
            try:
                result.append(pool.acquire(self.url, self.stop))
            except BaseException as exc:
                result.append(exc)

        worker = threading.Thread(target=blocked_acquire)
        worker.start()
        self.assertTrue(entered.wait(1))
        self.stop.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(result[0], ProxyPoolStopped)
        self.assertEqual(pool.snapshot()["requests"], 1)

    def test_cancelled_before_acquire_does_not_reserve_a_node(self):
        self.stop.set()
        with self.assertRaises(ProxyPoolStopped):
            self.pool.acquire(self.url, self.stop)
        self.assertEqual(self.pool.snapshot()["requests"], 0)

    def test_concurrent_acquire_preserves_minimum_intervals_and_fairness(self):
        pool = ProxyPool(config(3, per_proxy_interval=0.1))
        starts = []
        result_lock = threading.Lock()

        def request():
            lease = pool.acquire(self.url, self.stop)
            self.assertTrue(pool.mark_started(lease))
            with result_lock:
                starts.append((lease.index, time.monotonic()))
            pool.record_result(lease, status_code=200)

        with ThreadPoolExecutor(max_workers=9) as executor:
            list(executor.map(lambda _index: request(), range(9)))
        for index in range(3):
            times = sorted(at for node, at in starts if node == index)
            self.assertEqual(len(times), 3)
            self.assertTrue(all(right - left >= 0.09 for left, right in zip(times, times[1:])))
        self.assertEqual(pool.snapshot()["inFlight"], 0)
        self.assertEqual(pool.snapshot()["requests"], 9)
        self.assertEqual(pool.snapshot()["successes"], 9)

    def test_credentials_are_used_only_in_transport_map_not_repr_or_snapshot(self):
        cfg = ProxyPoolConfig.from_dict({"proxies": [{"name": "香港-1", "url": "http://alice:topsecret@localhost:7890"}]})
        pool = ProxyPool(cfg)
        lease = self.acquire_started(pool)
        self.assertIn("topsecret", lease.proxies["https"])
        for visible in (repr(cfg), repr(lease), repr(pool), json.dumps(pool.snapshot(), ensure_ascii=False)):
            self.assertNotIn("alice", visible)
            self.assertNotIn("topsecret", visible)
            self.assertNotIn("localhost", visible)
        self.assertEqual(pool.snapshot()["currentProxy"], "香港-1")
        pool.release(lease)

    def test_config_file_load_supports_utf8_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.json"
            path.write_text(json.dumps(config()), encoding="utf-8-sig")
            loaded = load_proxy_pool_config(path)
            self.assertEqual(len(loaded.proxies), 3)
            self.assertEqual(ProxyPool.from_file(path).snapshot()["total"], 3)
            path.write_text("not JSON", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_proxy_pool_config(path)

    def test_delayed_actual_start_sets_interval_from_start_not_reservation(self):
        pool = ProxyPool(config(1), time_fn=self.clock)
        lease = pool.acquire(self.url, self.stop)
        self.assertEqual(pool.snapshot()["requests"], 0)
        self.assertEqual(pool.snapshot()["inFlight"], 0)
        self.assertEqual(pool.cooldown_remaining(lease), 0)
        self.clock.advance(10)
        self.assertTrue(pool.mark_started(lease))
        self.assertEqual(pool.snapshot()["requests"], 1)
        self.assertEqual(pool.snapshot()["inFlight"], 1)
        self.assertEqual(pool.snapshot()["nodes"][0]["intervalRemainingSeconds"], 3.2)
        pool.release(lease)

    def test_unstarted_reservation_release_never_counts_a_request_or_failure(self):
        lease = self.pool.acquire(self.url, self.stop)
        self.assertEqual(self.pool.snapshot()["available"], 2)
        self.pool.record_result(lease, error=RuntimeError("cancelled before request"))
        snapshot = self.pool.snapshot()
        self.assertEqual(snapshot["requests"], 0)
        self.assertEqual(snapshot["failures"], 0)
        self.assertEqual(snapshot["inFlight"], 0)
        self.assertEqual(snapshot["available"], 3)

    def test_new_cooldown_between_reservation_and_start_is_rechecked(self):
        pool = ProxyPool(config(1), time_fn=self.clock)
        running = self.acquire_started(pool)
        self.clock.advance(3.2)
        reserved = pool.acquire(self.url, self.stop)
        self.assertEqual(pool.cooldown_remaining(reserved), 0)
        pool.record_result(running, status_code=429, retry_after=120)
        self.assertFalse(pool.mark_started(reserved))
        self.assertAlmostEqual(pool.cooldown_remaining(reserved), 120)
        self.assertEqual(pool.snapshot()["requests"], 1)
        self.clock.advance(120)
        self.assertTrue(pool.mark_started(reserved))
        self.assertEqual(pool.snapshot()["requests"], 2)
        pool.release(reserved)


if __name__ == "__main__":
    unittest.main()
