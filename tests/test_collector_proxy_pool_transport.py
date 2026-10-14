from __future__ import annotations

import json
import tempfile
import threading
import traceback
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data_get.collector import (
    CollectionConfig,
    CollectionRequestError,
    JMAdapter,
    NHAdapter,
    StopRequested,
    build_parser,
    config_from_args,
)
from data_get.proxy_pool import ProxyPool
from data_get.request_pacing import RequestPacer


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def wait(self, stop_event, seconds):
        if stop_event.is_set():
            return True
        self.advance(seconds)
        return stop_event.is_set()


class StubSession:
    def __init__(self, get):
        self.get = Mock(side_effect=get)
        self.close = Mock()
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


@contextmanager
def localhost_response_server(label: str):
    """A proxy/origin tripwire: respond locally without forwarding anywhere."""
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            requests.append(self.path)
            body = label.encode("ascii")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class CollectorProxyPoolTransportTests(unittest.TestCase):
    adapter_types = (NHAdapter, JMAdapter)
    target = "https://target.example.test/resource"

    def pool_payload(self):
        return {
            "proxies": [
                {"name": "node-a", "url": "http://fixture-user:fixture-secret@127.0.0.1:19001"},
                {"name": "node-b", "url": "http://127.0.0.1:19002"},
            ],
            "per_proxy_interval": 0.1,
            "cooldown_seconds": 1,
            "connection_cooldown_seconds": 0.1,
        }

    def make_adapter(self, adapter_type, root, *, payload=None, clock=None, reuse_connections=True):
        payload = payload if payload is not None else self.pool_payload()
        pool_file = root / "proxy-pool.json"
        pool_file.write_text(json.dumps(payload), encoding="utf-8")
        config = CollectionConfig(
            mode="nh-online" if adapter_type is NHAdapter else "jm-online",
            proxy_pool_file=pool_file,
            timeout=3,
            reuse_connections=reuse_connections,
        ).resolved()
        adapter = adapter_type(config)
        if clock is not None:
            adapter.proxy_pool = ProxyPool(payload, time_fn=clock)
        return adapter

    @contextmanager
    def mock_sessions(self, adapter_type, get, *, on_create=None):
        sessions = []

        def factory(**_kwargs):
            session = StubSession(get)
            sessions.append(session)
            if on_create is not None:
                on_create()
            return session

        name = "curl_cffi.requests.Session" if adapter_type is NHAdapter else "cloudscraper.create_scraper"
        with patch(name, side_effect=factory) as make_session:
            yield sessions, make_session

    def test_both_adapters_rotate_fairly_and_isolate_reused_sessions_per_node(self):
        for adapter_type in self.adapter_types:
            with self.subTest(adapter=adapter_type.__name__), tempfile.TemporaryDirectory() as temp:
                clock = FakeClock()
                adapter = self.make_adapter(adapter_type, Path(temp), clock=clock)
                sent_proxies = []

                def get(_url, **kwargs):
                    sent_proxies.append(kwargs["proxies"])
                    return SimpleNamespace(status_code=200, headers={})

                try:
                    with self.mock_sessions(adapter_type, get) as (sessions, factory):
                        for _ in range(4):
                            adapter._get(self.target)
                            clock.advance(0.2)
                        expected = [node["url"] for node in self.pool_payload()["proxies"]]
                        self.assertEqual([value["http"] for value in sent_proxies], expected * 2)
                        self.assertTrue(all(value["http"] == value["https"] for value in sent_proxies))
                        self.assertEqual(factory.call_count, 2)
                        self.assertEqual([session.get.call_count for session in sessions], [2, 2])
                        for index, session in enumerate(sessions):
                            self.assertTrue(all(call.kwargs["proxies"]["http"] == expected[index]
                                                for call in session.get.call_args_list))
                        snapshot = adapter.proxy_pool.snapshot(self.target)
                        self.assertEqual(snapshot["inFlight"], 0)
                        self.assertEqual([node["successes"] for node in snapshot["nodes"]], [2, 2])
                        adapter.close()
                        for session in sessions:
                            session.close.assert_called_once()
                finally:
                    adapter.close()

    def test_rate_limit_cools_only_its_node_and_does_not_set_global_host_cooldown(self):
        for adapter_type in self.adapter_types:
            with self.subTest(adapter=adapter_type.__name__), tempfile.TemporaryDirectory() as temp:
                clock = FakeClock()
                adapter = self.make_adapter(adapter_type, Path(temp), clock=clock)
                adapter.cooldown.defer = Mock()
                sent = []

                def get(_url, **kwargs):
                    sent.append(kwargs["proxies"]["http"])
                    if len(sent) == 1:
                        return SimpleNamespace(status_code=429, headers={"Retry-After": "12"})
                    return SimpleNamespace(status_code=200, headers={})

                try:
                    with self.mock_sessions(adapter_type, get):
                        with self.assertRaises(CollectionRequestError) as failure:
                            adapter._get(self.target)
                        self.assertEqual(failure.exception.status_code, 429)
                        self.assertEqual(failure.exception.retry_after, 12)
                        self.assertTrue(failure.exception.proxy_pool_cooldown)
                        snapshot = adapter.proxy_pool.snapshot(self.target)
                        self.assertEqual(snapshot["cooling"], 1)
                        self.assertEqual(snapshot["nodes"][0]["cooldownRemainingSeconds"], 12)
                        self.assertFalse(snapshot["nodes"][1]["cooling"])
                        for _ in range(2):
                            clock.advance(0.2)
                            self.assertEqual(adapter._get(self.target).status_code, 200)
                    adapter.cooldown.defer.assert_not_called()
                    expected = [node["url"] for node in self.pool_payload()["proxies"]]
                    self.assertEqual(sent, [expected[0], expected[1], expected[1]])
                    snapshot = adapter.proxy_pool.snapshot(self.target)
                    self.assertEqual(snapshot["rateLimited"], 1)
                    self.assertEqual(snapshot["failures"], 1)
                    self.assertEqual(snapshot["successes"], 2)
                    self.assertEqual(snapshot["inFlight"], 0)
                finally:
                    adapter.close()

    def test_connection_failure_is_redacted_and_releases_inflight_counters(self):
        for adapter_type in self.adapter_types:
            with self.subTest(adapter=adapter_type.__name__), tempfile.TemporaryDirectory() as temp:
                clock = FakeClock()
                adapter = self.make_adapter(adapter_type, Path(temp), clock=clock)
                pacer = RequestPacer(100, adaptive=False, time_fn=clock, wait_fn=clock.wait)
                adapter.set_request_pacer(pacer)

                def get(_url, **kwargs):
                    raise ConnectionError(f"cannot connect via {kwargs['proxies']['http']}")

                try:
                    with self.mock_sessions(adapter_type, get):
                        with self.assertRaises(CollectionRequestError) as failure:
                            adapter._get(self.target)
                    error = failure.exception
                    displayed = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                    for secret in ("fixture-user", "fixture-secret", "19001"):
                        self.assertNotIn(secret, str(error))
                        self.assertNotIn(secret, displayed)
                    self.assertIn("node-a", str(error))
                    self.assertTrue(error.retryable)
                    snapshot = adapter.proxy_pool.snapshot(self.target)
                    self.assertEqual(snapshot["inFlight"], 0)
                    self.assertEqual(snapshot["failures"], 1)
                    self.assertEqual(snapshot["cooling"], 1)
                    self.assertEqual(pacer.snapshot()["started_requests"], 1)
                    self.assertEqual(pacer.snapshot()["active_requests"], 0)
                finally:
                    adapter.close()

    def test_pacer_commits_pool_start_only_after_both_gates_accept(self):
        for adapter_type in self.adapter_types:
            with self.subTest(adapter=adapter_type.__name__), tempfile.TemporaryDirectory() as temp:
                clock = FakeClock()
                adapter = self.make_adapter(adapter_type, Path(temp), clock=clock)
                pacer = RequestPacer(100, adaptive=False, time_fn=clock, wait_fn=clock.wait)
                adapter.set_request_pacer(pacer)
                real_mark_started = adapter.proxy_pool.mark_started
                mark_calls = []

                def mark_started(lease):
                    mark_calls.append(lease)
                    # Simulate a cooldown race after ready_in() but before
                    # the pool can atomically commit the actual start.
                    return False if len(mark_calls) == 1 else real_mark_started(lease)

                def get(_url, **_kwargs):
                    snapshot = adapter.proxy_pool.snapshot(self.target)
                    self.assertEqual(snapshot["requests"], 1)
                    self.assertEqual(snapshot["inFlight"], 1)
                    self.assertEqual(pacer.snapshot()["started_requests"], 1)
                    self.assertEqual(pacer.snapshot()["active_requests"], 1)
                    return SimpleNamespace(status_code=200, headers={})

                try:
                    with self.mock_sessions(adapter_type, get), patch.object(adapter.proxy_pool, "mark_started", side_effect=mark_started):
                        adapter._get(self.target)
                    self.assertEqual(len(mark_calls), 2)
                    self.assertEqual(adapter.proxy_pool.snapshot(self.target)["requests"], 1)
                    self.assertEqual(adapter.proxy_pool.snapshot(self.target)["inFlight"], 0)
                    self.assertEqual(pacer.snapshot()["active_requests"], 0)
                finally:
                    adapter.close()

    def test_new_cooldown_on_reserved_node_reselects_other_node_without_blocking_pool(self):
        for adapter_type in self.adapter_types:
            for paced in (False, True):
                with self.subTest(adapter=adapter_type.__name__, paced=paced), tempfile.TemporaryDirectory() as temp:
                    clock = FakeClock()
                    adapter = self.make_adapter(adapter_type, Path(temp), clock=clock)
                    pool = adapter.proxy_pool
                    stop = adapter.cooldown.stop_event
                    # Leave an earlier node-a request in flight, then consume
                    # node-b's turn so the next reservation selects node-a.
                    older_a = pool.acquire(self.target, stop)
                    self.assertTrue(pool.mark_started(older_a))
                    prior_b = pool.acquire(self.target, stop)
                    self.assertTrue(pool.mark_started(prior_b))
                    pool.record_result(prior_b, status_code=200)
                    clock.advance(0.2)
                    pacer = RequestPacer(100, adaptive=False, time_fn=clock, wait_fn=clock.wait) if paced else None
                    adapter.set_request_pacer(pacer)
                    cooled = False
                    sent = []
                    before = clock.now

                    def after_node_selected():
                        nonlocal cooled
                        if not cooled:
                            cooled = True
                            # The response lands after node-a was reserved but
                            # before either gate commits its next actual start.
                            pool.record_result(older_a, status_code=429, retry_after=12)

                    def get(_url, **kwargs):
                        sent.append(kwargs["proxies"]["http"])
                        return SimpleNamespace(status_code=200, headers={})

                    try:
                        with (
                            self.mock_sessions(adapter_type, get, on_create=after_node_selected),
                            patch.object(stop, "wait", side_effect=lambda seconds: clock.wait(stop, seconds)),
                        ):
                            self.assertEqual(adapter._get(self.target).status_code, 200)
                        self.assertTrue(cooled)
                        self.assertEqual(sent, [self.pool_payload()["proxies"][1]["url"]])
                        self.assertLess(clock.now - before, 0.1, "ready node-b must not wait for node-a's 12-second cooldown")
                        snapshot = pool.snapshot(self.target)
                        self.assertEqual(snapshot["nodes"][0]["requests"], 1)
                        self.assertTrue(snapshot["nodes"][0]["cooling"])
                        self.assertEqual(snapshot["nodes"][1]["requests"], 2)
                        self.assertEqual(snapshot["inFlight"], 0)
                        if pacer is not None:
                            self.assertEqual(pacer.snapshot()["started_requests"], 1)
                            self.assertEqual(pacer.snapshot()["active_requests"], 0)
                    finally:
                        pool.release(older_a)
                        adapter.close()

    def test_cancellation_before_actual_start_never_sends_http_or_leaks_reservation(self):
        for adapter_type in self.adapter_types:
            for cancel_stage in ("session_creation", "commit_start"):
                with self.subTest(adapter=adapter_type.__name__, stage=cancel_stage), tempfile.TemporaryDirectory() as temp:
                    clock = FakeClock()
                    adapter = self.make_adapter(adapter_type, Path(temp), clock=clock)
                    pacer = RequestPacer(100, adaptive=False, time_fn=clock, wait_fn=clock.wait)
                    adapter.set_request_pacer(pacer)
                    stop = adapter.cooldown.stop_event

                    def cancel_commit(_lease):
                        stop.set()
                        return False

                    actual_get = Mock(side_effect=AssertionError("cancelled request must not reach HTTP"))
                    on_create = stop.set if cancel_stage == "session_creation" else None
                    try:
                        with self.mock_sessions(adapter_type, actual_get, on_create=on_create):
                            with patch.object(adapter.proxy_pool, "mark_started", side_effect=cancel_commit):
                                with self.assertRaises(StopRequested):
                                    adapter._get(self.target)
                        actual_get.assert_not_called()
                        snapshot = adapter.proxy_pool.snapshot(self.target)
                        self.assertEqual(snapshot["requests"], 0)
                        self.assertEqual(snapshot["inFlight"], 0)
                        self.assertEqual(snapshot["available"], 2)
                        self.assertEqual(pacer.snapshot()["started_requests"], 0)
                        self.assertEqual(pacer.snapshot()["active_requests"], 0)
                    finally:
                        adapter.close()

    def test_nh_no_reuse_still_uses_explicit_pool_and_closes_each_session(self):
        with tempfile.TemporaryDirectory() as temp:
            clock = FakeClock()
            adapter = self.make_adapter(NHAdapter, Path(temp), clock=clock, reuse_connections=False)
            sent = []

            def get(_url, **kwargs):
                sent.append(kwargs["proxies"]["http"])
                return SimpleNamespace(status_code=200, headers={})

            try:
                with self.mock_sessions(NHAdapter, get) as (sessions, factory):
                    for _ in range(3):
                        adapter._get(self.target)
                        clock.advance(0.2)
                    self.assertEqual(factory.call_count, 3)
                    for session in sessions:
                        session.close.assert_called_once()
                expected = [node["url"] for node in self.pool_payload()["proxies"]]
                self.assertEqual(sent, [*expected, expected[0]])
            finally:
                adapter.close()

    def test_local_explicit_proxies_override_no_proxy_and_environment_proxy(self):
        # Every endpoint, including the deliberately wrong environment proxy
        # and the direct-origin tripwire, is loopback-only. No remote target
        # is resolved or contacted even if proxy routing regresses.
        with (
            localhost_response_server("proxy-a") as (proxy_a, requests_a),
            localhost_response_server("proxy-b") as (proxy_b, requests_b),
            localhost_response_server("wrong-environment-proxy") as (wrong_proxy, wrong_requests),
            localhost_response_server("direct-origin") as (origin, direct_requests),
        ):
            env = {key: wrong_proxy for key in (
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
            )}
            env.update({"NO_PROXY": "*", "no_proxy": "*"})
            for adapter_type in self.adapter_types:
                for reuse in ((True, False) if adapter_type is NHAdapter else (True,)):
                    with self.subTest(adapter=adapter_type.__name__, reuse=reuse), tempfile.TemporaryDirectory() as temp:
                        payload = self.pool_payload()
                        payload["proxies"] = [{"name": "node-a", "url": proxy_a}, {"name": "node-b", "url": proxy_b}]
                        with patch.dict("os.environ", env), patch("data_get.collector.configured_proxies", side_effect=AssertionError("pool must not consult the default proxy")):
                            adapter = self.make_adapter(adapter_type, Path(temp), payload=payload, reuse_connections=reuse)
                            try:
                                target = f"{origin}/pool-routing-test"
                                responses = [adapter._get(target).text for _ in range(4)]
                                self.assertEqual(responses, ["proxy-a", "proxy-b", "proxy-a", "proxy-b"])
                                self.assertEqual(adapter.proxy_pool.snapshot(target)["inFlight"], 0)
                            finally:
                                adapter.close()
            self.assertEqual(wrong_requests, [])
            self.assertEqual(direct_requests, [])
            self.assertEqual(len(requests_a), 6)
            self.assertEqual(len(requests_b), 6)
            self.assertTrue(all(path.startswith(origin + "/") for path in requests_a + requests_b))

    def test_proxy_pool_config_does_not_change_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for mode in ("nh-online", "jm-online", "nh-local-info", "nh-local-images"):
                with self.subTest(mode=mode):
                    original = CollectionConfig(mode=mode).resolved()
                    pooled = CollectionConfig(mode=mode, proxy_pool_file=root / "pool.json").resolved()
                    self.assertEqual(original.identity, pooled.identity)
                    self.assertEqual(original.state_file, pooled.state_file)
                    self.assertEqual(pooled.proxy_pool_file, root / "pool.json")
                    with self.assertRaisesRegex(ValueError, "不能同时填写"):
                        CollectionConfig(mode=mode, proxy="http://127.0.0.1:7890", proxy_pool_file=root / "pool.json").resolved()

    def test_all_cli_modes_accept_pool_file(self):
        with tempfile.TemporaryDirectory() as temp:
            pool_file = Path(temp) / "pool.json"
            for mode in ("nh-online", "jm-online", "nh-local-info", "nh-local-images"):
                with self.subTest(mode=mode):
                    parsed = build_parser().parse_args([mode, "--proxy-pool-file", str(pool_file)])
                    self.assertEqual(config_from_args(parsed).resolved().proxy_pool_file, pool_file)


if __name__ == "__main__":
    unittest.main()
