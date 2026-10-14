"""Explicit proxy rotation with per-node pacing and per-host cooldowns.

This module never consults environment proxy settings or supplies a direct
connection fallback. Reserve a lease, call ``mark_started`` immediately before
the actual HTTP request, then return it with ``record_result`` in cleanup.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit


class ProxyPoolStopped(RuntimeError):
    """The caller cancelled while waiting for an explicit proxy endpoint."""


@dataclass(frozen=True, slots=True)
class ProxyEndpoint:
    name: str
    url: str = field(repr=False)


def _bounded_number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"代理池 {label} 必须是数值")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"代理池 {label} 必须是有限数值") from None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"代理池 {label} 必须在 {minimum:g} 至 {maximum:g} 之间")
    return number


def _endpoint_key(url: str) -> tuple[str, int]:
    """Validate without ever including the supplied URL in diagnostics."""
    if not url or any(character.isspace() or ord(character) < 32 for character in url):
        raise ValueError("代理池 URL 不可为空或包含空白字符，不允许直连节点")
    try:
        parsed = urlsplit(url)
        defaults = {"http": 80, "https": 443, "socks5": 1080, "socks5h": 1080}
        if parsed.scheme not in defaults or not parsed.hostname:
            raise ValueError()
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError()
        port = parsed.port if parsed.port is not None else defaults[parsed.scheme]
        if not 1 <= port <= 65535:
            raise ValueError()
        host = parsed.hostname.rstrip(".").casefold()
        if not host or any(character in host for character in "\\/%"):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("代理池 URL 必须是有效的 http/https/socks5/socks5h 代理地址") from None
    # Changing credentials or a URL spelling does not create another endpoint.
    return host, port


@dataclass(frozen=True, slots=True)
class ProxyPoolConfig:
    proxies: tuple[ProxyEndpoint, ...]
    per_proxy_interval: float = 3.2
    cooldown_seconds: float = 60.0
    connection_cooldown_seconds: float = 5.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProxyPoolConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("代理池配置必须是 JSON 对象")
        allowed = {"proxies", "per_proxy_interval", "cooldown_seconds", "connection_cooldown_seconds"}
        if set(payload) - allowed:
            raise ValueError("代理池配置包含不支持的字段")
        raw_proxies = payload.get("proxies")
        if not isinstance(raw_proxies, (list, tuple)) or not 1 <= len(raw_proxies) <= 32:
            raise ValueError("代理池必须包含 1 至 32 个显式代理节点")
        endpoints: list[ProxyEndpoint] = []
        seen_endpoints: set[tuple[str, int]] = set()
        seen_names: set[str] = set()
        for index, raw in enumerate(raw_proxies):
            if not isinstance(raw, Mapping) or set(raw) - {"name", "url"}:
                raise ValueError(f"代理池第 {index + 1} 个节点格式无效")
            name = raw.get("name", f"node{index + 1}")
            # Labels are public telemetry: disallow URLs / credential-like
            # strings, controls and terminal escapes in the label itself.
            if not isinstance(name, str) or not re.fullmatch(r"[\w .()\[\]#-]{1,64}", name.strip()):
                raise ValueError(f"代理池第 {index + 1} 个节点名称无效")
            name = name.strip()
            if name in seen_names:
                raise ValueError("代理池节点名称不可重复")
            url = raw.get("url")
            if not isinstance(url, str):
                raise ValueError(f"代理池第 {index + 1} 个节点必须提供显式代理 URL")
            url = url.strip()
            key = _endpoint_key(url)
            if key in seen_endpoints:
                raise ValueError("代理池代理地址不可重复（同一主机和端口只算一个节点）")
            seen_names.add(name)
            seen_endpoints.add(key)
            endpoints.append(ProxyEndpoint(name, url))
        return cls(
            tuple(endpoints),
            _bounded_number(payload.get("per_proxy_interval", 3.2), "per_proxy_interval", 0.1, 3600),
            _bounded_number(payload.get("cooldown_seconds", 60), "cooldown_seconds", 1, 86400),
            _bounded_number(payload.get("connection_cooldown_seconds", 5), "connection_cooldown_seconds", 0.1, 3600),
        )


def load_proxy_pool_config(path: str | Path) -> ProxyPoolConfig:
    """Load one explicit file; the integration layer resolves its path."""
    try:
        config_path = Path(path)
        if config_path.stat().st_size > 1024 * 1024:
            raise ValueError("代理池配置文件过大")
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("无法读取有效的代理池 JSON 配置文件") from None
    return ProxyPoolConfig.from_dict(payload)


@dataclass(frozen=True, slots=True)
class ProxyLease:
    index: int
    name: str
    proxies: dict[str, str] = field(repr=False)
    hostname: str = field(repr=False)
    _owner: object = field(repr=False, compare=False)
    _token: int = field(repr=False)


@dataclass(slots=True)
class _NodeState:
    endpoint: ProxyEndpoint
    next_start: float = 0.0
    pending_token: int | None = None
    connection_until: float = 0.0
    host_until: dict[str, float] = field(default_factory=dict)
    in_flight: int = 0
    requests: int = 0
    successes: int = 0
    failures: int = 0
    rate_limited: int = 0


@dataclass(slots=True)
class _LeaseState:
    lease: ProxyLease
    started: bool = False


class ProxyPool:
    """Fair rotation over ready endpoints; callers still own worker limits."""

    def __init__(
        self,
        config: ProxyPoolConfig | Mapping[str, Any],
        *,
        time_fn: Callable[[], float] = time.monotonic,
    ):
        if isinstance(config, ProxyPoolConfig):
            # Also validate directly constructed dataclass configurations.
            config = {
                "proxies": [{"name": endpoint.name, "url": endpoint.url} for endpoint in config.proxies],
                "per_proxy_interval": config.per_proxy_interval,
                "cooldown_seconds": config.cooldown_seconds,
                "connection_cooldown_seconds": config.connection_cooldown_seconds,
            }
        self.config = ProxyPoolConfig.from_dict(config)
        self._nodes = [_NodeState(endpoint) for endpoint in self.config.proxies]
        self._time_fn = time_fn
        self._lock = threading.Lock()
        self._owner = object()
        self._next_index = 0
        self._next_token = 0
        self._active: dict[int, _LeaseState] = {}
        self._current_index: int | None = None
        self._last_hostname = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "ProxyPool":
        return cls(load_proxy_pool_config(path))

    @staticmethod
    def _hostname(url: str) -> str:
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError()
            return parsed.hostname.rstrip(".").casefold()
        except (TypeError, ValueError):
            raise ValueError("代理池请求目标必须是有效的 HTTP(S) URL") from None

    @staticmethod
    def _ready_at(node: _NodeState, hostname: str) -> float:
        return max(node.next_start, node.connection_until, node.host_until.get(hostname, 0.0))

    def acquire(self, url: str, stop_event: threading.Event) -> ProxyLease:
        hostname = self._hostname(url)
        while True:
            if stop_event.is_set():
                raise ProxyPoolStopped("等待代理节点时已安全取消")
            with self._lock:
                if stop_event.is_set():
                    raise ProxyPoolStopped("等待代理节点时已安全取消")
                now = self._time_fn()
                wait_seconds = float("inf")
                for offset in range(len(self._nodes)):
                    index = (self._next_index + offset) % len(self._nodes)
                    node = self._nodes[index]
                    # A reservation may still be waiting on another HTTP
                    # scheduler. Do not issue overlapping unstarted leases.
                    if node.pending_token is not None:
                        continue
                    remaining = self._ready_at(node, hostname) - now
                    if remaining > 0:
                        wait_seconds = min(wait_seconds, remaining)
                        continue
                    self._next_index = (index + 1) % len(self._nodes)
                    self._next_token += 1
                    lease = ProxyLease(
                        index, node.endpoint.name,
                        {"http": node.endpoint.url, "https": node.endpoint.url},
                        hostname, self._owner, self._next_token,
                    )
                    node.pending_token = lease._token
                    self._active[lease._token] = _LeaseState(lease)
                    self._last_hostname = hostname
                    return lease
            # A bounded event wait is responsive to cancellation. It also
            # rechecks any cooldown installed by another in-flight request.
            if stop_event.wait(min(0.2, max(0.001, wait_seconds))):
                raise ProxyPoolStopped("等待代理节点时已安全取消")

    def _lease_state_locked(self, lease: ProxyLease) -> _LeaseState | None:
        if not isinstance(lease, ProxyLease) or lease._owner is not self._owner:
            raise ValueError("代理租约不属于当前代理池")
        return self._active.get(lease._token)

    def cooldown_remaining(self, lease: ProxyLease) -> float:
        """Latest start delay, suitable for a global pacer's ready callback.

        Acquiring a lease does not itself consume an interval. Only the last
        actual start and any newly observed cooldown contribute to this value.
        """
        with self._lock:
            state = self._lease_state_locked(lease)
            if state is None or state.started:
                raise ValueError("代理租约已释放或已发起请求")
            return max(0.0, self._ready_at(self._nodes[state.lease.index], state.lease.hostname) - self._time_fn())

    def mark_started(self, lease: ProxyLease) -> bool:
        """Atomically confirm an actual start; False requests another wait.

        A previous in-flight response can install a new cooldown after a global
        pacer's final check. In that race, retain the reservation and return
        False without counting or sending a request. The caller must rewait.
        """
        with self._lock:
            state = self._lease_state_locked(lease)
            if state is None or state.started:
                raise ValueError("代理租约已释放或已发起请求")
            actual = state.lease
            node = self._nodes[actual.index]
            now = self._time_fn()
            if self._ready_at(node, actual.hostname) > now:
                return False
            state.started = True
            node.pending_token = None
            node.next_start = now + self.config.per_proxy_interval
            node.requests += 1
            node.in_flight += 1
            self._current_index = actual.index
            self._last_hostname = actual.hostname
            return True

    def record_result(
        self,
        lease: ProxyLease,
        *,
        status_code: int | None = None,
        retry_after: float = 0.0,
        error: BaseException | None = None,
    ) -> bool:
        """Release once; apply host-specific HTTP or node-wide network cooldown.

        ``retry_after`` is already parsed to seconds by the HTTP integration.
        Returning False means this lease was previously returned, so a caller
        can safely call ``release`` from its cleanup path as well.
        """
        with self._lock:
            state = self._lease_state_locked(lease)
            if state is None:
                return False
            self._active.pop(lease._token)
            actual = state.lease
            node = self._nodes[actual.index]
            if not state.started:
                if node.pending_token == actual._token:
                    node.pending_token = None
                return True
            node.in_flight -= 1
            now = self._time_fn()
            if status_code in {429, 503}:
                try:
                    server_delay = float(retry_after)
                except (TypeError, ValueError, OverflowError):
                    server_delay = 0.0
                if not math.isfinite(server_delay) or server_delay < 0:
                    server_delay = 0.0
                until = now + max(self.config.cooldown_seconds, server_delay)
                node.host_until[actual.hostname] = max(node.host_until.get(actual.hostname, 0.0), until)
                node.failures += 1
                node.rate_limited += 1
            elif error is not None:
                node.connection_until = max(node.connection_until, now + self.config.connection_cooldown_seconds)
                node.failures += 1
            elif status_code is not None:
                if 200 <= status_code < 400:
                    node.successes += 1
                else:
                    node.failures += 1
            return True

    def release(self, lease: ProxyLease) -> bool:
        """Return a lease with no result, e.g. when cancelled before sending."""
        return self.record_result(lease)

    def snapshot(self, url: str | None = None) -> dict[str, Any]:
        hostname = self._hostname(url) if url is not None else None
        with self._lock:
            hostname = self._last_hostname if hostname is None else hostname
            now = self._time_fn()
            nodes: list[dict[str, Any]] = []
            for index, node in enumerate(self._nodes):
                cooldown = max(0.0, node.connection_until - now, node.host_until.get(hostname, 0.0) - now)
                interval = max(0.0, node.next_start - now)
                nodes.append({
                    "index": index,
                    "name": node.endpoint.name,
                    "available": cooldown <= 0 and interval <= 0 and node.pending_token is None,
                    "cooling": cooldown > 0,
                    "inFlight": node.in_flight,
                    "cooldownRemainingSeconds": round(cooldown, 3),
                    "intervalRemainingSeconds": round(interval, 3),
                    "requests": node.requests,
                    "successes": node.successes,
                    "failures": node.failures,
                    "rateLimited": node.rate_limited,
                })
            return {
                "total": len(nodes),
                "configured": len(nodes),
                "available": sum(node["available"] for node in nodes),
                "cooling": sum(node["cooling"] for node in nodes),
                "waitingInterval": sum(node["intervalRemainingSeconds"] > 0 for node in nodes),
                "inFlight": sum(node["inFlight"] for node in nodes),
                "requests": sum(node["requests"] for node in nodes),
                "successes": sum(node["successes"] for node in nodes),
                "failures": sum(node["failures"] for node in nodes),
                "rateLimited": sum(node["rateLimited"] for node in nodes),
                "currentProxy": self._nodes[self._current_index].endpoint.name if self._current_index is not None else None,
                "nodes": nodes,
            }
