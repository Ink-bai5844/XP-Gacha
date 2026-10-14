"""Measure resumable collection without counting startup as throughput.

The default invocation prints a plan only.  ``--run`` explicitly enables real
requests. Completed CSV rows and images are retained across every trial; the
backup is evidence/recovery material, never automatically restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_get.collection_lock import CollectionOutputBusyError, collection_output_lock
from data_get.collector import AdaptiveConcurrency, CollectionConfig, NHAdapter, OnlineCollectionRunner


def _validate_fixed_concurrency(value: int | None, maximum: int) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum):
        raise ValueError(f"fixed-concurrency 必须是 1 到 workers 上限 {maximum} 之间的整数")


def _benchmark_config(config: CollectionConfig, fixed_rate: float | None) -> CollectionConfig:
    if fixed_rate is None:
        return config
    limit = float(config.request_rate_limit)
    if isinstance(fixed_rate, bool) or not math.isfinite(fixed_rate) or not 0.1 <= fixed_rate <= limit:
        raise ValueError(f"fixed-rate 必须是 0.1 到 rate-limit 上限 {limit:g} 之间的有限数值")
    return replace(config, strategy="paced", request_start_rate=float(fixed_rate))


class FixedConcurrency(AdaptiveConcurrency):
    """Benchmark-only constant limit, retaining the production request gate."""

    def __init__(self, max_concurrency: int, fixed_concurrency: int, **kwargs):
        _validate_fixed_concurrency(fixed_concurrency, max_concurrency)
        super().__init__(max_concurrency, **kwargs)
        self.current = fixed_concurrency
        self._trend = "fixed"

    def _finish_window_locked(self):
        # Continue reporting successful windows without evaluating a new level.
        self._previous_window_success = self._last_window_success
        self._last_window_success = self._window_success
        return None


def file_evidence(path: Path | None, *, destination: Path | None = None) -> dict:
    if path is None or not path.exists():
        return {"path": str(path or ""), "exists": False}
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    evidence = {"path": str(path), "exists": True, "bytes": size, "sha256": digest.hexdigest()}
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if file_evidence(destination)["sha256"] != evidence["sha256"]:
            raise RuntimeError(f"备份校验失败: {path}")
        evidence["backup"] = str(destination)
    return evidence


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def algorithm_source_hashes() -> dict[str, str | None]:
    """Capture only source identity, with repository-relative public names."""
    sources = ("data_get/collector.py", "data_get/concurrency_strategies.py", "data_get/request_pacing.py", "tools/benchmark_collection.py")
    return {name: file_evidence(PROJECT_ROOT / name).get("sha256") for name in sources}


class MeasuredAdapter:
    """Only record aggregate operation timings/statuses, never content or URLs."""

    def __init__(self, adapter: object):
        self.adapter = adapter
        self._lock = threading.Lock()
        self._operations: dict[str, dict] = {}

    def __getattr__(self, name: str):
        # Forward cooperative cooldown/stop hooks used by the real runner.
        return getattr(self.adapter, name)

    def _call(self, name: str, value):
        started = time.monotonic()
        successful = False
        status = "unknown"
        try:
            result = getattr(self.adapter, name)(value)
            successful = True
            status = "200"
            return result
        except BaseException as exc:
            status = str(getattr(exc, "status_code", None) or type(exc).__name__)
            raise
        finally:
            with self._lock:
                item = self._operations.setdefault(name, {"attempts": 0, "operationsReturned": 0, "seconds": 0.0, "status": Counter()})
                item["attempts"] += 1
                # An adapter return is not a runner-validated image success.
                item["operationsReturned"] += int(successful)
                item["seconds"] += time.monotonic() - started
                item["status"][status] += 1

    def discover_page(self, page: int):
        return self._call("discover_page", page)

    def fetch_detail(self, item):
        return self._call("fetch_detail", item)

    def fetch_thumbnail(self, url: str):
        return self._call("fetch_thumbnail", url)

    def close(self) -> None:
        close = getattr(self.adapter, "close", None)
        if close is not None:
            close()

    def snapshot(self) -> dict:
        with self._lock:
            return {name: {**item, "seconds": round(item["seconds"], 3), "status": dict(item["status"])} for name, item in self._operations.items()}


class ProgressRecorder:
    def __init__(self, path: Path, warmup: float, measured: MeasuredAdapter, *,
                 clock: Callable[[], float] = time.monotonic, fixed_concurrency: int | None = None,
                 paced: bool = False, fixed_rate: float | None = None):
        self.path = path
        self.warmup = warmup
        self.measured = measured
        self.clock = clock
        self._lock = threading.RLock()
        self.latest: dict = {}
        self.started: float | None = None
        self.baseline: dict | None = None
        self.active_end: dict | None = None
        self._csv_store = None
        self.fixed_concurrency = fixed_concurrency
        self._controller = None
        self._samples: list[dict] = []
        self.paced = paced
        self.fixed_rate = fixed_rate

    def bind_runner(self, runner: OnlineCollectionRunner) -> None:
        with self._lock:
            self._csv_store = runner.csv_store
            self._controller = getattr(runner, "concurrency", None)

    def update(self, payload: dict) -> None:
        with self._lock:
            self.latest = dict(payload)
            if self.fixed_concurrency is not None:
                self.latest.update(concurrencyStrategy="paced" if self.paced else "fixed", currentConcurrency=self.fixed_concurrency)

    def start(self) -> None:
        with self._lock:
            self.started = self.clock()
            self.sample()

    def _snapshot(self) -> dict:
        item = self.latest
        complete = int(item.get("complete", 0))
        controller_stats = getattr(self._controller, "stats", None)
        pacer = getattr(self._controller, "pacer", None)
        pacing_stats = getattr(pacer, "stats", None)
        pacing = pacing_stats() if callable(pacing_stats) else None
        return {
            "elapsedSeconds": round(self.clock() - self.started, 4),
            # Progress callbacks can lag a detail write by a long thumbnail
            # retry. Read the counter incremented immediately after CSV flush.
            "csvWrites": int(self._csv_store.write_count) if self._csv_store is not None else int(item.get("csvWrites", 0)),
            "csvWritesSource": "csv_store" if self._csv_store is not None else "progress-cache",
            "informationCompleted": int(item.get("informationCompleted", complete + int(item.get("missingImage", 0)))),
            "imageCompleted": int(item.get("imagesCompleted", complete + int(item.get("missingInfo", 0)))),
            "completed": complete,
            "fixedConcurrency": self.fixed_concurrency,
            "fixedRate": self.fixed_rate,
            "concurrencyStrategy": item.get("concurrencyStrategy"),
            **{name: item.get(name) for name in ("stage", "round", "discovered", "missingInfo", "missingImage", "missingBoth", "pagesCompleted", "pagesPending", "currentConcurrency", "windowSuccess", "lastWindowSuccess", "previousWindowSuccess", "windowElapsedSeconds", "throughputTrend")},
            "requestRate": pacing.get("current_rate") if pacing is not None else item.get("requestRate"),
            "requestRateLimit": pacing.get("max_rate") if pacing is not None else item.get("requestRateLimit"),
            "actualRequestStarts": pacing.get("started_requests") if pacing is not None else item.get("actualRequestStarts"),
            "activeRequests": pacing.get("active_requests") if pacing is not None else item.get("activeRequests"),
            "requests": self.measured.snapshot(),
            "controllerStats": controller_stats() if callable(controller_stats) else None,
            "requestPacing": self.paced,
            "pacingStats": pacing,
        }

    def sample(self, *, stop: bool = False) -> dict:
        with self._lock:
            row = self._snapshot()
            row["sampleIndex"] = len(self._samples)
            if self.baseline is None and row["elapsedSeconds"] >= self.warmup:
                self.baseline = dict(row)
            if stop and self.active_end is None:
                self.active_end = dict(row)
            row["afterWarmup"] = self.baseline is not None
            self._samples.append({"elapsedSeconds": row["elapsedSeconds"], "csvWrites": row["csvWrites"],
                                  "sampleIndex": row["sampleIndex"], "actualRequestStarts": row["actualRequestStarts"]})
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
            return row

    def throughput(self, final: dict) -> dict:
        end = self.active_end or final
        start = self.baseline
        seconds = max(0.0, end["elapsedSeconds"] - start["elapsedSeconds"]) if start else 0.0
        count = max(0, end["csvWrites"] - start["csvWrites"]) if start else 0
        return {"warmupRequestedSeconds": self.warmup, "baselineElapsedSeconds": start["elapsedSeconds"] if start else None,
                "endElapsedSeconds": end["elapsedSeconds"], "measuredSeconds": round(seconds, 4),
                "newInformation": count, "newInformationPerMinute": round(count * 60 / seconds, 3) if seconds else None,
                "metric": "CSV rows persisted after warmup; shutdown drain excluded"}

    def throughput_intervals(self, final: dict, *, window_seconds: float = 30) -> dict:
        """Show several real sampling intervals instead of ranking one phase.

        Boundaries use the nearest observed sample, with actual times exposed.
        No interpolation can manufacture writes between snapshots. Consecutive
        windows share a boundary, so periodic bursts are neither dropped nor
        counted twice merely because a timer wakes a few milliseconds late.
        """
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        end = self.active_end or final
        duration = end["elapsedSeconds"]
        rows = [row for row in self._samples if row["sampleIndex"] <= end["sampleIndex"]]
        if not rows:
            return {}

        def nearest(target: float) -> dict:
            if target <= 0:
                return rows[0]
            if target >= duration:
                return rows[-1]
            return min(rows, key=lambda row: abs(row["elapsedSeconds"] - target))

        def interval(lower: float, upper: float) -> dict:
            first, last = nearest(lower), nearest(upper)
            seconds = max(0.0, last["elapsedSeconds"] - first["elapsedSeconds"])
            count = max(0, last["csvWrites"] - first["csvWrites"])
            starts = (max(0, last["actualRequestStarts"] - first["actualRequestStarts"])
                      if first["actualRequestStarts"] is not None and last["actualRequestStarts"] is not None else None)
            return {"requestedStartSeconds": round(lower, 4), "requestedEndSeconds": round(upper, 4),
                    "actualStartSeconds": first["elapsedSeconds"], "actualEndSeconds": last["elapsedSeconds"],
                    "measuredSeconds": round(seconds, 4), "newInformation": count,
                    "newInformationPerMinute": round(count * 60 / seconds, 3) if seconds else None,
                    "actualRequestStarts": starts,
                    "actualStartsPerSecond": round(starts / seconds, 4) if starts is not None and seconds else None}

        windows = []
        lower = 0.0
        while lower < duration:
            upper = min(duration, lower + window_seconds)
            window = interval(lower, upper)
            # Timer drift can leave a tiny last nominal interval whose two
            # boundaries resolve to the same final observation. It has no
            # measured time or writes, so omit it from the window comparison.
            if window["measuredSeconds"] > 0:
                windows.append(window)
            lower = upper
        return {"wholeActiveRun": interval(0, duration), "secondHalf": interval(duration / 2, duration),
                "middleHalf": interval(duration / 4, duration * 3 / 4), "windows": windows,
                "nominalWindowSeconds": window_seconds, "shutdownExcluded": True,
                "boundaryMethod": "nearest observed CSV snapshot; actual sample times reported; no interpolation"}


@contextmanager
def cooperative_sigint(stop: Callable[[str], None]):
    """Keep signal handlers free of locks and defer cancellation to a thread.

    SIGINT can interrupt the main thread while a CSV or adapter lock is held.
    Even Event.set() takes a lock, so the handler only assigns a boolean. The
    control thread uses the same cutoff-before-stop flow as the duration timer.
    """
    enabled = threading.current_thread() is threading.main_thread()
    if not enabled:
        yield None
        return
    previous = signal.getsignal(signal.SIGINT)
    requested = False
    done = threading.Event()

    def mark_requested(_signum, _frame):
        nonlocal requested
        requested = True

    def dispatch():
        while not done.wait(0.02):
            if requested:
                stop("keyboard_interrupt")
                return
        # Cover a request arriving just before normal run completion.
        if requested:
            stop("keyboard_interrupt")

    control = threading.Thread(target=dispatch, name="collection-benchmark-control", daemon=True)
    signal.signal(signal.SIGINT, mark_requested)
    try:
        control.start()
        yield control
    finally:
        # All runner operations have unwound before joining. No CSV or adapter
        # lock is held by this scope while the control callback completes.
        signal.signal(signal.SIGINT, previous)
        done.set()
        if control.ident is not None:
            control.join()


def benchmark_plan(config: CollectionConfig, duration: float, warmup: float,
                   fixed_concurrency: int | None = None, fixed_rate: float | None = None) -> dict:
    config = _benchmark_config(config, fixed_rate)
    resolved = config.resolved()
    _validate_fixed_concurrency(fixed_concurrency, resolved.workers)
    paced = resolved.strategy == "paced"
    return {"mode": resolved.mode, "strategy": "fixed" if fixed_concurrency is not None and not paced else resolved.strategy,
            "fixedConcurrency": fixed_concurrency,
            "fixedRate": fixed_rate, "requestPacing": paced,
            "requestRateLimit": getattr(resolved, "request_rate_limit", None),
            "requestStartRate": getattr(resolved, "request_start_rate", None),
            "adaptivePacing": paced and fixed_rate is None,
            "reuseConnections": getattr(resolved, "reuse_connections", True), "workers": resolved.workers,
            "maxPages": resolved.max_pages, "durationSeconds": duration, "warmupSeconds": warmup,
            "outputCsv": str(resolved.output_csv), "imageDir": str(resolved.image_dir),
            "stateFile": str(resolved.state_file), "errorLog": str(resolved.error_log), "identity": resolved.identity,
            "resume": resolved.resume, "requestsEnabled": False}


def run_benchmark(config: CollectionConfig, *, report_dir: Path, duration: float = 360,
                  warmup: float = 30, sample_interval: float = 5, adapter: object | None = None,
                  fixed_concurrency: int | None = None, fixed_rate: float | None = None) -> dict:
    if duration <= 0 or warmup < 0 or sample_interval <= 0:
        raise ValueError("duration/sample_interval 必须为正数，warmup 不得为负数")
    config = _benchmark_config(config, fixed_rate)
    resolved = config.resolved()
    _validate_fixed_concurrency(fixed_concurrency, resolved.workers)
    if resolved.mode != "nh-online" or not resolved.resume:
        raise ValueError("基准测试仅支持 nh-online 安全续爬（resume=True）")
    source_hashes = algorithm_source_hashes()
    with collection_output_lock(resolved):
        report_dir = report_dir.resolve()
        report_dir.mkdir(parents=True, exist_ok=False)
        evidence = {
            "csv": file_evidence(resolved.output_csv, destination=report_dir / "before" / "output.csv"),
            "state": file_evidence(resolved.state_file, destination=report_dir / "before" / "state.jsonl"),
            "errors": file_evidence(resolved.error_log),
        }
        atomic_json(report_dir / "before.json", evidence)
        measured = MeasuredAdapter(adapter if adapter is not None else NHAdapter(resolved))
        recorder = ProgressRecorder(report_dir / "telemetry.jsonl", warmup, measured,
                                    fixed_concurrency=fixed_concurrency, paced=resolved.strategy == "paced", fixed_rate=fixed_rate)
        stop_event = threading.Event()
        sample_done = threading.Event()
        stopping: dict = {}
        stop_lock = threading.RLock()

        def request_stop(reason: str) -> None:
            with stop_lock:
                if stopping:
                    return
                stopping.update(reason=reason, at=time.monotonic())
            try:
                # Freeze the active interval before workers can react to stop
                # and finish any in-flight detail writes during shutdown.
                if recorder.started is not None:
                    recorder.sample(stop=True)
            finally:
                stop_event.set()

        def sample_loop() -> None:
            while not sample_done.wait(sample_interval):
                recorder.sample()

        started_init = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        runner = None
        timer = None
        sampler = None
        result = None
        error_type = None
        try:
            with cooperative_sigint(request_stop):
                runner = OnlineCollectionRunner(resolved, measured, stop_event=stop_event, progress_callback=recorder.update)
                if resolved.strategy == "paced" and (fixed_concurrency is not None or fixed_rate is not None):
                    from data_get.concurrency_strategies import create_controller

                    runner.concurrency = create_controller(
                        "paced", resolved.workers, request_rate_limit=resolved.request_rate_limit,
                        request_start_rate=fixed_rate if fixed_rate is not None else resolved.request_start_rate,
                        adaptive_pacing=fixed_rate is None, request_concurrency=fixed_concurrency,
                    )
                    # __getattr__ forwards to the real adapter; assigning an
                    # attribute to this measurement wrapper would not bind it.
                    measured.set_request_pacer(runner.concurrency.pacer)
                elif fixed_concurrency is not None:
                    # Construction only indexes local state. Install the fixed
                    # gate before run() can submit the first network request.
                    runner.concurrency = FixedConcurrency(resolved.workers, fixed_concurrency)
                recorder.bind_runner(runner)
                recorder.start()
                timer = threading.Timer(duration, request_stop, args=("duration",))
                timer.daemon = True
                timer.start()
                sampler = threading.Thread(target=sample_loop, name="collection-benchmark-telemetry", daemon=True)
                sampler.start()
                try:
                    result = runner.run()
                except KeyboardInterrupt:
                    request_stop("keyboard_interrupt")
                    raise
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            if timer is not None:
                timer.cancel()
            sample_done.set()
            if sampler is not None:
                sampler.join()
            measured.close()
            finished = time.monotonic()
            final = recorder.sample() if recorder.started is not None else {}
            controller = getattr(runner, "concurrency", None)
            controller_stats = getattr(controller, "stats", None)
            payload = {
                "schemaVersion": 3, "metricVersion": 3, "startedAt": started_at,
                "sourceSha256": source_hashes,
                "controllerStats": controller_stats() if callable(controller_stats) else None,
                "config": {**benchmark_plan(config, duration, warmup, fixed_concurrency, fixed_rate), "requestsEnabled": True},
                "before": evidence, "initializationSeconds": round((recorder.started or finished) - started_init, 4),
                "elapsedSeconds": round(finished - recorder.started, 4) if recorder.started else 0,
                "stopReason": stopping.get("reason", "error" if error_type else "completed"),
                "stopDrainSeconds": round(finished - stopping["at"], 4) if stopping else 0,
                "expectedInterruption": stopping.get("reason") == "duration" and error_type is None,
                "collection": result.as_dict() if result else None, "errorType": error_type,
                "final": final, "stableThroughput": recorder.throughput(final) if final else None,
                "throughputIntervals": recorder.throughput_intervals(final) if final else None,
                "requests": measured.snapshot(),
                "requestMetric": "attempts counts adapter calls including pacing waits cancelled before HTTP; operationsReturned is before runner validation, not valid images; pacingStats.started_requests counts actual HTTP starts",
            }
            atomic_json(report_dir / "summary.json", payload)
        payload["exitCode"] = 0 if payload["expectedInterruption"] else (result.exit_code if result else 1)
        atomic_json(report_dir / "summary.json", payload)
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="显式开启实网、安全续爬测试；默认仅展示计划")
    parser.add_argument("--strategy", choices=("stable", "adaptive-step", "bandit", "paced"), default="stable")
    parser.add_argument("--fixed-rate", type=float, help="固定实际HTTP请求发起频率（次/秒）；自动选paced并关闭频率自适应")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="paced请求发起频率上限（次/秒，0.1..100）")
    parser.add_argument("--start-rate", type=float, help="paced起始请求频率（次/秒，不超过rate-limit）")
    parser.add_argument("--fixed-concurrency", type=int,
                        help="测试专用固定请求并发数（1..workers）；指定后覆盖本次策略，不修改生产配置")
    parser.add_argument("--duration", type=float, default=360)
    parser.add_argument("--warmup", type=float, default=30)
    parser.add_argument("--sample-interval", type=float, default=5)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--base-url", default="https://nhentai.net")
    parser.add_argument("--start-url", default="https://nhentai.net/language/chinese/?sort=date")
    parser.add_argument("--max-pages", type=int, default=700)
    parser.add_argument("--output-csv", default="data/gallery_info_origin/NH_info_chinese.csv")
    parser.add_argument("--image-dir", default="onlineimgtmp")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--request-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retry-backoff", type=float, default=5)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0)
    parser.add_argument("--state-file")
    parser.add_argument("--error-log")
    parser.add_argument("--proxy", help="使用项目代理配置时留空；该值不写入报告")
    parser.add_argument("--no-reuse-connections", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.warmup < 0 or args.sample_interval <= 0:
        parser.error("duration/sample-interval 必须为正数，warmup 不得为负数")
    fields = ("base_url", "start_url", "max_pages", "output_csv", "image_dir", "workers", "request_attempts",
              "timeout", "retry_backoff", "max_rounds", "interval", "state_file", "error_log", "proxy", "strategy")
    config = CollectionConfig(mode="nh-online", resume=True, reuse_connections=not args.no_reuse_connections,
                              request_rate_limit=args.rate_limit, request_start_rate=args.start_rate,
                              **{name: getattr(args, name) for name in fields})
    try:
        plan = benchmark_plan(config, args.duration, args.warmup, args.fixed_concurrency, args.fixed_rate)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = f"fixed-{args.fixed_concurrency}" if args.fixed_concurrency is not None else args.strategy
    if args.fixed_rate is not None:
        label = f"paced-fixed-{args.fixed_rate:g}-concurrency-{args.fixed_concurrency or args.workers}"
    report_dir = args.report_dir or PROJECT_ROOT / "logs" / "collection_benchmarks" / f"{stamp}-{label}-{uuid.uuid4().hex[:6]}"
    try:
        summary = run_benchmark(config, report_dir=report_dir, duration=args.duration,
                                warmup=args.warmup, sample_interval=args.sample_interval,
                                fixed_concurrency=args.fixed_concurrency, fixed_rate=args.fixed_rate)
    except KeyboardInterrupt:
        # Before the runner starts (for example while backing up), source data
        # has not been touched. run_benchmark's output-lock context unwinds.
        print("基准测试已中止；已完成的数据和备份保留。", file=sys.stderr)
        return 130
    except (CollectionOutputBusyError, FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"reportDir": str(report_dir.resolve()), "stopReason": summary["stopReason"],
                      "stableThroughput": summary["stableThroughput"], "exitCode": summary["exitCode"]},
                     ensure_ascii=False, indent=2))
    return int(summary["exitCode"])


if __name__ == "__main__":
    raise SystemExit(main())
