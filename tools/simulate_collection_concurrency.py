"""Offline synthetic controller comparison; this is NOT an NH benchmark.

Run from a checkout with its normal Python dependencies installed::

    python tools/simulate_collection_concurrency.py --include-quota

The harness calls the real controllers' acquire/release/record_success/
record_failure methods using a fake clock.  It never starts a collector, sends
network requests, or reads/writes a checkpoint, CSV, or image.  A request limit
is converted into a synthetic completion rate; the mean concurrency reported
is the time-weighted *configured limit*, not an assertion about actual sockets.

Poisson completions and a separate startup-burst random stream let paired runs
verify that false startup throughput does not affect later controller choices.
The three required curves are deliberately simple and cannot establish which
strategy is fastest on a real platform.  The optional fourth curve adds a
periodic success quota to expose short-window phase sensitivity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
import threading


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_get.concurrency_strategies import create_controller


STRATEGIES = ("stable", "adaptive-step", "bandit")
SCENARIOS = ("rising_then_plateau", "excess_concurrency_penalty", "changing_optimum")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def poisson(rng: random.Random, mean: float) -> int:
    """Small-mean Poisson variate, without a third-party numeric dependency."""

    threshold = math.exp(-mean)
    product = 1.0
    count = 0
    while product > threshold:
        product *= rng.random()
        count += 1
    return max(0, count - 1)


def success_rate(scenario: str, concurrency: int, second: int, duration: int) -> float:
    if scenario in ("rising_then_plateau", "periodic_quota_plateau"):
        # All limits >= 8 provide the same useful throughput.
        return 4.0 * min(concurrency / 8.0, 1.0)
    if scenario == "excess_concurrency_penalty":
        optimum = 8.0
    elif scenario == "changing_optimum":
        phase = min(2, second * 3 // duration)
        optimum = (16.0, 4.0, 10.0)[phase]
    else:
        raise ValueError(scenario)
    ratio = concurrency / optimum
    return 4.0 * (ratio if ratio <= 1.0 else ratio ** -1.4)


def simulate(
    strategy: str,
    scenario: str,
    *,
    duration: int = 1800,
    cap: int = 30,
    seed: int = 1,
    startup_burst: bool = True,
) -> dict:
    clock = FakeClock()
    controller = create_controller(strategy, cap, time_fn=clock)
    controller.acquire(threading.Event())
    controller.release()
    rng = random.Random(seed)
    startup_rng = random.Random(seed + 1_000_000)
    count_total = count_steady = startup_extra = 0
    area = 0
    cap_violations = 0
    levels: dict[int, int] = {}
    phases = [{"successes": 0, "limit_seconds": 0, "seconds": 0} for _ in range(3)]
    decisions = []
    quota_remaining = 20
    expected_steady = oracle_expected_steady = 0.0

    def record_decision(decision) -> None:
        if decision is not None and decision.stable_score is not None:
            decisions.append({
                "second": round(clock.now, 3),
                "from": decision.from_concurrency,
                "to": decision.to_concurrency,
                "reason": decision.reason,
                "score": round(decision.stable_score, 6),
            })

    for second in range(duration):
        clock.now = float(second)
        # A zero-score clock tick represents request completions that provide
        # no useful new item.  Its frequency is fixed and is never a score input.
        record_decision(controller.record_failure())
        limit = controller.current
        cap_violations += int(not 1 <= limit <= cap)
        area += limit
        levels[limit] = levels.get(limit, 0) + 1
        rate = success_rate(scenario, limit, second, duration)
        completions = poisson(rng, rate)
        if scenario == "periodic_quota_plateau":
            if second % 65 == 0:
                quota_remaining = 20
            completions = min(completions, quota_remaining)
            quota_remaining -= completions
        if second >= 30:
            count_steady += completions
            expected_steady += rate
            oracle_expected_steady += max(
                success_rate(scenario, candidate, second, duration)
                for candidate in range(1, cap + 1)
            )
        if startup_burst and second < 5:
            extra = poisson(startup_rng, 40.0)
            completions += extra
            startup_extra += extra
        clock.now = second + 0.5
        for _ in range(completions):
            record_decision(controller.record_success())
        count_total += completions
        phase = min(2, second * 3 // duration)
        phases[phase]["successes"] += completions
        phases[phase]["limit_seconds"] += limit
        phases[phase]["seconds"] += 1

    clock.now = float(duration)
    record_decision(controller.record_failure())
    moves = [row for row in decisions if row["from"] != row["to"]]
    result = {
        "scenario": scenario,
        "strategy": strategy,
        "seed": seed,
        "duration_seconds": duration,
        "total_successes": count_total,
        "steady_successes_after_30s": count_steady,
        "steady_successes_per_minute": round(count_steady * 60 / (duration - 30), 3),
        "startup_extra_successes": startup_extra,
        "mean_configured_concurrency": round(area / duration, 3),
        "seconds_per_concurrency": dict(sorted(levels.items())),
        "cap_violations": cap_violations,
        "moves": len(moves),
        "first_move": moves[0] if moves else None,
        "last_concurrency": controller.current,
        "phases": [
            {"phase": index + 1, "successes": phase["successes"],
             "mean_configured_concurrency": round(phase["limit_seconds"] / phase["seconds"], 3)}
            for index, phase in enumerate(phases)
        ],
        "decision_trace": decisions,
    }
    if scenario != "periodic_quota_plateau":
        result["expected_steady_efficiency_vs_oracle_pct"] = round(
            100 * expected_steady / oracle_expected_steady, 3,
        )
    return result


def summarize(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return round(statistics.mean(row[key] for row in rows), 3)
    result = {
        "scenario": rows[0]["scenario"],
        "strategy": rows[0]["strategy"],
        "runs": len(rows),
        "mean_total_successes": mean("total_successes"),
        "mean_steady_successes_after_30s": mean("steady_successes_after_30s"),
        "steady_successes_min_max": [min(row["steady_successes_after_30s"] for row in rows),
                                      max(row["steady_successes_after_30s"] for row in rows)],
        "mean_steady_successes_per_minute": mean("steady_successes_per_minute"),
        "mean_configured_concurrency": mean("mean_configured_concurrency"),
        "mean_moves": mean("moves"),
        "cap_violations": sum(row["cap_violations"] for row in rows),
        "startup_invariance_passed": all(row["startup_invariance_passed"] for row in rows),
        "phase_mean_successes": [round(statistics.mean(row["phases"][phase]["successes"]
                                                       for row in rows), 3) for phase in range(3)],
    }
    if "expected_steady_efficiency_vs_oracle_pct" in rows[0]:
        result["mean_expected_steady_efficiency_vs_oracle_pct"] = mean(
            "expected_steady_efficiency_vs_oracle_pct"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--cap", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--include-quota", action="store_true")
    parser.add_argument("--details", action="store_true", help="Include per-seed measurements and decisions")
    args = parser.parse_args()
    if args.duration <= 30 or args.cap < 1 or args.seeds < 1:
        parser.error("duration must exceed 30; cap and seeds must be positive")
    scenarios = SCENARIOS + (("periodic_quota_plateau",) if args.include_quota else ())
    rows = []
    summaries = []
    for scenario in scenarios:
        for strategy in STRATEGIES:
            group = []
            for seed in range(1, args.seeds + 1):
                row = simulate(strategy, scenario, duration=args.duration, cap=args.cap, seed=seed)
                control = simulate(strategy, scenario, duration=args.duration, cap=args.cap,
                                   seed=seed, startup_burst=False)
                row["startup_invariance_passed"] = (
                    row["decision_trace"] == control["decision_trace"]
                    and row["steady_successes_after_30s"] == control["steady_successes_after_30s"]
                )
                group.append(row)
                rows.append(row)
            summaries.append(summarize(group))
    report = {
        "synthetic_only_not_nh_measurement": True,
        "network_requests": 0,
        "collection_data_files_read_or_written": 0,
        "duration_seconds_per_run": args.duration,
        "cap": args.cap,
        "seeds": args.seeds,
        "startup_burst": "extra mean 40 successes/sec for the first 5 seconds; independent random stream",
        "configured_concurrency_mean_is_not_socket_count": True,
        "summaries": summaries,
    }
    if args.details:
        report["runs"] = rows
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(any(row["cap_violations"] or not row["startup_invariance_passed"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
