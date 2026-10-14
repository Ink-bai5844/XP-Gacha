"""Read-only CSV/checkpoint preservation audit; never print collected content.

An active writer can leave an incomplete final record in a read snapshot. Use
--allow-partial-tail for a running collection; only that final record is ignored.
The final stopped-file audit should omit it. Inputs are never opened for writing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


def _read_snapshot(path: Path, allow_partial_tail: bool) -> tuple[bytes, str, bool]:
    raw = path.read_bytes()
    try:
        return raw, raw.decode("utf-8").removeprefix("\ufeff"), False
    except UnicodeDecodeError as exc:
        if allow_partial_tail and exc.reason == "unexpected end of data" and exc.end == len(raw):
            return raw, raw[:exc.start].decode("utf-8").removeprefix("\ufeff"), True
        raise ValueError("invalid_utf8") from None


def csv_snapshot(path: Path, *, allow_partial_tail: bool = False) -> dict:
    raw, text, ignored_tail = _read_snapshot(path, allow_partial_tail)
    stream = io.StringIO(text, newline="")
    reader = csv.reader(stream, strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error):
        raise ValueError("missing_csv_header") from None
    if "ID" not in header or len(set(header)) != len(header):
        raise ValueError("invalid_csv_header")
    id_column = header.index("ID")
    rows: dict[str, tuple[str, ...]] = {}
    counts: Counter[str] = Counter()
    while True:
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error:
            if allow_partial_tail and stream.tell() == len(text) and not text.endswith(("\n", "\r")):
                ignored_tail = True
                break
            raise ValueError(f"invalid_csv_record_at_line_{reader.line_num}") from None
        if not row:
            continue
        if len(row) != len(header) or not row[id_column].strip():
            if allow_partial_tail and stream.tell() == len(text) and not text.endswith(("\n", "\r")):
                ignored_tail = True
                break
            raise ValueError(f"invalid_csv_record_at_line_{reader.line_num}")
        # The collector flushes newline-terminated records. A nonterminated
        # final row may still be mid-write even if it happens to parse today.
        if allow_partial_tail and stream.tell() == len(text) and not text.endswith(("\n", "\r")):
            ignored_tail = True
            break
        gallery_id = row[id_column].strip()
        counts[gallery_id] += 1
        rows[gallery_id] = tuple(row)
    return {"raw": raw, "header": header, "rows": rows, "counts": counts,
            "utf8Bom": raw.startswith(b"\xef\xbb\xbf"), "ignoredPartialTail": ignored_tail,
            "sha256": hashlib.sha256(raw).hexdigest()}


def state_snapshot(path: Path, *, allow_partial_tail: bool = False) -> dict:
    raw, text, ignored_tail = _read_snapshot(path, allow_partial_tail)
    state = {"identity": "", "tasks": {}, "completedPages": set(), "terminalPages": set()}
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if allow_partial_tail and index == len(lines) and not line.endswith(("\n", "\r")):
                ignored_tail = True
                break
            raise ValueError(f"invalid_state_record_at_line_{index}") from None
        if not isinstance(event, dict):
            raise ValueError(f"invalid_state_record_at_line_{index}")
        kind = event.get("event")
        if kind == "run_start":
            state = {"identity": str(event.get("identity") or ""), "tasks": {},
                     "completedPages": set(), "terminalPages": set()}
        elif kind == "run_resume":
            identity = str(event.get("identity") or "")
            if identity and state["identity"] and identity != state["identity"]:
                raise ValueError("state_resume_identity_mismatch")
        elif kind in {"page_complete", "page_terminal"}:
            try:
                state["completedPages" if kind == "page_complete" else "terminalPages"].add(int(event["page"]))
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"invalid_state_page_at_line_{index}") from None
        elif kind == "task_state":
            item = event.get("item")
            if not isinstance(item, dict) or not item.get("id"):
                raise ValueError(f"invalid_state_task_at_line_{index}")
            state["tasks"][str(item["id"])] = (bool(event.get("info_ok")), bool(event.get("thumb_ok")))
        elif kind == "task_delete":
            state["tasks"].pop(str(event.get("id") or ""), None)
    state.update(ignoredPartialTail=ignored_tail, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return state


def audit_collection(*, before_csv: Path, after_csv: Path, before_state: Path,
                     after_state: Path, allow_partial_tail: bool = False, anomaly_limit: int = 20) -> dict:
    old_csv = csv_snapshot(before_csv)
    old_state = state_snapshot(before_state)
    # Read state first: a checkpoint can only mark info_ok after the CSV flush.
    # Reading CSV last avoids flagging new checkpointed rows during append-only
    # activity merely because they were written between the two snapshots.
    new_state = state_snapshot(after_state, allow_partial_tail=allow_partial_tail)
    new_csv = csv_snapshot(after_csv, allow_partial_tail=allow_partial_tail)
    old_rows, new_rows = old_csv["rows"], new_csv["rows"]
    old_tasks, new_tasks = old_state["tasks"], new_state["tasks"]
    anomalies = {
        "missingCsvIds": sorted(old_rows.keys() - new_rows.keys()),
        "changedCsvIds": sorted(key for key in old_rows.keys() & new_rows.keys() if old_rows[key] != new_rows[key]),
        "existingDuplicateCsvIds": sorted(key for key, count in old_csv["counts"].items() if count > 1),
        "newDuplicateCsvIds": sorted(key for key, count in new_csv["counts"].items()
                                      if count > 1 and count > old_csv["counts"].get(key, 0)),
        "missingTaskIds": sorted(old_tasks.keys() - new_tasks.keys()),
        "lostInformationSuccessIds": sorted(key for key, flags in old_tasks.items()
                                             if flags[0] and not new_tasks.get(key, (False, False))[0]),
        "lostImageSuccessIds": sorted(key for key, flags in old_tasks.items()
                                       if flags[1] and not new_tasks.get(key, (False, False))[1]),
        "lostCompletedPages": sorted(old_state["completedPages"] - new_state["completedPages"]),
        "lostTerminalPages": sorted(old_state["terminalPages"] - new_state["terminalPages"]),
        "checkpointInformationMissingCsvIds": sorted(key for key, flags in new_tasks.items() if flags[0] and key not in new_rows),
    }
    checks = {
        "headerUnchanged": old_csv["header"] == new_csv["header"],
        "utf8BomPreserved": old_csv["utf8Bom"] == new_csv["utf8Bom"],
        "checkpointIdentityUnchanged": old_state["identity"] == new_state["identity"],
        "originalBytePrefixPreserved": new_csv["raw"].startswith(old_csv["raw"]),
    }
    def counts(csv_data: dict, state: dict) -> dict:
        tasks = state["tasks"]
        return {"csvRows": sum(csv_data["counts"].values()), "uniqueCsvIds": len(csv_data["rows"]),
                "csvBytes": len(csv_data["raw"]), "utf8Bom": csv_data["utf8Bom"],
                "csvSha256": csv_data["sha256"], "stateSha256": state["sha256"],
                "tasks": len(tasks), "completedPages": len(state["completedPages"]),
                "informationSuccess": sum(flags[0] for flags in tasks.values()),
                "imageSuccess": sum(flags[1] for flags in tasks.values()),
                "complete": sum(all(flags) for flags in tasks.values()),
                "ignoredCsvPartialTail": csv_data["ignoredPartialTail"],
                "ignoredStatePartialTail": state["ignoredPartialTail"]}
    # A rewrite can preserve every CSV field while changing byte formatting;
    # report byte-prefix equality separately from semantic preservation.
    ok = all(value for key, value in checks.items() if key != "originalBytePrefixPreserved") and not any(anomalies.values())
    return {"ok": ok, "snapshotWhileRunning": allow_partial_tail, "checks": checks,
            "before": counts(old_csv, old_state), "after": counts(new_csv, new_state),
            "newCsvIds": len(new_rows.keys() - old_rows.keys()),
            "anomalies": {name: {"count": len(values), "ids": values[:anomaly_limit]}
                          for name, values in anomalies.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before-csv", "after-csv", "before-state", "after-state"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--allow-partial-tail", action="store_true")
    parser.add_argument("--anomaly-limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.anomaly_limit < 0:
        parser.error("anomaly-limit must be nonnegative")
    try:
        report = audit_collection(**vars(args))
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "errorType": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
