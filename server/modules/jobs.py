from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOB_CONTROL_DIR = PROJECT_ROOT / "run" / "job-control"
COLLECTOR_PROGRESS_PREFIX = "[COLLECTOR_PROGRESS] "
JOB_LOG_LINE_LIMIT = 500
SCRIPT_IDS = {
    "addname", "add-id", "add-lang", "clean-date", "title-words", "tag-set", "map-add-name",
    "db-sync", "db-rebuild", "db-optimize", "title-translate", "b64", "text-vector", "clip-vector",
    "cache-delete", "prefix-rename", "merge-b64", "clean-title-jsonl", "delete-gallery-rows",
    "clear-title-translation", "export-title-translations", "collection-nh-online", "collection-jm-online",
    "collection-nh-local-info", "collection-nh-local-images",
}


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Best-effort termination for a job process and any descendants."""
    try:
        if process.poll() is not None:
            return
    except OSError:
        return

    terminated = False
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            terminated = result.returncode == 0
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            terminated = True
    except (OSError, subprocess.SubprocessError):
        terminated = False

    if not terminated:
        try:
            process.terminate()
        except (AttributeError, OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except (AttributeError, OSError, subprocess.SubprocessError):
                pass


@dataclass
class Job:
    id: str
    script_id: str
    status: str = "queued"
    lines: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    return_code: int | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)
    progress: dict[str, object] = field(default_factory=dict)
    cancel_file: Path | None = field(default=None, repr=False)
    force_cancel_requested: bool = field(default=False, repr=False)
    _line_offset: int = field(default=0, init=False, repr=False)
    _line_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._line_count = len(self.lines)
        if len(self.lines) > JOB_LOG_LINE_LIMIT:
            overflow = len(self.lines) - JOB_LOG_LINE_LIMIT
            del self.lines[:overflow]
            self._line_offset = overflow

    def append_line(self, line: str) -> None:
        self.lines.append(line)
        self._line_count += 1
        overflow = len(self.lines) - JOB_LOG_LINE_LIMIT
        if overflow > 0:
            del self.lines[:overflow]
            self._line_offset += overflow

    def public(self, after: int = 0) -> dict:
        start = max(max(0, after), self._line_offset) - self._line_offset
        return {
            "id": self.id,
            "scriptId": self.script_id,
            "status": self.status,
            "lines": self.lines[start:],
            "lineCount": self._line_count,
            "createdAt": self.created_at,
            "finishedAt": self.finished_at,
            "returnCode": self.return_code,
            "progress": dict(self.progress),
        }


def _collector_progress(line: str) -> dict[str, object] | None:
    """Decode a collector progress line without exposing it in the job log."""
    if not line.startswith(COLLECTOR_PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(COLLECTOR_PROGRESS_PREFIX):])
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _request_collection_cancel(path: Path) -> None:
    """Atomically make the per-job cancellation signal visible to the child."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _cleanup_collection_cancel(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # The UUID-named file is harmless if cleanup is temporarily blocked.
        pass


class JobsModule:
    def __init__(self, on_complete=None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._on_complete = on_complete

    def scripts(self) -> dict:
        return {"scripts": [{"id": script_id, "available": True} for script_id in sorted(SCRIPT_IDS)]}

    def start(self, script_id: str, parameters: dict) -> dict:
        if script_id not in SCRIPT_IDS:
            raise KeyError(script_id)
        with self._lock:
            if any(job.status in {"queued", "running", "cancelling"} for job in self._jobs.values()):
                raise RuntimeError("已有数据处理任务正在运行")
            job_id = uuid.uuid4().hex
            job = Job(
                id=job_id,
                script_id=script_id,
                cancel_file=(JOB_CONTROL_DIR / f"{job_id}.cancel") if script_id.startswith("collection-") else None,
            )
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, parameters), daemon=True).start()
        with self._lock:
            return job.public()

    def get(self, job_id: str, after: int = 0) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            return job.public(after=max(0, after))

    def cancel(self, job_id: str, *, force: bool = False, after: int = 0) -> dict:
        after = max(0, after)
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            process = job.process
            is_collection = job.cancel_file is not None
            if job.status == "cancelling" and not (force and is_collection):
                return job.public(after=after)
            if job.status not in {"queued", "running", "cancelling"}:
                return job.public(after=after)
            if is_collection and not force:
                assert job.cancel_file is not None
                # Create the sentinel while holding the same lock used by
                # _run's final cleanup, otherwise a just-finished child could
                # clean first and leave a late sentinel behind forever.
                try:
                    _request_collection_cancel(job.cancel_file)
                except OSError as exc:
                    job.append_line(f"[CANCEL] 无法创建安全中止信号：{exc}")
                    return job.public(after=after)
            job.status = "cancelling"
            if force and is_collection:
                job.force_cancel_requested = True
                job.append_line("[CANCEL] 已请求强制终止采集任务")
            elif is_collection:
                job.append_line("[CANCEL] 已请求安全中止，正在等待当前请求结束并写回数据")
            else:
                job.append_line("[CANCEL] 已请求中止任务")
        if (not is_collection or force) and process:
            _terminate_process_tree(process)
        with self._lock:
            return job.public(after=after)

    def _run(self, job: Job, parameters: dict) -> None:
        encoded = base64.urlsafe_b64encode(
            json.dumps(parameters, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        command = [sys.executable, "-m", "server.job_tasks", job.script_id, encoded]
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        if job.cancel_file is not None:
            env["XP_GACHA_JOB_CANCEL_FILE"] = str(job.cancel_file)
        return_code: int | None = None
        run_error: Exception | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            terminate_before_running = False
            with self._lock:
                job.process = process
                if job.status == "cancelling":
                    terminate_before_running = job.cancel_file is None or job.force_cancel_requested
                else:
                    job.status = "running"
                    job.append_line(f"> {' '.join(command[:4])} <parameters>")
            if terminate_before_running:
                _terminate_process_tree(process)
            if process.stdout:
                for line in process.stdout:
                    output = line.rstrip()
                    with self._lock:
                        progress = _collector_progress(output)
                        if progress is None:
                            job.append_line(output)
                        else:
                            job.progress.update(progress)
            return_code = process.wait()
        except Exception as exc:
            run_error = exc
        finally:
            with self._lock:
                if return_code is not None:
                    job.return_code = return_code
                elif run_error is not None:
                    job.return_code = 1

                if run_error is not None:
                    job.status = "failed"
                    job.append_line(f"[ERROR] {run_error}")
                elif job.status == "cancelling":
                    if job.cancel_file is not None:
                        if return_code == 0:
                            # The collection won the race with the stop signal.
                            job.status = "completed"
                        elif return_code == 130:
                            job.status = "cancelled"
                        else:
                            job.status = "failed"
                            job.append_line(
                                f"[ERROR] 采集器在安全中止期间异常退出（代码 {return_code}），"
                                "CSV 或断点可能未完整写回"
                            )
                    else:
                        job.status = "cancelled"
                elif return_code is not None:
                    job.status = "completed" if return_code == 0 else "failed"
                else:
                    # A BaseException escaping the worker still must not leave
                    # a permanently running job that can create a late signal.
                    job.status = "failed"
                    job.return_code = 1

                # Terminal status and signal cleanup are one atomic state
                # transition with respect to cancel().  Once this lock is
                # released, cancel() sees a terminal job and cannot touch a
                # fresh sentinel after cleanup.
                _cleanup_collection_cancel(job.cancel_file)
                job.finished_at = time.time()
                completed = job.status == "completed"
            if completed and self._on_complete:
                try:
                    self._on_complete()
                except Exception as exc:
                    with self._lock:
                        job.append_line(f"[CACHE] 任务已完成，但缓存刷新失败：{exc}")
