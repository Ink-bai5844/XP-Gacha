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

    def public(self, after: int = 0) -> dict:
        return {
            "id": self.id,
            "scriptId": self.script_id,
            "status": self.status,
            "lines": self.lines[after:],
            "lineCount": len(self.lines),
            "createdAt": self.created_at,
            "finishedAt": self.finished_at,
            "returnCode": self.return_code,
        }


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
            job = Job(id=uuid.uuid4().hex, script_id=script_id)
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, parameters), daemon=True).start()
        return job.public()

    def get(self, job_id: str, after: int = 0) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            return job.public(after=max(0, after))

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            process = job.process
            if job.status not in {"queued", "running"}:
                return job.public()
            job.status = "cancelling"
            job.lines.append("[CANCEL] 已请求中止任务")
        if process:
            _terminate_process_tree(process)
        return job.public()

    def _run(self, job: Job, parameters: dict) -> None:
        encoded = base64.urlsafe_b64encode(
            json.dumps(parameters, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        command = [sys.executable, "-m", "server.job_tasks", job.script_id, encoded]
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
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
            cancel_before_running = False
            with self._lock:
                job.process = process
                if job.status == "cancelling":
                    cancel_before_running = True
                else:
                    job.status = "running"
                    job.lines.append(f"> {' '.join(command[:4])} <parameters>")
            if cancel_before_running:
                _terminate_process_tree(process)
            if process.stdout:
                for line in process.stdout:
                    with self._lock:
                        job.lines.append(line.rstrip())
            return_code = process.wait()
            with self._lock:
                job.return_code = return_code
                if job.status == "cancelling":
                    job.status = "cancelled"
                else:
                    job.status = "completed" if return_code == 0 else "failed"
        except Exception as exc:
            with self._lock:
                if job.status == "cancelling":
                    job.status = "cancelled"
                else:
                    job.status = "failed"
                    job.lines.append(f"[ERROR] {exc}")
                    job.return_code = 1
        finally:
            with self._lock:
                job.finished_at = time.time()
                completed = job.status == "completed"
            if completed and self._on_complete:
                try:
                    self._on_complete()
                except Exception as exc:
                    with self._lock:
                        job.lines.append(f"[CACHE] 任务已完成，但缓存刷新失败：{exc}")
