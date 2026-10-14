"""Advisory, process-scoped locks for collection output files.

Keep lock files in place: deleting a live lock file would allow a second process
to lock a new inode while the first process still owns the original one.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class CollectionOutputBusyError(RuntimeError):
    """Another cooperating collector already owns an output target."""


def collection_lock_paths(config: object) -> list[Path]:
    paths = {}
    for name in ("output_csv", "state_file", "output_dir"):
        value = getattr(config, name, None)
        if value is None:
            continue
        target = Path(value).resolve()
        key = os.path.normcase(str(target))
        paths[key] = target.with_name(target.name + ".collection.lock")
    return [paths[key] for key in sorted(paths)]


@contextmanager
def collection_output_lock(config: object) -> Iterator[None]:
    """Lock all distinct targets, non-blocking and in a fixed global order."""
    handles = []
    try:
        for path in collection_lock_paths(config):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+b")
            try:
                if os.name == "nt":
                    import msvcrt

                    # Windows byte-range locking requires an existing byte.
                    if path.stat().st_size == 0:
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                handle.close()
                raise CollectionOutputBusyError(
                    f"采集输出正在被另一个任务使用，请先安全停止该任务: {path}"
                ) from exc
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
