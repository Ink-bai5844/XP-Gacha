from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def open_local_folder(target: Path, *, allowed: bool) -> bool:
    """Open a directory with the host's file manager when explicitly enabled."""
    if not allowed:
        return False
    if os.name == "nt":
        os.startfile(str(target))
        return True
    if sys.platform == "darwin":
        subprocess.run(["/usr/bin/open", str(target)], check=True)
        return True
    return False
