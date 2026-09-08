from __future__ import annotations
from pathlib import Path
def ensure_dirs(*dirs):
    for d in dirs: Path(d).mkdir(parents=True, exist_ok=True)
