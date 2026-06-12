"""Minimal .env loader (avoids a python-dotenv dependency)."""
import os
from pathlib import Path


def load_env(path: Path) -> None:
    """Read KEY=VALUE lines; real environment variables take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
