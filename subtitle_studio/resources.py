from __future__ import annotations

import sys
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / relative_path
    return package_root() / relative_path
