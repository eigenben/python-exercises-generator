from __future__ import annotations

from pathlib import Path
from typing import Optional


_PROJECT_ROOT_CACHE: Optional[Path] = None


def project_root(start: Optional[Path] = None) -> Path:
    global _PROJECT_ROOT_CACHE
    if _PROJECT_ROOT_CACHE is not None:
        return _PROJECT_ROOT_CACHE

    start = start or Path.cwd()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            _PROJECT_ROOT_CACHE = parent
            return parent

    _PROJECT_ROOT_CACHE = start
    return start


def prompts_dir() -> Path:
    return project_root() / "prompts"


def data_dir() -> Path:
    return project_root() / "data" / "exercises"


def output_dir() -> Path:
    return project_root() / "output"
