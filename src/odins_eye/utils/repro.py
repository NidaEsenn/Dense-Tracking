"""Reproducibility helpers.

Every experiment run lives in ``experiments/YYYYMMDD_<experiment-id>_<short-description>/``
and stores the resolved config, the git hash, metrics and logs.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds global state used by third-party code
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_hash(repo_dir: str | Path = ".") -> str:
    """Return the current commit hash, suffixed with ``-dirty`` if there are local changes.

    Returns ``"unknown"`` outside a git repository.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def make_run_dir(
    root: str | Path,
    experiment_id: str,
    description: str,
    config: dict[str, Any] | None = None,
    date: dt.date | None = None,
) -> Path:
    """Create ``root/YYYYMMDD_<experiment_id>_<description>/`` and record run metadata.

    Writes ``run.json`` with the git hash and the resolved config. Refuses to overwrite
    an existing run directory.
    """
    for name, value in (("experiment_id", experiment_id), ("description", description)):
        if not _SLUG.match(value):
            raise ValueError(f"{name} must be lowercase kebab-case, got {value!r}")

    date = date or dt.date.today()
    run_dir = Path(root) / f"{date:%Y%m%d}_{experiment_id}_{description}"
    run_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "experiment_id": experiment_id,
        "description": description,
        "date": date.isoformat(),
        "git_hash": git_hash(),
        "config": config or {},
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run_dir
