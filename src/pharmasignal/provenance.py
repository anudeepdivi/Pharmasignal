"""Provenance helpers for model outputs: canonical bytes, hashes, identity.

Currently used by model training/artifact code (M3+). The M1 dataset build
keeps its own identical helpers so dataset evidence is not disturbed.
Hashes prove content identity, not authenticity.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy
import sklearn


def canonical_json(value: object) -> bytes:
    """Deterministic UTF-8 JSON bytes: sorted keys, no NaN, trailing newline."""
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": version("pandas"),
        "spacy": version("spacy"),
        "joblib": version("joblib"),
    }


def source_identity(source_dir: Path, project_root: Path | None = None) -> dict:
    """SHA-256 of every .py in `source_dir` plus git/lockfile state.

    `git_revision` stays None and `git_dirty` records what `git status`
    reported when git is unavailable; never invent a revision for an
    uncommitted tree.
    """
    project_root = project_root or source_dir.parents[2]
    hashes = {p.name: sha256(p.read_bytes()) for p in sorted(source_dir.glob("*.py"))}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    lock = project_root / "uv.lock"
    return {
        "source_sha256": hashes,
        "git_revision": revision,
        "git_dirty": dirty,
        "lockfile_sha256": sha256(lock.read_bytes()) if lock.exists() else None,
    }
