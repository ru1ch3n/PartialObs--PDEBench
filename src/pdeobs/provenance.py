"""Reproducibility metadata captured beside every run or dataset shard."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import config_hash


def generation_identity(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    """Select stable fields that identify the code/config used by a data plan."""

    values = dict(provenance or {})
    git = values.get("git", {})
    return {
        "config_hash": values.get("config_hash"),
        "git": {
            "commit": git.get("commit") if isinstance(git, Mapping) else None,
            "dirty": git.get("dirty") if isinstance(git, Mapping) else None,
            "status": git.get("status") if isinstance(git, Mapping) else None,
        },
    }


def _command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_provenance(
    config: Mapping[str, Any] | None = None,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Collect stable, non-secret host, package, Git, and Slurm information."""

    repo = Path(repository).resolve() if repository else Path.cwd()
    packages = {}
    for name in ("pdeobs", "numpy", "scipy", "h5py", "torch", "PyYAML"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    git_commit = _command(["git", "rev-parse", "HEAD"], repo)
    git_status = _command(["git", "status", "--porcelain"], repo)
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
    )
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config) if config is not None else None,
        "git": {
            "commit": git_commit,
            "dirty": bool(git_status) if git_status is not None else None,
            "status": git_status,
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        },
        "packages": packages,
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
    }


def write_provenance(
    destination: str | Path,
    config: Mapping[str, Any] | None = None,
    repository: str | Path | None = None,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_provenance(config=config, repository=repository)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
