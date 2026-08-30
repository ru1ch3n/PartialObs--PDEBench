# Copyright 2026 PDE-OBS contributors
# SPDX-License-Identifier: MIT
"""Local and SeaWulf preflight checks."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _path_check(name: str, raw_path: str | None, required: bool) -> Check:
    if not raw_path:
        return Check(name, not required, "not configured", required)
    path = Path(raw_path).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".pdeobs-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return Check(name, True, str(path.resolve()), required)
    except OSError as exc:
        return Check(name, False, f"{path}: {exc}", required)


def run_doctor(
    cluster: str = "local", require_gpu: bool = False, offline: bool = False
) -> list[Check]:
    checks: list[Check] = []
    checks.append(Check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    for package in ("numpy", "scipy", "h5py", "yaml"):
        checks.append(
            Check(f"package: {package}", importlib.util.find_spec(package) is not None, package)
        )

    torch_available = importlib.util.find_spec("torch") is not None
    checks.append(
        Check("package: torch", torch_available, "needed for learned baselines", require_gpu)
    )
    if require_gpu and torch_available:
        import torch

        checks.append(Check("CUDA available", torch.cuda.is_available(), str(torch.version.cuda)))

    checks.append(_path_check("data root", os.environ.get("PDEOBS_DATA"), cluster == "seawulf"))
    checks.append(_path_check("run root", os.environ.get("PDEOBS_RUNS"), cluster == "seawulf"))

    if cluster == "seawulf":
        checks.append(
            Check("Slurm command", shutil.which("sbatch") is not None, str(shutil.which("sbatch")))
        )
        checks.append(Check("SeaWulf filesystem", Path("/gpfs").exists(), "/gpfs"))
        checks.append(
            Check(
                "not on login node",
                "SLURM_JOB_ID" in os.environ,
                "run preflight inside an interactive or batch allocation",
            )
        )
    if offline:
        checks.append(Check("offline mode", True, "no network operation requested", False))
    return checks


def format_checks(checks: Iterable[Check]) -> str:
    lines = []
    for check in checks:
        label = "PASS" if check.ok else ("FAIL" if check.required else "WARN")
        lines.append(f"[{label}] {check.name}: {check.detail}")
    return "\n".join(lines)


def checks_succeeded(checks: Iterable[Check]) -> bool:
    return all(check.ok or not check.required for check in checks)
