#!/usr/bin/env python3
"""Build deterministic, auditable plans for strict-quality recovery.

This helper never deletes or mutates dataset files.  It derives either a small
refinement pilot or a recovery plan from the frozen campaign plan and stored
``*.quality-failures.jsonl`` records.  Failed temporal strata keep the same
logical sample indices and seeds while using a finer internal time grid; the
published trajectory length remains unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

TEMPORAL_FAMILIES = frozenset({"heat", "reaction_diffusion", "burgers", "navier_stokes"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _failure_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(dataset_root.rglob("*.quality-failures.jsonl")):
        for record in _read_jsonl(path):
            quality = record.get("quality", {})
            context = quality.get("calibration_context", {})
            metrics = quality.get("metrics", {})
            rows.append(
                {
                    "path": path,
                    "record": record,
                    "pde": context.get("pde") or quality.get("pde"),
                    "boundary": context.get("boundary") or quality.get("boundary"),
                    "setting": context.get("setting"),
                    "regime": context.get("regime"),
                    "loss": metrics.get("pde_loss_normalized"),
                }
            )
    return rows


def _output_for_failure(path: Path) -> str:
    suffix = ".quality-failures.jsonl"
    value = str(path)
    if not value.endswith(suffix):
        raise ValueError(f"unexpected failure-record path: {path}")
    return value[: -len(suffix)] + ".h5"


def _semantic_output_key(path: str | Path) -> tuple[str, ...]:
    """Return ``pde/boundary/setting/regime/shard`` independent of root."""

    parts = Path(path).parts
    if len(parts) < 5:
        raise ValueError(f"output path does not contain the canonical factors: {path}")
    return tuple(parts[-5:])


def _refine_time_steps(row: dict[str, Any], factor: int) -> None:
    if row.get("pde") not in TEMPORAL_FAMILIES:
        return
    base = int(row["time_steps"])
    row["time_steps"] = 1 + factor * (base - 1)


def build_pilot(
    plan: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    factor: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_output = {_semantic_output_key(row["output_path"]): row for row in plan}
    worst: dict[str, dict[str, Any]] = {}
    for failure in failures:
        pde = str(failure["pde"])
        loss = failure["loss"]
        if pde not in TEMPORAL_FAMILIES or not isinstance(loss, (int, float)):
            continue
        if pde not in worst or float(loss) > float(worst[pde]["loss"]):
            worst[pde] = failure
    if not worst:
        raise ValueError("no temporal PDE-loss failures were found")

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for offset, pde in enumerate(sorted(worst)):
        failure = worst[pde]
        output = _semantic_output_key(_output_for_failure(failure["path"]))
        if output not in by_output:
            raise ValueError(f"failure has no matching frozen plan row: {output}")
        row = dict(by_output[output])
        sample_id = str(failure["record"]["sample_id"])
        logical_index = int(sample_id.rsplit("/", 1)[-1])
        base_steps = int(row["time_steps"])
        row["sample_start"] = logical_index
        row["sample_count"] = 1
        row["shard_index"] = 90000 + offset
        _refine_time_steps(row, factor)
        row["job_id"] = f"{pde}/quality-refinement-pilot/{logical_index}"
        rows.append(row)
        cases.append(
            {
                "pde": pde,
                "sample_id": sample_id,
                "original_loss": float(failure["loss"]),
                "base_time_steps": base_steps,
                "refined_time_steps": int(row["time_steps"]),
            }
        )
    return rows, {"mode": "pilot", "factor": factor, "cases": cases}


def build_recovery(
    plan: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    dataset_root: Path,
    factor: int,
    refine_all_temporal: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    failed_strata = {
        (failure["pde"], failure["boundary"], failure["setting"], failure["regime"])
        for failure in failures
    }
    recovery: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    refined = 0
    for original in plan:
        output = dataset_root.joinpath(*_semantic_output_key(original["output_path"]))
        complete = output.is_file() and output.with_suffix(".manifest.json").is_file()
        if complete:
            combined.append(original)
            continue
        row = dict(original)
        key = (row["pde"], row["boundary"], row["setting"], row["regime"])
        if row["pde"] in TEMPORAL_FAMILIES and (refine_all_temporal or key in failed_strata):
            _refine_time_steps(row, factor)
            refined += 1
        recovery.append(row)
        combined.append(row)
    summary = {
        "mode": "recovery",
        "factor": factor,
        "frozen_plan_rows": len(plan),
        "already_complete_rows": len(plan) - len(recovery),
        "recovery_rows": len(recovery),
        "refined_recovery_rows": refined,
        "refine_all_temporal": refine_all_temporal,
        "failure_record_count": len(failures),
        "failed_strata": len(failed_strata),
    }
    return recovery, combined, summary


def quarantine_incomplete(
    rows: list[dict[str, Any]],
    *,
    dataset_root: Path,
    quarantine_dir: Path,
) -> dict[str, Any]:
    """Move only incomplete-shard artifacts into a reversible quarantine."""

    dataset_root = dataset_root.resolve()
    quarantine_dir.mkdir(parents=True, exist_ok=False)
    moved: list[dict[str, Any]] = []
    skipped_complete = 0
    for row in rows:
        output = dataset_root.joinpath(*_semantic_output_key(row["output_path"]))
        manifest = output.with_suffix(".manifest.json")
        if output.is_file() and manifest.is_file():
            skipped_complete += 1
            continue
        try:
            relative_parent = output.parent.resolve().relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(f"recovery output escapes dataset root: {output}") from exc
        target_parent = quarantine_dir / relative_parent
        target_parent.mkdir(parents=True, exist_ok=True)
        for source in sorted(output.parent.glob(output.stem + ".*")):
            if not source.is_file():
                continue
            target = target_parent / source.name
            if target.exists():
                raise FileExistsError(f"quarantine target already exists: {target}")
            size = source.stat().st_size
            os.replace(source, target)
            moved.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "bytes": size,
                }
            )
    manifest_path = quarantine_dir / "quarantine_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_root": str(dataset_root),
                "recovery_rows": len(rows),
                "skipped_complete_rows": skipped_complete,
                "moved_file_count": len(moved),
                "moved": moved,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "quarantine_dir": str(quarantine_dir),
        "quarantined_files": len(moved),
        "skipped_complete_rows": skipped_complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "recovery"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--combined-plan", type=Path)
    parser.add_argument("--refinement-factor", type=int, default=2)
    parser.add_argument("--refine-all-temporal", action="store_true")
    parser.add_argument("--quarantine-dir", type=Path)
    args = parser.parse_args()
    if args.refinement_factor < 2:
        parser.error("--refinement-factor must be at least 2")

    plan = _read_jsonl(args.plan)
    failures = _failure_rows(args.dataset_root)
    if args.mode == "pilot":
        rows, summary = build_pilot(plan, failures, factor=args.refinement_factor)
    else:
        rows, combined, summary = build_recovery(
            plan,
            failures,
            dataset_root=args.dataset_root,
            factor=args.refinement_factor,
            refine_all_temporal=args.refine_all_temporal,
        )
        if args.combined_plan is None:
            parser.error("--combined-plan is required in recovery mode")
        _write_jsonl(args.combined_plan, combined)
        if args.quarantine_dir is not None:
            summary.update(
                quarantine_incomplete(
                    rows,
                    dataset_root=args.dataset_root,
                    quarantine_dir=args.quarantine_dir,
                )
            )
    _write_jsonl(args.output_plan, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
