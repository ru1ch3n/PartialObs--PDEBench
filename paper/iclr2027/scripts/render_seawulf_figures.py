#!/usr/bin/env python3
"""Render appendix figures from accepted SeaWulf full-tier shards.

This script is intentionally read-only with respect to the dataset. It reads
completed HDF5 shards and their strict quality sidecars, then writes figure
artifacts and a machine-readable snapshot record to a separate output folder.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdeobs.masks import generate_mask  # noqa: E402


PDE_ORDER = (
    "darcy",
    "poisson",
    "helmholtz",
    "heat",
    "reaction_diffusion",
    "burgers",
    "navier_stokes",
)

PDE_LABELS = {
    "darcy": "Darcy",
    "poisson": "Poisson",
    "helmholtz": "Helmholtz",
    "heat": "Heat",
    "reaction_diffusion": "Reaction--diffusion",
    "burgers": "Burgers",
    "navier_stokes": "Navier--Stokes",
}

MASKS = (
    ("random_1pct", "Random 1%"),
    ("random_3pct", "Random 3%"),
    ("random_5pct", "Random 5%"),
    ("random_10pct", "Random 10%"),
    ("regular_grid", "Regular grid"),
    ("block_missing", "Missing block"),
    ("line_sensors", "Line sensors"),
    ("boundary_sensors", "Boundary sensors"),
    ("clustered_sensors", "Clustered sensors"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def completed_shards(data_root: Path, pde: str) -> list[Path]:
    return sorted(data_root.glob(f"{pde}/**/*.h5"))


def representative_shard(data_root: Path, pde: str) -> Path:
    preferred = data_root / pde / "periodic" / "smooth_grf" / "medium" / "shard_00000.h5"
    if preferred.exists():
        return preferred
    shards = completed_shards(data_root, pde)
    if not shards:
        raise FileNotFoundError(f"no completed shard found for {pde}")
    return shards[0]


def load_pair(shard: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(shard, "r") as handle:
        condition = np.asarray(handle["condition"][0, ..., 0], dtype=np.float64)
        trajectory = np.asarray(handle["trajectory"][0, ..., 0], dtype=np.float64)
    if trajectory.ndim == 3:
        target = trajectory[-1]
    elif trajectory.ndim == 2:
        target = trajectory
    else:
        raise ValueError(f"unexpected trajectory shape {trajectory.shape} in {shard}")
    return condition, target


def robust_limits(field: np.ndarray, *, symmetric: bool = False) -> tuple[float, float]:
    lo, hi = np.nanpercentile(field, (1.0, 99.0))
    if symmetric:
        bound = max(abs(float(lo)), abs(float(hi)), np.finfo(float).eps)
        return -bound, bound
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def render_observation_protocols(data_root: Path, output_dir: Path) -> dict[str, object]:
    shard = representative_shard(data_root, "heat")
    _, field = load_pair(shard)
    vmin, vmax = robust_limits(field)
    fig, axes = plt.subplots(3, 3, figsize=(9.4, 8.7), constrained_layout=True)
    counts: dict[str, int] = {}
    image = None
    for index, ((protocol, label), axis) in enumerate(zip(MASKS, axes.flat, strict=True)):
        mask = generate_mask(protocol, field.shape, seed=20260804 + index)
        counts[protocol] = int(mask.sum())
        axis.imshow(field, cmap="Greys", vmin=vmin, vmax=vmax, alpha=0.16, origin="lower")
        observed = np.ma.masked_where(~mask, field)
        image = axis.imshow(observed, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        axis.set_title(f"{label}\n{mask.sum():,} observed ({100.0 * mask.mean():.2f}%)", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#5f6b73")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.018)
    colorbar.set_label("Accepted heat state")
    fig.suptitle("Official partial-observation protocols on one accepted full-tier state", fontsize=13)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_observation_protocols.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {"source_shard": str(shard), "mask_counts": counts}


def render_pde_gallery(data_root: Path, output_dir: Path) -> dict[str, object]:
    fig, axes = plt.subplots(2, len(PDE_ORDER), figsize=(13.8, 4.7), constrained_layout=True)
    sources: dict[str, str] = {}
    for column, pde in enumerate(PDE_ORDER):
        shard = representative_shard(data_root, pde)
        sources[pde] = str(shard)
        condition, target = load_pair(shard)
        for row, (field, row_label) in enumerate(((condition, "Condition / initial"), (target, "Solution / final"))):
            axis = axes[row, column]
            symmetric = bool(np.nanmin(field) < 0.0 < np.nanmax(field))
            vmin, vmax = robust_limits(field, symmetric=symmetric)
            cmap = "RdBu_r" if symmetric else "viridis"
            axis.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(PDE_LABELS[pde], fontsize=9.5)
            if column == 0:
                axis.set_ylabel(row_label, fontsize=9)
            for spine in axis.spines.values():
                spine.set_linewidth(0.45)
                spine.set_color("#5f6b73")
    fig.suptitle("Accepted full-tier fields rendered directly from SeaWulf HDF5 shards", fontsize=13)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_pde_gallery.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {"source_shards": sources}


def collect_quality_snapshot(data_root: Path) -> tuple[dict[str, dict[str, float]], int, int]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "sum": 0.0, "max": 0.0})
    sidecar_count = 0
    for raw_path in glob.iglob(str(data_root / "**" / "*.quality.json"), recursive=True):
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        quality = payload.get("quality", {})
        for pde, record in quality.get("pde_losses", {}).items():
            stats = record.get("pde_loss_normalized", {})
            count = int(stats.get("count", 0))
            if count <= 0:
                continue
            totals[pde]["count"] += count
            totals[pde]["sum"] += count * float(stats["mean"])
            totals[pde]["max"] = max(totals[pde]["max"], float(stats["max"]))
        sidecar_count += 1
    summary: dict[str, dict[str, float]] = {}
    for pde in PDE_ORDER:
        record = totals[pde]
        count = int(record["count"])
        if count:
            summary[pde] = {
                "count": count,
                "mean": record["sum"] / count,
                "max": record["max"],
            }
    return summary, sidecar_count, sum(int(item["count"]) for item in summary.values())


def render_quality_snapshot(data_root: Path, output_dir: Path) -> dict[str, object]:
    summary, sidecar_count, sample_count = collect_quality_snapshot(data_root)
    labels = [PDE_LABELS[pde] for pde in PDE_ORDER]
    means = np.asarray([summary[pde]["mean"] for pde in PDE_ORDER])
    maxima = np.asarray([summary[pde]["max"] for pde in PDE_ORDER])
    x = np.arange(len(PDE_ORDER))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.5, 4.5), constrained_layout=True)
    axis.bar(x - width / 2, means, width, label="Mean accepted loss", color="#4C78A8")
    axis.bar(x + width / 2, maxima, width, label="Maximum accepted loss", color="#F58518")
    axis.axhline(0.05, color="#B84842", linewidth=1.3, linestyle="--", label="Strict gate (0.05)")
    axis.set_yscale("log")
    axis.set_ylabel("Normalized discrete PDE loss (log scale)")
    axis.set_xticks(x, labels, rotation=24, ha="right")
    axis.grid(axis="y", which="both", alpha=0.22, linewidth=0.6)
    axis.legend(
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="none",
        ncol=3,
        fontsize=8.5,
        loc="upper center",
    )
    axis.set_title(f"Full-T15 accepted-shard snapshot: {sidecar_count:,} shards, {sample_count:,} samples")
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_full_quality_snapshot.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {
        "quality_sidecars": sidecar_count,
        "accepted_samples": sample_count,
        "by_pde": summary,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "observation_protocols": render_observation_protocols(data_root, output_dir),
        "pde_gallery": render_pde_gallery(data_root, output_dir),
        "quality_snapshot": render_quality_snapshot(data_root, output_dir),
        "interpretation": (
            "Read-only snapshot of completed, accepted shards. This is not the final strict "
            "aggregate and excludes absent or rejected shards."
        ),
    }
    (output_dir / "figure_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
