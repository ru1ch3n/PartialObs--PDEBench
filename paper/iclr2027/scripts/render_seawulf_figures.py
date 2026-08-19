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

PDE_FIGURE_LABELS = {
    pde: label.replace("--", "-") for pde, label in PDE_LABELS.items()
}

STATIC_PDES = {"darcy", "poisson", "helmholtz"}

PREFERRED_CASES = {
    "darcy": ("periodic", "piecewise_blocks", "medium"),
    "poisson": ("periodic", "gaussian_blobs", "medium"),
    "helmholtz": ("periodic", "multi_frequency_fourier", "high"),
    "heat": ("periodic", "gaussian_blobs", "medium"),
    "reaction_diffusion": ("periodic", "threshold_level_set", "medium"),
    "burgers": ("periodic", "front_ring_shock", "medium"),
    "navier_stokes": ("periodic", "dipole_vortex_pair", "medium"),
}

PDE_ACCENTS = {
    "darcy": "#20639B",
    "poisson": "#3CAEA3",
    "helmholtz": "#6C5CE7",
    "heat": "#F39C12",
    "reaction_diffusion": "#D3548C",
    "burgers": "#E76F51",
    "navier_stokes": "#264653",
}

STATIC_FIELD_LABELS = {
    "darcy": ("Permeability coefficient a(x)", "Pressure / head u(x)"),
    "poisson": ("Source field f(x)", "Potential u(x)"),
    "helmholtz": ("Source field f(x)", "Real solution u(x)"),
}

TEMPORAL_FIELD_LABELS = {
    "heat": "Temperature u(x,t)",
    "reaction_diffusion": "Order parameter u(x,t)",
    "burgers": "Scalar state u(x,t)",
    "navier_stokes": "Vorticity omega(x,t)",
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
    boundary, setting, regime = PREFERRED_CASES[pde]
    preferred = data_root / pde / boundary / setting / regime / "shard_00000.h5"
    if preferred.exists():
        return preferred
    shards = completed_shards(data_root, pde)
    if not shards:
        raise FileNotFoundError(f"no completed shard found for {pde}")
    return shards[0]


def load_sample(shard: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    with h5py.File(shard, "r") as handle:
        condition = np.asarray(handle["condition"][0, ..., 0], dtype=np.float64)
        trajectory = np.asarray(handle["trajectory"][0, ..., 0], dtype=np.float64)
        raw_spec = handle.attrs.get("spec_json", "{}")
    if isinstance(raw_spec, bytes):
        raw_spec = raw_spec.decode("utf-8")
    spec = json.loads(str(raw_spec))
    if trajectory.ndim == 2:
        trajectory = trajectory[np.newaxis, ...]
    elif trajectory.ndim != 3:
        raise ValueError(f"unexpected trajectory shape {trajectory.shape} in {shard}")
    return condition, trajectory, spec


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
    _, trajectory, _ = load_sample(shard)
    field = trajectory[-1]
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
        condition, trajectory, _ = load_sample(shard)
        target = trajectory[-1]
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


def _case_metadata(data_root: Path, shard: Path, trajectory: np.ndarray) -> dict[str, object]:
    relative = shard.relative_to(data_root)
    pde, boundary, setting, regime = relative.parts[:4]
    return {
        "pde": pde,
        "boundary": boundary,
        "setting": setting,
        "regime": regime,
        "sample_index": 0,
        "saved_frames": int(trajectory.shape[0]),
        "spatial_shape": [int(trajectory.shape[-2]), int(trajectory.shape[-1])],
        "source_shard": str(shard),
    }


def _field_style(field: np.ndarray, *, shared_limits: tuple[float, float] | None = None):
    symmetric = bool(np.nanmin(field) < 0.0 < np.nanmax(field))
    limits = shared_limits if shared_limits is not None else robust_limits(field, symmetric=symmetric)
    cmap = "RdBu_r" if symmetric or (limits[0] < 0.0 < limits[1]) else "viridis"
    return cmap, limits


def _style_field_axis(axis, accent: str) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("#F4F6F8")
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(accent)


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _add_card_header(fig, pde: str, metadata: dict[str, object]) -> None:
    accent = PDE_ACCENTS[pde]
    fig.suptitle(
        f"{PDE_FIGURE_LABELS[pde]} | accepted SeaWulf sample",
        x=0.055,
        y=0.985,
        ha="left",
        va="top",
        fontsize=14.0,
        fontweight="bold",
        color="#17212B",
    )
    badge = (
        f"{str(metadata['boundary']).upper()}  |  "
        f"{str(metadata['setting']).replace('_', ' ').upper()}  |  "
        f"{str(metadata['regime']).upper()}  |  128 x 128  |  ACCEPTED"
    )
    fig.text(
        0.055,
        0.925,
        badge,
        ha="left",
        va="top",
        fontsize=7.8,
        color="white",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": accent, "edgecolor": "none"},
    )


def render_static_pde_card(
    data_root: Path,
    output_dir: Path,
    pde: str,
    shard: Path,
    condition: np.ndarray,
    trajectory: np.ndarray,
) -> dict[str, object]:
    target = trajectory[-1]
    metadata = _case_metadata(data_root, shard, trajectory)
    accent = PDE_ACCENTS[pde]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.12, top=0.84, wspace=0.34, hspace=0.30)
    _add_card_header(fig, pde, metadata)

    condition_label, target_label = STATIC_FIELD_LABELS[pde]
    image_axes = (
        (axes[0, 0], condition, condition_label),
        (axes[0, 1], target, target_label),
    )
    for axis, field, label in image_axes:
        cmap, (vmin, vmax) = _field_style(field)
        image = axis.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axis.set_title(label, fontsize=9.2, fontweight="semibold", color="#25313C", pad=6)
        _style_field_axis(axis, accent)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        colorbar.ax.tick_params(labelsize=6.5, length=2)
        colorbar.outline.set_linewidth(0.4)

    gy, gx = np.gradient(target)
    gradient = np.hypot(gx, gy)
    _, (vmin, vmax) = _field_style(gradient)
    gradient_image = axes[1, 0].imshow(
        gradient, cmap="magma", vmin=vmin, vmax=vmax, origin="lower"
    )
    axes[1, 0].set_title("Solution gradient magnitude", fontsize=9.2, fontweight="semibold")
    _style_field_axis(axes[1, 0], accent)
    colorbar = fig.colorbar(gradient_image, ax=axes[1, 0], fraction=0.046, pad=0.025)
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    colorbar.outline.set_linewidth(0.4)

    center = condition.shape[0] // 2
    x = np.linspace(0.0, 1.0, condition.shape[1])
    condition_line = condition[center]
    target_line = target[center]

    def normalize(line: np.ndarray) -> np.ndarray:
        return (line - np.nanmean(line)) / (np.nanstd(line) + np.finfo(float).eps)

    axes[1, 1].plot(x, normalize(condition_line), color="#7F8C8D", linewidth=1.35, label="condition")
    axes[1, 1].plot(x, normalize(target_line), color=accent, linewidth=1.8, label="solution")
    axes[1, 1].axhline(0.0, color="#CBD2D9", linewidth=0.7)
    axes[1, 1].set_title("Normalized horizontal centerline", fontsize=9.2, fontweight="semibold")
    axes[1, 1].tick_params(labelsize=7, width=0.6)
    axes[1, 1].grid(alpha=0.18, linewidth=0.6)
    axes[1, 1].legend(frameon=False, fontsize=7.3, loc="best")
    for spine in axes[1, 1].spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#AAB3BB")

    fig.text(
        0.965,
        0.012,
        "Rendered read-only from sample 0 of an accepted full-tier HDF5 shard",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#66727D",
    )
    _save_figure(fig, output_dir, f"fig_pde_{pde}")
    return metadata


def render_temporal_pde_card(
    data_root: Path,
    output_dir: Path,
    pde: str,
    shard: Path,
    condition: np.ndarray,
    trajectory: np.ndarray,
) -> dict[str, object]:
    del condition  # The first stored frame is the released initial state for temporal PDEs.
    metadata = _case_metadata(data_root, shard, trajectory)
    accent = PDE_ACCENTS[pde]
    frame_indices = (0, len(trajectory) // 2, len(trajectory) - 1)
    combined = trajectory[np.asarray(frame_indices)]
    shared_limits = robust_limits(
        combined,
        symmetric=bool(np.nanmin(combined) < 0.0 < np.nanmax(combined)),
    )
    cmap, _ = _field_style(combined, shared_limits=shared_limits)

    fig = plt.figure(figsize=(7.8, 5.15), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.94,
        bottom=0.16,
        top=0.80,
        hspace=0.40,
        wspace=0.12,
        height_ratios=(1.0, 0.72),
    )
    _add_card_header(fig, pde, metadata)
    field_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    titles = ("Initial saved state", "Middle saved state", "Final saved state")
    image = None
    for axis, frame_index, title in zip(field_axes, frame_indices, titles, strict=True):
        image = axis.imshow(
            trajectory[frame_index],
            cmap=cmap,
            vmin=shared_limits[0],
            vmax=shared_limits[1],
            origin="lower",
        )
        axis.set_title(f"{title}\nframe {frame_index + 1}/{len(trajectory)}", fontsize=8.8, fontweight="semibold")
        _style_field_axis(axis, accent)
    assert image is not None
    colorbar = fig.colorbar(image, ax=field_axes, fraction=0.025, pad=0.018)
    colorbar.set_label(TEMPORAL_FIELD_LABELS[pde], fontsize=7.4)
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    colorbar.outline.set_linewidth(0.4)

    diagnostic_axis = fig.add_subplot(grid[1, :])
    time = np.linspace(0.0, 1.0, len(trajectory))
    rms = np.sqrt(np.nanmean(np.square(trajectory), axis=(1, 2)))
    peak = np.nanmax(np.abs(trajectory), axis=(1, 2))
    rms /= max(float(rms[0]), np.finfo(float).eps)
    peak /= max(float(peak[0]), np.finfo(float).eps)
    diagnostic_axis.plot(time, rms, color=accent, linewidth=2.1, marker="o", markersize=3.0, label="spatial RMS")
    diagnostic_axis.plot(time, peak, color="#7F8C8D", linewidth=1.45, linestyle="--", marker="s", markersize=2.7, label="peak magnitude")
    for frame_index in frame_indices:
        diagnostic_axis.axvline(time[frame_index], color="#D6DCE1", linewidth=0.7, zorder=0)
    diagnostic_axis.set_title("Evolution over the 15 released solver states", fontsize=9.3, fontweight="semibold")
    diagnostic_axis.set_xlabel("normalized physical time", fontsize=8)
    diagnostic_axis.set_ylabel("value / initial value", fontsize=8)
    diagnostic_axis.tick_params(labelsize=7, width=0.6)
    diagnostic_axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    diagnostic_axis.legend(frameon=False, fontsize=7.4, ncol=2, loc="best")
    for spine in diagnostic_axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#AAB3BB")

    fig.text(
        0.94,
        0.012,
        "Rendered read-only from sample 0 of an accepted full-tier HDF5 shard",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#66727D",
    )
    _save_figure(fig, output_dir, f"fig_pde_{pde}")
    return metadata


def render_pde_cards(data_root: Path, output_dir: Path) -> dict[str, object]:
    records: dict[str, object] = {}
    for pde in PDE_ORDER:
        shard = representative_shard(data_root, pde)
        condition, trajectory, spec = load_sample(shard)
        if pde in STATIC_PDES:
            metadata = render_static_pde_card(data_root, output_dir, pde, shard, condition, trajectory)
        else:
            metadata = render_temporal_pde_card(data_root, output_dir, pde, shard, condition, trajectory)
        metadata["spec"] = spec
        records[pde] = metadata
    return records


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
        "pde_figures": render_pde_cards(data_root, output_dir),
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
