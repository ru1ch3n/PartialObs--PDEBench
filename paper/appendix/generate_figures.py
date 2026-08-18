"""Generate the figures and tabular source data for Appendix A--D.

The PDE gallery deliberately evaluates exactly one reduced-resolution sample
per family.  It is an explanatory rendering of the checked-in numerical
routes, not a local benchmark campaign and not a model-training run.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdeobs.masks import MASK_PROTOCOL_NAMES, generate_mask  # noqa: E402
from pdeobs.pdes import generate_sample  # noqa: E402


FIGURES = HERE / "figures"
DATA = HERE / "figure_data"

COLORS = {
    "navy": "#153B5B",
    "blue": "#3579A8",
    "cyan": "#61B5C2",
    "orange": "#E07A3F",
    "red": "#B84842",
    "gray": "#68737D",
    "light": "#E9EFF3",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png")
    plt.close(fig)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_pde_gallery() -> None:
    cases = [
        ("darcy", "Darcy flow", "smooth_grf", 1101),
        ("poisson", "Poisson", "gaussian_blobs", 1202),
        ("helmholtz", "Helmholtz", "multi_frequency_fourier", 1303),
        ("heat", "Heat", "rough_grf", 1404),
        ("reaction_diffusion", "Reaction--diffusion", "threshold_level_set", 1505),
        ("burgers", "Burgers", "front_ring_shock", 1606),
        ("navier_stokes", "Navier--Stokes", "dipole_vortex_pair", 1707),
    ]
    # Two paired blocks keep every panel readable on a portrait paper page:
    # four PDEs on the left and three on the right.
    fig, axes = plt.subplots(4, 4, figsize=(7.1, 7.45), constrained_layout=True)
    for row, (family, label, setting, seed) in enumerate(cases):
        output = generate_sample(
            family,
            boundary="periodic",
            setting=setting,
            regime="medium",
            seed=seed,
            resolution=64,
            time_steps=15 if family in {"heat", "reaction_diffusion", "burgers", "navier_stokes"} else None,
        )
        condition = np.asarray(output.condition[..., 0], dtype=float)
        solution = np.asarray(output.trajectory[-1, ..., 0], dtype=float)
        block = 0 if row < 4 else 1
        plot_row = row if row < 4 else row - 4
        for local_col, (field, title) in enumerate(((condition, "Condition / initial"), (solution, "Solution / final"))):
            col = 2 * block + local_col
            ax = axes[plot_row, col]
            vmax = float(np.nanpercentile(np.abs(field), 99.5))
            vmax = max(vmax, np.finfo(float).eps)
            image = ax.imshow(field, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if plot_row == 0:
                ax.set_title(title, fontweight="bold", pad=6)
            if local_col == 0:
                ax.set_ylabel(label, fontweight="bold", rotation=90, labelpad=7)
            inset = ax.inset_axes([0.73, 0.06, 0.23, 0.045])
            cbar = fig.colorbar(image, cax=inset, orientation="horizontal")
            cbar.ax.tick_params(labelsize=5, length=1, pad=1)
            cbar.outline.set_linewidth(0.3)
    for col in (2, 3):
        axes[3, col].axis("off")
    axes[3, 2].text(
        0.02,
        0.92,
        "Illustrative route check",
        transform=axes[3, 2].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    axes[3, 2].text(
        0.02,
        0.72,
        "One deterministic 64 x 64 sample\nper PDE; periodic boundary;\nmedium regime; no model training.",
        transform=axes[3, 2].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        linespacing=1.35,
        color=COLORS["gray"],
    )
    _save(fig, "fig_pde_gallery")


def make_observation_protocols() -> None:
    output = generate_sample(
        "heat",
        boundary="periodic",
        setting="multi_frequency_fourier",
        regime="medium",
        seed=2718,
        resolution=128,
        time_steps=15,
    )
    field = np.asarray(output.trajectory[7, ..., 0], dtype=float)
    display_names = {
        "random_1pct": "Random 1%",
        "random_3pct": "Random 3%",
        "random_5pct": "Random 5%",
        "random_10pct": "Random 10%",
        "regular_grid": "Regular grid",
        "block_missing": "Missing block",
        "line_sensors": "Line sensors",
        "boundary_sensors": "Boundary sensors",
        "clustered_sensors": "Clustered sensors",
    }
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(3, 3, figsize=(7.1, 7.15), constrained_layout=True)
    vmax = max(float(np.nanpercentile(np.abs(field), 99.5)), np.finfo(float).eps)
    for index, protocol in enumerate(MASK_PROTOCOL_NAMES):
        ax = axes.flat[index]
        mask = generate_mask(protocol, (128, 128), seed=31415 + index)
        count = int(mask.sum())
        ratio = 100.0 * count / mask.size
        rows.append(
            {
                "protocol": protocol,
                "observed_pixels": count,
                "total_pixels": int(mask.size),
                "observed_percent": f"{ratio:.6f}",
            }
        )
        ax.imshow(field, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.18)
        yy, xx = np.nonzero(mask)
        if count > 3000:
            observed = np.ma.masked_where(~mask, field)
            ax.imshow(observed, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            colors = field[yy, xx]
            marker_size = 1.2 if count > 1000 else 3.0
            ax.scatter(xx, yy, c=colors, s=marker_size, cmap="RdBu_r", vmin=-vmax, vmax=vmax, linewidths=0)
        ax.set_title(f"{display_names[protocol]}\n{count:,} / 16,384 ({ratio:.2f}%)", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#CCD4DA")
            spine.set_linewidth(0.5)
    _write_csv(
        DATA / "mask_counts_128.csv",
        ["protocol", "observed_pixels", "total_pixels", "observed_percent"],
        rows,
    )
    _save(fig, "fig_observation_protocols")


def make_validation_losses() -> None:
    rows = [
        {"pde": "Darcy", "mean": 0.00002262, "max": 0.00033192},
        {"pde": "Poisson", "mean": 0.00001553, "max": 0.00008154},
        {"pde": "Helmholtz", "mean": 0.00003100, "max": 0.00126776},
        {"pde": "Heat", "mean": 0.00000348, "max": 0.00004155},
        {"pde": "Reaction--diffusion", "mean": 0.00257861, "max": 0.03826684},
        {"pde": "Burgers", "mean": 0.01279232, "max": 0.04900286},
        {"pde": "Navier--Stokes", "mean": 0.01205351, "max": 0.04825938},
    ]
    _write_csv(
        DATA / "validation20_pde_losses.csv",
        ["pde", "mean", "max"],
        rows,
    )
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.1, 3.8), constrained_layout=True)
    mean_values = np.array([float(row["mean"]) for row in rows])
    max_values = np.array([float(row["max"]) for row in rows])
    ax.bar(x - width / 2, mean_values, width, label="Mean", color=COLORS["blue"])
    ax.bar(x + width / 2, max_values, width, label="Maximum", color=COLORS["orange"])
    ax.axhline(0.05, color=COLORS["red"], linewidth=1.3, linestyle="--", label="Acceptance gate: 0.05")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e-1)
    ax.set_ylabel("Normalized discrete PDE loss")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Darcy", "Poisson", "Helmholtz", "Heat", "Reaction--\ndiffusion", "Burgers", "Navier--\nStokes"]
    )
    ax.grid(axis="y", which="both", color="#DDE3E8", linewidth=0.55)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.text(
        0.99,
        0.04,
        "validation20: 5,600 samples; operator-matched preflight",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=COLORS["gray"],
    )
    _save(fig, "fig_validation_losses")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    _style()
    make_pde_gallery()
    make_observation_protocols()
    make_validation_losses()
    print(f"Wrote figures to {FIGURES}")
    print(f"Wrote source tables to {DATA}")


if __name__ == "__main__":
    main()
