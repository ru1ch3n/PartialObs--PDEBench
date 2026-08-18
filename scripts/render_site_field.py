"""Render the deterministic PDE-OBS field used by the project website hero."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pdeobs.masks import generate_mask  # noqa: E402
from pdeobs.pdes import generate_sample  # noqa: E402


def render(output: Path) -> None:
    """Generate one benchmark sample and render its field plus observed sensors."""

    sample = generate_sample(
        "poisson",
        boundary="periodic",
        setting="smooth_grf",
        regime="low",
        seed=17,
        resolution=128,
    )
    field = np.asarray(sample.trajectory[0, ..., 0], dtype=np.float64)
    mask = generate_mask("random_3pct", field.shape, seed=23)

    palette = LinearSegmentedColormap.from_list(
        "pdeobs",
        ("#071225", "#111f53", "#284fc5", "#188bc8", "#27d2e7", "#effcff"),
    )

    figure = plt.figure(figsize=(16, 9), dpi=100, facecolor="#071225")
    canvas = figure.add_axes((0, 0, 1, 1))
    canvas.set_axis_off()
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)

    # A quiet technical grid keeps the left half usable for live HTML text.
    for position in np.linspace(0, 1, 19):
        canvas.plot((position, position), (0, 1), color="#93a4c5", alpha=0.045, lw=0.7)
    for position in np.linspace(0, 1, 12):
        canvas.plot((0, 1), (position, position), color="#93a4c5", alpha=0.045, lw=0.7)

    field_axes = figure.add_axes((0.43, 0.055, 0.55, 0.89))
    field_axes.imshow(field, cmap=palette, origin="lower", interpolation="bicubic")
    levels = np.linspace(float(field.min()), float(field.max()), 17)
    field_axes.contour(field, levels=levels, colors="#c7f7ff", linewidths=0.55, alpha=0.27)

    sensor_y, sensor_x = np.where(mask)
    sensor_values = field[mask]
    field_axes.scatter(
        sensor_x,
        sensor_y,
        c=sensor_values,
        cmap=palette,
        vmin=float(field.min()),
        vmax=float(field.max()),
        s=9,
        edgecolors="#eaffff",
        linewidths=0.28,
        alpha=0.86,
    )
    for position in np.linspace(0, field.shape[0] - 1, 9):
        field_axes.axhline(position, color="#c5d8ff", alpha=0.075, lw=0.55)
        field_axes.axvline(position, color="#c5d8ff", alpha=0.075, lw=0.55)
    field_axes.set_axis_off()

    # Fade the scientific field toward the headline area without modifying data values.
    fade = np.zeros((2, 640, 4), dtype=np.float64)
    fade[..., 0] = 7 / 255
    fade[..., 1] = 18 / 255
    fade[..., 2] = 37 / 255
    fade[..., 3] = np.linspace(1.0, 0.0, fade.shape[1])
    fade_axes = figure.add_axes((0.28, 0, 0.34, 1))
    fade_axes.imshow(fade, origin="lower", aspect="auto")
    fade_axes.set_facecolor("none")
    fade_axes.set_axis_off()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, facecolor="#071225", bbox_inches=None)
    plt.close(figure)
    buffer.seek(0)

    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(buffer) as image:
        image.convert("RGB").save(output, format="WEBP", quality=88, method=6)

    print(
        f"saved {output} | field={field.shape} | observed={int(mask.sum())} "
        f"({mask.mean():.3%}) | seed=17/mask_seed=23"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "assets" / "pdeobs-field.webp",
    )
    arguments = parser.parse_args()
    render(arguments.output)


if __name__ == "__main__":
    main()
