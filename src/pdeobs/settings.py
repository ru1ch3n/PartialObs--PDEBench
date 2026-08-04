"""Deterministic input/source/initial-condition setting generators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .registry import SETTING_REGISTRY
from .schema import normalize_resolution

SETTING_NAMES = (
    "smooth_grf",
    "medium_grf",
    "rough_grf",
    "low_frequency_fourier",
    "multi_frequency_fourier",
    "gaussian_blobs",
    "piecewise_blocks",
    "threshold_level_set",
    "dipole_vortex_pair",
    "front_ring_shock",
)

SETTING_IDS = {f"S{index}": name for index, name in enumerate(SETTING_NAMES)}


def _coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    y = np.linspace(0.0, 1.0, height, endpoint=False, dtype=np.float64)
    x = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float64)
    return np.meshgrid(x, y)


def _standardize(field: np.ndarray, *, center: bool = True) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64)
    if center:
        result = result - np.mean(result)
    scale = float(np.std(result))
    if scale < 1e-12:
        scale = max(float(np.max(np.abs(result))), 1.0)
    return (result / scale).astype(np.float32)


def _spectral_grf(
    shape: tuple[int, int], rng: np.random.Generator, correlation: float
) -> np.ndarray:
    height, width = shape
    noise = rng.normal(size=shape)
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.rfftfreq(width)[None, :]
    radius2 = kx * kx + ky * ky
    # Gaussian covariance in Fourier space.  A larger correlation length damps
    # more high-frequency content.
    spectrum = np.exp(-2.0 * (np.pi * correlation) ** 2 * radius2 * height * width)
    spectrum[0, 0] = 0.0
    field = np.fft.irfft2(np.fft.rfft2(noise) * spectrum, s=shape).real
    return _standardize(field)


@SETTING_REGISTRY.register("smooth_grf", aliases=("s0", "smooth-grf", "smooth"))
def smooth_grf(shape: tuple[int, int], rng: np.random.Generator, **_: Any) -> np.ndarray:
    return _spectral_grf(shape, rng, correlation=0.22)


@SETTING_REGISTRY.register("medium_grf", aliases=("s1", "medium-grf"))
def medium_grf(shape: tuple[int, int], rng: np.random.Generator, **_: Any) -> np.ndarray:
    return _spectral_grf(shape, rng, correlation=0.09)


@SETTING_REGISTRY.register("rough_grf", aliases=("s2", "rough-grf", "rough"))
def rough_grf(shape: tuple[int, int], rng: np.random.Generator, **_: Any) -> np.ndarray:
    return _spectral_grf(shape, rng, correlation=0.025)


@SETTING_REGISTRY.register(
    "low_frequency_fourier", aliases=("s3", "low-frequency-fourier", "low_fourier")
)
def low_frequency_fourier(shape: tuple[int, int], rng: np.random.Generator, **_: Any) -> np.ndarray:
    x, y = _coordinates(shape)
    field = np.zeros(shape, dtype=np.float64)
    for _mode in range(4):
        kx, ky = rng.integers(0, 4, size=2)
        if kx == 0 and ky == 0:
            kx = 1
        amplitude = rng.normal() / np.sqrt(1.0 + kx * kx + ky * ky)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        field += amplitude * np.cos(2.0 * np.pi * (kx * x + ky * y) + phase)
    return _standardize(field)


@SETTING_REGISTRY.register(
    "multi_frequency_fourier",
    aliases=(
        "s4",
        "multi-frequency-fourier",
        "multi_fourier",
        "multifrequency_fourier",
    ),
)
def multi_frequency_fourier(
    shape: tuple[int, int], rng: np.random.Generator, **_: Any
) -> np.ndarray:
    x, y = _coordinates(shape)
    field = np.zeros(shape, dtype=np.float64)
    max_frequency = max(3, min(shape) // 4)
    for _mode in range(12):
        kx, ky = rng.integers(0, max_frequency + 1, size=2)
        if kx == 0 and ky == 0:
            ky = 1
        amplitude = rng.normal() / (1.0 + kx * kx + ky * ky) ** 0.45
        phase = rng.uniform(0.0, 2.0 * np.pi)
        field += amplitude * np.sin(2.0 * np.pi * (kx * x + ky * y) + phase)
    return _standardize(field)


@SETTING_REGISTRY.register("gaussian_blobs", aliases=("s5", "blobs", "gaussian-blobs"))
def gaussian_blobs(shape: tuple[int, int], rng: np.random.Generator, **_: Any) -> np.ndarray:
    x, y = _coordinates(shape)
    field = np.zeros(shape, dtype=np.float64)
    for _blob in range(int(rng.integers(2, 7))):
        center_x, center_y = rng.uniform(0.08, 0.92, size=2)
        sigma_x, sigma_y = rng.uniform(0.035, 0.16, size=2)
        amplitude = rng.uniform(0.5, 1.5) * rng.choice((-1.0, 1.0))
        field += amplitude * np.exp(
            -0.5 * (((x - center_x) / sigma_x) ** 2 + ((y - center_y) / sigma_y) ** 2)
        )
    return _standardize(field)


@SETTING_REGISTRY.register("piecewise_blocks", aliases=("s6", "blocks", "piecewise-blocks"))
def piecewise_blocks(shape: tuple[int, int], rng: np.random.Generator, **_: Any) -> np.ndarray:
    height, width = shape
    rows = int(rng.integers(2, min(7, height) + 1))
    columns = int(rng.integers(2, min(7, width) + 1))
    row_edges = np.linspace(0, height, rows + 1, dtype=int)
    col_edges = np.linspace(0, width, columns + 1, dtype=int)
    field = np.empty(shape, dtype=np.float64)
    values = rng.normal(size=(rows, columns))
    for row in range(rows):
        for column in range(columns):
            field[
                row_edges[row] : row_edges[row + 1],
                col_edges[column] : col_edges[column + 1],
            ] = values[row, column]
    return _standardize(field)


@SETTING_REGISTRY.register(
    "threshold_level_set", aliases=("s7", "threshold", "level_set", "level-set")
)
def threshold_level_set(shape: tuple[int, int], rng: np.random.Generator, **_: Any) -> np.ndarray:
    base = _spectral_grf(shape, rng, correlation=0.1)
    threshold = float(np.quantile(base, rng.uniform(0.35, 0.65)))
    return np.where(base >= threshold, 1.0, -1.0).astype(np.float32)


@SETTING_REGISTRY.register(
    "dipole_vortex_pair",
    aliases=("s8", "dipole", "vortex_pair", "vortex-pair"),
)
def dipole_vortex_pair(shape: tuple[int, int], rng: np.random.Generator, **_: Any) -> np.ndarray:
    x, y = _coordinates(shape)
    center = rng.uniform(0.3, 0.7, size=2)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    separation = rng.uniform(0.12, 0.28)
    delta = 0.5 * separation * np.array([np.cos(angle), np.sin(angle)])
    sigma = rng.uniform(0.045, 0.12)
    first = np.exp(
        -((x - center[0] - delta[0]) ** 2 + (y - center[1] - delta[1]) ** 2) / (2 * sigma**2)
    )
    second = np.exp(
        -((x - center[0] + delta[0]) ** 2 + (y - center[1] + delta[1]) ** 2) / (2 * sigma**2)
    )
    return _standardize(first - second)


@SETTING_REGISTRY.register(
    "front_ring_shock",
    aliases=("s9", "front", "ring", "shock", "front-ring-shock"),
)
def front_ring_shock(shape: tuple[int, int], rng: np.random.Generator, **_: Any) -> np.ndarray:
    x, y = _coordinates(shape)
    choice = int(rng.integers(0, 3))
    thickness = rng.uniform(0.012, 0.045)
    if choice == 0:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        coordinate = np.cos(angle) * (x - 0.5) + np.sin(angle) * (y - 0.5)
        field = np.tanh((coordinate - rng.uniform(-0.15, 0.15)) / thickness)
    elif choice == 1:
        center_x, center_y = rng.uniform(0.35, 0.65, size=2)
        radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        field = np.exp(-(((radius - rng.uniform(0.16, 0.34)) / thickness) ** 2))
    else:
        location = rng.uniform(0.25, 0.75)
        field = np.where(x < location, -1.0, 1.0)
    return _standardize(field)


def generate_setting(
    name: str,
    resolution: int | tuple[int, int],
    seed: int = 0,
    *,
    family: str | None = None,
    regime: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Generate one finite ``[H,W]`` field with an isolated RNG stream."""

    shape = normalize_resolution(resolution)
    generator: Callable[..., np.ndarray] = SETTING_REGISTRY.get(name)
    result = np.asarray(
        generator(
            shape,
            rng=np.random.default_rng(int(seed)),
            family=family,
            regime=regime,
            **kwargs,
        ),
        dtype=np.float32,
    )
    if result.shape != shape:
        raise RuntimeError(f"setting {name!r} returned {result.shape}, expected {shape}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"setting {name!r} produced non-finite values")
    return result


def list_settings() -> tuple[str, ...]:
    return SETTING_REGISTRY.names()
