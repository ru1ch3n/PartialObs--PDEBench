"""Shared, dependency-light utilities for PDE-OBS reference generators.

The solvers in :mod:`pdeobs.pdes` are deliberately compact reference
implementations.  They produce deterministic benchmark fixtures and small
research datasets without requiring a heavyweight PDE package.  They are not
intended to replace a converged, application-specific simulation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]
Resolution: TypeAlias = int | tuple[int, int]

STATIC_TIME_STEPS = 1
TEMPORAL_TIME_STEPS = 9
MIN_RESOLUTION = 4


def _token(value: str) -> str:
    """Return a forgiving identifier token used by all public normalizers."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("identifier values must be non-empty strings")
    text = value.strip().lower()
    for dash in ("\N{EN DASH}", "\N{EM DASH}", "\N{MINUS SIGN}"):
        text = text.replace(dash, "-")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


FAMILY_ALIASES: dict[str, str] = {
    "f0": "darcy",
    "darcy": "darcy",
    "darcy_flow": "darcy",
    "f1": "poisson",
    "poisson": "poisson",
    "f2": "helmholtz",
    "helmholtz": "helmholtz",
    "f3": "heat",
    "heat": "heat",
    "diffusion": "heat",
    "heat_diffusion": "heat",
    "f4": "reaction_diffusion",
    "reaction_diffusion": "reaction_diffusion",
    "reactiondiffusion": "reaction_diffusion",
    "f5": "burgers",
    "burger": "burgers",
    "burgers": "burgers",
    "f6": "navier_stokes",
    "navier_stokes": "navier_stokes",
    "navierstokes": "navier_stokes",
    "ns": "navier_stokes",
}

BOUNDARY_ALIASES: dict[str, str] = {
    "b0": "dirichlet",
    "dirichlet": "dirichlet",
    "no_slip": "dirichlet",
    "dirichlet_no_slip": "dirichlet",
    "b1": "neumann",
    "neumann": "neumann",
    "free_slip": "neumann",
    "neumann_free_slip": "neumann",
    "zero_flux": "neumann",
    "b2": "periodic",
    "periodic": "periodic",
    "wrap": "periodic",
    "b3": "robin",
    "robin": "robin",
    "mixed": "robin",
    "obstacle": "robin",
    "robin_obstacle": "robin",
    "robin_mixed_obstacle": "robin",
    "mixed_obstacle": "robin",
}

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

SETTING_ALIASES: dict[str, str] = {f"s{i}": name for i, name in enumerate(SETTING_NAMES)}
SETTING_ALIASES.update(
    {
        "smooth_grf": "smooth_grf",
        "medium_grf": "medium_grf",
        "rough_grf": "rough_grf",
        "low_frequency_fourier": "low_frequency_fourier",
        "low_fourier": "low_frequency_fourier",
        "low_freq_fourier": "low_frequency_fourier",
        "fourier_low": "low_frequency_fourier",
        "multi_frequency_fourier": "multi_frequency_fourier",
        "multifrequency_fourier": "multi_frequency_fourier",
        "multi_freq_fourier": "multi_frequency_fourier",
        "fourier_multi": "multi_frequency_fourier",
        "gaussian_blobs": "gaussian_blobs",
        "gaussian_blob": "gaussian_blobs",
        "blobs": "gaussian_blobs",
        "piecewise_blocks": "piecewise_blocks",
        "piecewise_block": "piecewise_blocks",
        "blocks": "piecewise_blocks",
        "threshold_level_set": "threshold_level_set",
        "threshold": "threshold_level_set",
        "level_set": "threshold_level_set",
        "dipole_vortex_pair": "dipole_vortex_pair",
        "dipole": "dipole_vortex_pair",
        "vortex_pair": "dipole_vortex_pair",
        "front_ring_shock": "front_ring_shock",
        "front": "front_ring_shock",
        "ring": "front_ring_shock",
        "shock": "front_ring_shock",
    }
)

REGIME_ALIASES: dict[str, str] = {
    "r0": "low",
    "low": "low",
    "easy": "low",
    "r1": "medium",
    "medium": "medium",
    "mid": "medium",
    "base": "medium",
    "r2": "high",
    "high": "high",
    "hard": "high",
}


def _normalize(value: str, aliases: Mapping[str, str], kind: str) -> str:
    token = _token(value)
    try:
        return aliases[token]
    except KeyError as exc:
        choices = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"unknown {kind} {value!r}; choose one of: {choices}") from exc


def normalize_family(value: str) -> str:
    return _normalize(value, FAMILY_ALIASES, "PDE family")


def normalize_boundary(value: str) -> str:
    return _normalize(value, BOUNDARY_ALIASES, "boundary")


def normalize_setting(value: str) -> str:
    # Import built-ins lazily to avoid a package cycle while keeping the public
    # registry as the single source of truth for generated setting fields.
    from .. import settings as _built_in_settings  # noqa: F401
    from ..registry import SETTING_REGISTRY

    token = _token(value)
    if token in SETTING_REGISTRY:
        return SETTING_REGISTRY.resolve_name(token)
    return _normalize(value, SETTING_ALIASES, "setting")


def normalize_regime(value: str) -> str:
    return _normalize(value, REGIME_ALIASES, "regime")


def parse_resolution(resolution: Resolution) -> tuple[int, int]:
    """Normalize ``H x W`` resolution input and reject unsafe tiny grids."""

    if isinstance(resolution, Integral):
        height = width = int(resolution)
    elif isinstance(resolution, tuple) and len(resolution) == 2:
        height, width = resolution
        if not isinstance(height, Integral) or not isinstance(width, Integral):
            raise TypeError("resolution tuple entries must be integers")
        height, width = int(height), int(width)
    else:
        raise TypeError("resolution must be an integer or an (H, W) tuple")
    if height < MIN_RESOLUTION or width < MIN_RESOLUTION:
        raise ValueError(f"each resolution dimension must be at least {MIN_RESOLUTION}")
    return height, width


def resolve_time_steps(time_steps: int | None, *, temporal: bool) -> int:
    if time_steps is None:
        return TEMPORAL_TIME_STEPS if temporal else STATIC_TIME_STEPS
    if not isinstance(time_steps, Integral) or int(time_steps) < 1:
        raise ValueError("time_steps must be a positive integer")
    steps = int(time_steps)
    if not temporal and steps != STATIC_TIME_STEPS:
        raise ValueError("static PDE families always have T=1")
    return steps


def make_rng(seed: int, stream: int = 0) -> np.random.Generator:
    """Create a reproducible independent RNG stream without global state."""

    if not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if not isinstance(stream, Integral) or int(stream) < 0:
        raise ValueError("stream must be a non-negative integer")
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def grid(resolution: Resolution) -> tuple[FloatArray, FloatArray, float, float]:
    """Return cell-centered ``x, y`` meshes and grid spacings."""

    height, width = parse_resolution(resolution)
    dx, dy = 1.0 / width, 1.0 / height
    x = (np.arange(width, dtype=np.float64) + 0.5) * dx
    y = (np.arange(height, dtype=np.float64) + 0.5) * dy
    xx, yy = np.meshgrid(x, y)
    return xx, yy, dx, dy


def normalize_signed(values: FloatArray, *, eps: float = 1.0e-12) -> FloatArray:
    """Center and scale a field to the closed interval ``[-1, 1]``."""

    result = np.asarray(values, dtype=np.float64)
    result = result - float(np.mean(result))
    scale = float(np.max(np.abs(result)))
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(result)
    return np.clip(result / scale, -1.0, 1.0)


def _grf(
    rng: np.random.Generator,
    shape: tuple[int, int],
    cutoff: float,
) -> FloatArray:
    noise = rng.standard_normal(shape)
    ky = np.fft.fftfreq(shape[0]) * shape[0]
    kx = np.fft.fftfreq(shape[1]) * shape[1]
    kkx, kky = np.meshgrid(kx, ky)
    radius2 = kkx * kkx + kky * kky
    spectral_filter = np.exp(-0.5 * radius2 / max(cutoff * cutoff, 1.0e-12))
    spectral_filter[0, 0] = 0.0
    return normalize_signed(np.fft.ifft2(np.fft.fft2(noise) * spectral_filter).real)


def _legacy_make_setting_field(
    setting: str,
    resolution: Resolution,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate one of the ten canonical source/coefficient/initial fields."""

    setting = normalize_setting(setting)
    height, width = parse_resolution(resolution)
    xx, yy, _, _ = grid((height, width))
    min_size = min(height, width)

    if setting == "smooth_grf":
        field_values = _grf(rng, (height, width), max(1.5, min_size / 16.0))
    elif setting == "medium_grf":
        field_values = _grf(rng, (height, width), max(2.5, min_size / 8.0))
    elif setting == "rough_grf":
        field_values = _grf(rng, (height, width), max(4.0, min_size / 4.0))
    elif setting == "low_frequency_fourier":
        field_values = np.zeros((height, width), dtype=np.float64)
        for _ in range(5):
            kx, ky = rng.integers(1, 4, size=2)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amplitude = rng.normal() / np.sqrt(float(kx * kx + ky * ky))
            field_values += amplitude * np.sin(2.0 * np.pi * (kx * xx + ky * yy) + phase)
        field_values = normalize_signed(field_values)
    elif setting == "multi_frequency_fourier":
        field_values = np.zeros((height, width), dtype=np.float64)
        max_mode = max(3, min(12, min_size // 2 - 1))
        for _ in range(10):
            kx, ky = rng.integers(1, max_mode + 1, size=2)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amplitude = rng.normal() / np.sqrt(float(kx * kx + ky * ky))
            field_values += amplitude * np.sin(2.0 * np.pi * (kx * xx + ky * yy) + phase)
        field_values = normalize_signed(field_values)
    elif setting == "gaussian_blobs":
        field_values = np.zeros((height, width), dtype=np.float64)
        for _ in range(int(rng.integers(2, 6))):
            cx, cy = rng.uniform(0.12, 0.88, size=2)
            sigma = rng.uniform(0.035, 0.14)
            amplitude = rng.uniform(0.5, 1.5) * rng.choice((-1.0, 1.0))
            field_values += amplitude * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)
            )
        field_values = normalize_signed(field_values)
    elif setting == "piecewise_blocks":
        rows = int(rng.integers(2, 5))
        cols = int(rng.integers(2, 5))
        block_values = rng.uniform(-1.0, 1.0, size=(rows, cols))
        row_ids = np.minimum((np.arange(height) * rows) // height, rows - 1)
        col_ids = np.minimum((np.arange(width) * cols) // width, cols - 1)
        field_values = block_values[row_ids[:, None], col_ids[None, :]]
        field_values = normalize_signed(field_values)
    elif setting == "threshold_level_set":
        base = _grf(rng, (height, width), max(2.0, min_size / 10.0))
        threshold = float(np.quantile(base, rng.uniform(0.4, 0.6)))
        field_values = np.where(base >= threshold, 1.0, -1.0)
        field_values = normalize_signed(field_values)
    elif setting == "dipole_vortex_pair":
        angle = rng.uniform(0.0, 2.0 * np.pi)
        center = rng.uniform(0.35, 0.65, size=2)
        separation = rng.uniform(0.12, 0.25)
        offset = 0.5 * separation * np.array([np.cos(angle), np.sin(angle)])
        sigma = rng.uniform(0.045, 0.10)
        p1, p2 = center + offset, center - offset
        positive = np.exp(-((xx - p1[0]) ** 2 + (yy - p1[1]) ** 2) / (2.0 * sigma**2))
        negative = np.exp(-((xx - p2[0]) ** 2 + (yy - p2[1]) ** 2) / (2.0 * sigma**2))
        field_values = normalize_signed(positive - negative)
    else:  # front_ring_shock
        variant = int(rng.integers(0, 3))
        if variant == 0:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            offset = rng.uniform(-0.2, 0.2)
            signed_distance = np.cos(angle) * (xx - 0.5) + np.sin(angle) * (yy - 0.5) - offset
            width_scale = rng.uniform(0.015, 0.05)
            field_values = np.tanh(signed_distance / width_scale)
        elif variant == 1:
            cx, cy = rng.uniform(0.38, 0.62, size=2)
            radius = rng.uniform(0.18, 0.34)
            width_scale = rng.uniform(0.015, 0.045)
            distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            field_values = np.tanh((radius - distance) / width_scale)
        else:
            location = rng.uniform(0.3, 0.7)
            width_scale = rng.uniform(0.008, 0.025)
            field_values = np.tanh((location - xx) / width_scale)
        field_values = normalize_signed(field_values)

    return np.asarray(field_values, dtype=np.float64)


def make_setting_field(
    setting: str,
    resolution: Resolution,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate one setting through the public, extensible registry."""

    from .. import settings as _built_in_settings  # noqa: F401
    from ..registry import SETTING_REGISTRY

    canonical = normalize_setting(setting)
    shape = parse_resolution(resolution)
    field = np.asarray(SETTING_REGISTRY.create(canonical, shape, rng=rng), dtype=np.float64)
    if field.shape != shape:
        raise RuntimeError(f"setting {canonical!r} returned {field.shape}, expected {shape}")
    if not np.all(np.isfinite(field)):
        raise RuntimeError(f"setting {canonical!r} produced non-finite values")
    return field


def make_geometry(
    boundary: str,
    resolution: Resolution,
    *,
    family: str,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Return a binary solid/wall mask; B3 Navier--Stokes adds an obstacle."""

    boundary = normalize_boundary(boundary)
    family = normalize_family(family)
    height, width = parse_resolution(resolution)
    geometry = np.zeros((height, width), dtype=np.float64)
    if boundary == "periodic":
        return geometry

    geometry[[0, -1], :] = 1.0
    geometry[:, [0, -1]] = 1.0
    if boundary == "robin" and family == "navier_stokes":
        if rng is None:
            rng = make_rng(0, 991)
        xx, yy, _, _ = grid((height, width))
        cx = 0.42 + rng.uniform(-0.035, 0.035)
        cy = 0.50 + rng.uniform(-0.06, 0.06)
        radius = rng.uniform(0.09, 0.14)
        geometry[((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius] = 1.0
    return geometry


def apply_scalar_boundary(
    values: FloatArray,
    boundary: str,
    *,
    value: float = 0.0,
    robin_alpha: float = 1.0,
    robin_beta: float = 0.15,
) -> FloatArray:
    """Apply a compact scalar boundary approximation in place and return it."""

    boundary = normalize_boundary(boundary)
    if boundary == "periodic":
        return values
    if boundary == "dirichlet":
        values[[0, -1], :] = value
        values[:, [0, -1]] = value
    elif boundary == "neumann":
        values[0, :] = values[1, :]
        values[-1, :] = values[-2, :]
        values[:, 0] = values[:, 1]
        values[:, -1] = values[:, -2]
    else:
        # Mixed B3: Dirichlet on horizontal walls and homogeneous Robin on
        # vertical walls.  The cell-width scale is absorbed into alpha.
        values[[0, -1], :] = value
        factor = robin_beta / max(robin_beta + robin_alpha / values.shape[1], 1.0e-12)
        values[:, 0] = factor * values[:, 1]
        values[:, -1] = factor * values[:, -2]
    return values


def apply_velocity_boundary(
    velocity: FloatArray,
    boundary: str,
    *,
    geometry: FloatArray | None = None,
    inflow_speed: float = 0.75,
) -> FloatArray:
    """Apply no-slip, free-slip, periodic, or obstacle/inflow velocity BCs."""

    boundary = normalize_boundary(boundary)
    if velocity.ndim != 3 or velocity.shape[-1] != 2:
        raise ValueError("velocity must have shape [H, W, 2]")
    if boundary == "periodic":
        return velocity
    if boundary == "dirichlet":
        velocity[[0, -1], :, :] = 0.0
        velocity[:, [0, -1], :] = 0.0
    elif boundary == "neumann":
        # Vertical walls: zero normal x velocity, copied tangential velocity.
        velocity[:, 0, 0] = 0.0
        velocity[:, -1, 0] = 0.0
        velocity[:, 0, 1] = velocity[:, 1, 1]
        velocity[:, -1, 1] = velocity[:, -2, 1]
        # Horizontal walls: zero normal y velocity, copied tangential velocity.
        velocity[0, :, 1] = 0.0
        velocity[-1, :, 1] = 0.0
        velocity[0, :, 0] = velocity[1, :, 0]
        velocity[-1, :, 0] = velocity[-2, :, 0]
    else:
        height = velocity.shape[0]
        y = (np.arange(height, dtype=np.float64) + 0.5) / height
        profile = 4.0 * inflow_speed * y * (1.0 - y)
        velocity[[0, -1], :, :] = 0.0
        velocity[:, 0, 0] = profile
        velocity[:, 0, 1] = 0.0
        velocity[:, -1, :] = velocity[:, -2, :]
        if geometry is not None:
            solid = np.asarray(geometry).squeeze(-1) > 0.5 if geometry.ndim == 3 else geometry > 0.5
            # Keep the inflow and outflow open; only interior B3 geometry is an obstacle.
            solid = solid.copy()
            solid[:, [0, -1]] = False
            velocity[solid, :] = 0.0
    return velocity


def _neighbors(
    values: FloatArray, boundary: str
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    boundary = normalize_boundary(boundary)
    if boundary == "periodic":
        east = np.roll(values, -1, axis=1)
        west = np.roll(values, 1, axis=1)
        north = np.roll(values, -1, axis=0)
        south = np.roll(values, 1, axis=0)
        return east, west, north, south
    padded = np.pad(values, ((1, 1), (1, 1)), mode="edge")
    east = padded[1:-1, 2:]
    west = padded[1:-1, :-2]
    north = padded[2:, 1:-1]
    south = padded[:-2, 1:-1]
    return east, west, north, south


def laplacian(values: FloatArray, boundary: str, dx: float, dy: float) -> FloatArray:
    east, west, north, south = _neighbors(values, boundary)
    return (east - 2.0 * values + west) / (dx * dx) + (north - 2.0 * values + south) / (dy * dy)


def gradient(
    values: FloatArray, boundary: str, dx: float, dy: float
) -> tuple[FloatArray, FloatArray]:
    east, west, north, south = _neighbors(values, boundary)
    return (east - west) / (2.0 * dx), (north - south) / (2.0 * dy)


def spectral_diffuse(
    values: FloatArray, diffusivity: float, dt: float, dx: float, dy: float
) -> FloatArray:
    """Apply one stable periodic spectral diffusion step.

    Bounded solvers use this as an efficient interior approximation and then
    re-impose their family-conditioned boundary protocol.
    """

    height, width = values.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
    kkx, kky = np.meshgrid(kx, ky)
    decay = np.exp(-max(float(diffusivity), 0.0) * max(float(dt), 0.0) * (kkx**2 + kky**2))
    return np.fft.ifft2(np.fft.fft2(values) * decay).real


def semi_lagrangian(
    values: FloatArray,
    velocity_x: FloatArray,
    velocity_y: FloatArray,
    dt: float,
    dx: float,
    dy: float,
    boundary: str,
) -> FloatArray:
    """Bilinearly sample a scalar field at back-traced grid locations."""

    height, width = values.shape
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    source_x = cols - dt * velocity_x / dx
    source_y = rows - dt * velocity_y / dy
    if normalize_boundary(boundary) == "periodic":
        source_x %= width
        source_y %= height
        # Floating-point remainder can round a tiny negative coordinate to the
        # modulus itself; keep floor-based indices strictly inside the grid.
        source_x = np.minimum(source_x, np.nextafter(float(width), -np.inf))
        source_y = np.minimum(source_y, np.nextafter(float(height), -np.inf))
    else:
        source_x = np.clip(source_x, 0.0, width - 1.0)
        source_y = np.clip(source_y, 0.0, height - 1.0)
    x0 = np.floor(source_x).astype(np.int64)
    y0 = np.floor(source_y).astype(np.int64)
    if normalize_boundary(boundary) == "periodic":
        x1, y1 = (x0 + 1) % width, (y0 + 1) % height
    else:
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = source_x - x0, source_y - y0
    return (
        (1.0 - wx) * (1.0 - wy) * values[y0, x0]
        + wx * (1.0 - wy) * values[y0, x1]
        + (1.0 - wx) * wy * values[y1, x0]
        + wx * wy * values[y1, x1]
    )


def solve_poisson_like(
    source: FloatArray,
    boundary: str,
    dx: float,
    dy: float,
    *,
    coefficient: FloatArray | None = None,
    iterations: int = 160,
    relaxation: float = 0.78,
) -> FloatArray:
    """Weighted Jacobi solve for ``-div(a grad(u)) = source``."""

    boundary = normalize_boundary(boundary)
    if not isinstance(iterations, Integral) or int(iterations) < 1:
        raise ValueError("iterations must be a positive integer")
    rhs = np.asarray(source, dtype=np.float64).copy()
    if boundary in {"periodic", "neumann"}:
        rhs -= float(np.mean(rhs))
    if coefficient is None:
        coefficient = np.ones_like(rhs)
    else:
        coefficient = np.maximum(np.asarray(coefficient, dtype=np.float64), 1.0e-6)
    solution = np.zeros_like(rhs)
    inv_dx2, inv_dy2 = 1.0 / (dx * dx), 1.0 / (dy * dy)
    relaxation = float(np.clip(relaxation, 0.05, 1.0))

    for _ in range(int(iterations)):
        ue, uw, un, us = _neighbors(solution, boundary)
        ae, aw, an, ass = _neighbors(coefficient, boundary)
        ae = 0.5 * (coefficient + ae)
        aw = 0.5 * (coefficient + aw)
        an = 0.5 * (coefficient + an)
        ass = 0.5 * (coefficient + ass)
        denominator = (ae + aw) * inv_dx2 + (an + ass) * inv_dy2
        candidate = (
            (ae * ue + aw * uw) * inv_dx2 + (an * un + ass * us) * inv_dy2 + rhs
        ) / np.maximum(denominator, 1.0e-12)
        solution = (1.0 - relaxation) * solution + relaxation * candidate
        if boundary in {"periodic", "neumann"}:
            solution -= float(np.mean(solution))
        apply_scalar_boundary(solution, boundary)
    return solution


def stream_velocity(vorticity: FloatArray, dx: float, dy: float) -> tuple[FloatArray, FloatArray]:
    """Recover a periodic divergence-free velocity from scalar vorticity."""

    height, width = vorticity.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
    kkx, kky = np.meshgrid(kx, ky)
    wave2 = kkx**2 + kky**2
    omega_hat = np.fft.fft2(vorticity - float(np.mean(vorticity)))
    psi_hat = np.zeros_like(omega_hat, dtype=np.complex128)
    nonzero = wave2 > 0.0
    psi_hat[nonzero] = omega_hat[nonzero] / wave2[nonzero]
    velocity_x = np.fft.ifft2(1j * kky * psi_hat).real
    velocity_y = np.fft.ifft2(-1j * kkx * psi_hat).real
    return velocity_x, velocity_y


def _safe_array(values: FloatArray, dtype: np.dtype[Any] | type[np.floating[Any]]) -> FloatArray:
    target_dtype = np.dtype(dtype)
    if not np.issubdtype(target_dtype, np.floating):
        raise TypeError("dtype must be a floating-point dtype")
    limit = min(float(np.finfo(target_dtype).max) / 16.0, 1.0e12)
    result = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=limit, neginf=-limit
    )
    return np.ascontiguousarray(np.clip(result, -limit, limit), dtype=target_dtype)


@dataclass(frozen=True, slots=True)
class PDEOutput:
    """One unbatched generated PDE instance in the canonical array layout.

    ``condition`` has shape ``[H, W, V_cond]``, ``trajectory`` has shape
    ``[T, H, W, V_state]``, and ``geometry`` has shape ``[H, W, 1]``.
    """

    family: str
    boundary: str
    setting: str
    regime: str
    seed: int
    condition: FloatArray
    trajectory: FloatArray
    geometry: FloatArray
    parameters: Mapping[str, float | int] = field(default_factory=dict)
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            family = normalize_family(self.family)
        except ValueError:
            # Runtime-registered PDE plugins may introduce a new family name
            # while retaining this common output container.
            family = _token(self.family)
        boundary = normalize_boundary(self.boundary)
        setting = normalize_setting(self.setting)
        regime = normalize_regime(self.regime)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "setting", setting)
        object.__setattr__(self, "regime", regime)
        if self.condition.ndim != 3:
            raise ValueError("condition must have shape [H, W, V_cond]")
        if self.trajectory.ndim != 4:
            raise ValueError("trajectory must have shape [T, H, W, V_state]")
        if self.geometry.ndim != 3 or self.geometry.shape[-1] != 1:
            raise ValueError("geometry must have shape [H, W, 1]")
        spatial = self.condition.shape[:2]
        if self.trajectory.shape[1:3] != spatial or self.geometry.shape[:2] != spatial:
            raise ValueError("condition, trajectory, and geometry spatial shapes must match")
        for name, array in (
            ("condition", self.condition),
            ("trajectory", self.trajectory),
            ("geometry", self.geometry),
        ):
            if not np.issubdtype(array.dtype, np.floating):
                raise TypeError(f"{name} must use a floating-point dtype")
            if not bool(np.all(np.isfinite(array))):
                raise ValueError(f"{name} contains non-finite values")

    @property
    def time_steps(self) -> int:
        return int(self.trajectory.shape[0])

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.trajectory.shape[1]), int(self.trajectory.shape[2])

    def as_dict(self, *, copy: bool = False) -> dict[str, Any]:
        """Return a writer-friendly mapping, optionally copying array storage."""

        arrays = (self.condition, self.trajectory, self.geometry)
        condition, trajectory, geometry = (array.copy() for array in arrays) if copy else arrays
        return {
            "family": self.family,
            "boundary": self.boundary,
            "setting": self.setting,
            "regime": self.regime,
            "seed": self.seed,
            "condition": condition,
            "trajectory": trajectory,
            "geometry": geometry,
            "parameters": dict(self.parameters),
            "diagnostics": dict(self.diagnostics),
        }


def build_output(
    *,
    family: str,
    boundary: str,
    setting: str,
    regime: str,
    seed: int,
    condition: FloatArray,
    trajectory: FloatArray,
    geometry: FloatArray,
    parameters: Mapping[str, float | int],
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> PDEOutput:
    """Sanitize arrays, attach lightweight diagnostics, and validate layout."""

    condition_array = _safe_array(condition, dtype)
    trajectory_array = _safe_array(trajectory, dtype)
    geometry_array = _safe_array(geometry, dtype)
    diagnostics = {
        "condition_min": float(np.min(condition_array)),
        "condition_max": float(np.max(condition_array)),
        "trajectory_min": float(np.min(trajectory_array)),
        "trajectory_max": float(np.max(trajectory_array)),
        "trajectory_rms": float(np.sqrt(np.mean(np.square(trajectory_array, dtype=np.float64)))),
        "solid_fraction": float(np.mean(geometry_array > 0.5)),
    }
    return PDEOutput(
        family=family,
        boundary=boundary,
        setting=setting,
        regime=regime,
        seed=int(seed),
        condition=condition_array,
        trajectory=trajectory_array,
        geometry=geometry_array,
        parameters=dict(parameters),
        diagnostics=diagnostics,
    )


def add_channel(values: FloatArray) -> FloatArray:
    """Convert ``[H, W]`` to the canonical ``[H, W, 1]`` layout."""

    return np.asarray(values)[..., None]
