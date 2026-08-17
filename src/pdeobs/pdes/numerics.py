"""Boundary-consistent numerical kernels used by the dataset generators.

The routines in this module deliberately separate periodic Fourier solvers
from bounded finite-difference/finite-volume solvers.  A periodic update must
never be followed by a cosmetic boundary overwrite: the boundary protocol is
part of the discrete operator that is solved.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.fft import dstn, idstn
from scipy.sparse import linalg as spla

from .common import (
    apply_scalar_boundary,
    apply_velocity_boundary,
    gradient,
    normalize_boundary,
    semi_lagrangian,
)

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class SolverInfo:
    """Convergence information stored with every generated sample."""

    converged: bool
    iterations: int
    relative_residual: float


def _bounded_embed(
    interior: np.ndarray,
    shape: tuple[int, int],
    boundary: str,
    *,
    robin_alpha: float = 1.0,
    robin_beta: float = 0.15,
) -> np.ndarray:
    values = np.zeros(shape, dtype=np.float64)
    values[1:-1, 1:-1] = interior.reshape(shape[0] - 2, shape[1] - 2)
    boundary = normalize_boundary(boundary)
    if boundary == "dirichlet":
        return values
    if boundary == "neumann":
        values[0, 1:-1] = values[1, 1:-1]
        values[-1, 1:-1] = values[-2, 1:-1]
        values[1:-1, 0] = values[1:-1, 1]
        values[1:-1, -1] = values[1:-1, -2]
        values[0, 0] = values[1, 1]
        values[0, -1] = values[1, -2]
        values[-1, 0] = values[-2, 1]
        values[-1, -1] = values[-2, -2]
        return values
    if boundary != "robin":
        raise ValueError(f"bounded embed does not support {boundary!r}")
    # B3 scalar protocol: horizontal Dirichlet and vertical homogeneous Robin.
    factor = robin_beta / max(robin_beta + robin_alpha / shape[1], 1.0e-12)
    values[1:-1, 0] = factor * values[1:-1, 1]
    values[1:-1, -1] = factor * values[1:-1, -2]
    return values


def _variable_operator(
    values: np.ndarray,
    coefficient: np.ndarray,
    boundary: str,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Return ``-div(a grad(values))`` with arithmetic face coefficients."""

    boundary = normalize_boundary(boundary)
    if boundary == "periodic":
        east = np.roll(values, -1, axis=1)
        west = np.roll(values, 1, axis=1)
        north = np.roll(values, -1, axis=0)
        south = np.roll(values, 1, axis=0)
        ae = 0.5 * (coefficient + np.roll(coefficient, -1, axis=1))
        aw = 0.5 * (coefficient + np.roll(coefficient, 1, axis=1))
        an = 0.5 * (coefficient + np.roll(coefficient, -1, axis=0))
        ass = 0.5 * (coefficient + np.roll(coefficient, 1, axis=0))
    else:
        pad_u = np.pad(values, 1, mode="edge")
        pad_a = np.pad(coefficient, 1, mode="edge")
        east, west = pad_u[1:-1, 2:], pad_u[1:-1, :-2]
        north, south = pad_u[2:, 1:-1], pad_u[:-2, 1:-1]
        ae = 0.5 * (coefficient + pad_a[1:-1, 2:])
        aw = 0.5 * (coefficient + pad_a[1:-1, :-2])
        an = 0.5 * (coefficient + pad_a[2:, 1:-1])
        ass = 0.5 * (coefficient + pad_a[:-2, 1:-1])
    return -(
        (ae * (east - values) - aw * (values - west)) / (dx * dx)
        + (an * (north - values) - ass * (values - south)) / (dy * dy)
    )


def _elliptic_matrix(
    coefficient: np.ndarray,
    boundary: str,
    dx: float,
    dy: float,
    reaction: float,
) -> sparse.csr_matrix:
    """Assemble the symmetric face-flux operator with the BC in its stencil."""

    height, width = coefficient.shape
    periodic = boundary == "periodic"
    if periodic:
        rows_count, cols_count = height, width
    else:
        rows_count, cols_count = height - 2, width - 2
        if rows_count < 1 or cols_count < 1:
            raise ValueError("bounded elliptic solve needs at least one interior cell")

    def unknown(row: int, col: int) -> int:
        if periodic:
            return (row % height) * width + (col % width)
        return (row - 1) * cols_count + (col - 1)

    row_entries: list[int] = []
    col_entries: list[int] = []
    values: list[float] = []
    robin_factor = 0.15 / max(0.15 + 1.0 / width, 1.0e-12)
    row_range = range(height) if periodic else range(1, height - 1)
    col_range = range(width) if periodic else range(1, width - 1)
    for row in row_range:
        for col in col_range:
            center = unknown(row, col)
            diagonal = float(reaction)
            for dr, dc, spacing in (
                (0, -1, dx),
                (0, 1, dx),
                (-1, 0, dy),
                (1, 0, dy),
            ):
                neighbor_row, neighbor_col = row + dr, col + dc
                if periodic:
                    wrapped_row, wrapped_col = neighbor_row % height, neighbor_col % width
                    face = (
                        0.5
                        * (coefficient[row, col] + coefficient[wrapped_row, wrapped_col])
                        / (spacing * spacing)
                    )
                    diagonal += face
                    row_entries.append(center)
                    col_entries.append(unknown(wrapped_row, wrapped_col))
                    values.append(-face)
                    continue

                face = (
                    0.5
                    * (coefficient[row, col] + coefficient[neighbor_row, neighbor_col])
                    / (spacing * spacing)
                )
                neighbor_is_interior = (
                    1 <= neighbor_row < height - 1 and 1 <= neighbor_col < width - 1
                )
                if neighbor_is_interior:
                    diagonal += face
                    row_entries.append(center)
                    col_entries.append(unknown(neighbor_row, neighbor_col))
                    values.append(-face)
                elif boundary == "dirichlet":
                    diagonal += face
                elif boundary == "neumann":
                    # Homogeneous normal derivative: the ghost/boundary value
                    # equals this adjacent interior unknown, so this face adds zero.
                    continue
                elif boundary == "robin":
                    # B3 has horizontal Dirichlet walls and homogeneous Robin
                    # on the vertical sides, matching apply_scalar_boundary.
                    diagonal += face if dr else face * (1.0 - robin_factor)
                else:  # pragma: no cover - normalize_boundary rejects this earlier
                    raise ValueError(f"unsupported bounded boundary {boundary!r}")
            row_entries.append(center)
            col_entries.append(center)
            values.append(diagonal)

    size = rows_count * cols_count
    return sparse.csr_matrix((values, (row_entries, col_entries)), shape=(size, size))


def solve_elliptic(
    source: FloatArray,
    boundary: str,
    dx: float,
    dy: float,
    *,
    coefficient: FloatArray | None = None,
    reaction: float = 0.0,
    rtol: float = 1.0e-9,
    maxiter: int = 4000,
) -> tuple[np.ndarray, SolverInfo]:
    """Solve ``-div(a grad(u)) + reaction*u = source``.

    Periodic and homogeneous-Neumann Poisson problems are solved in their
    zero-mean gauge.  Dirichlet/Neumann/Robin values are reconstructed from
    the interior unknowns, so the bounded operator actually contains the BC.
    """

    rhs_full = np.asarray(source, dtype=np.float64)
    shape = rhs_full.shape
    boundary = normalize_boundary(boundary)
    if coefficient is None:
        coeff = np.ones(shape, dtype=np.float64)
    else:
        coeff = np.asarray(coefficient, dtype=np.float64)
        if coeff.shape != shape or np.any(~np.isfinite(coeff)) or np.any(coeff <= 0.0):
            raise ValueError("elliptic coefficient must be finite, positive, and match source")
    if rtol <= 0.0 or not np.isfinite(rtol):
        raise ValueError("rtol must be finite and positive")
    if maxiter < 1:
        raise ValueError("maxiter must be positive")

    periodic = boundary == "periodic"
    if periodic:
        rhs = rhs_full.copy()
        if reaction == 0.0:
            rhs -= float(np.mean(rhs))
        vector_size = rhs.size

        def unpack(vector: np.ndarray) -> np.ndarray:
            return vector.reshape(shape)

        def pack(values: np.ndarray) -> np.ndarray:
            return values.reshape(-1)

    else:
        rhs = rhs_full[1:-1, 1:-1].copy()
        if boundary == "neumann" and reaction == 0.0:
            rhs -= float(np.mean(rhs))
        vector_size = rhs.size

        def unpack(vector: np.ndarray) -> np.ndarray:
            return _bounded_embed(vector, shape, boundary)

        def pack(values: np.ndarray) -> np.ndarray:
            return values[1:-1, 1:-1].reshape(-1)

    nullspace = reaction == 0.0 and boundary in {"periodic", "neumann"}
    matrix = _elliptic_matrix(coeff, boundary, dx, dy, reaction)
    if matrix.shape != (vector_size, vector_size):  # pragma: no cover - invariant
        raise RuntimeError("elliptic matrix size does not match packed unknowns")
    if nullspace:
        # Add a rank-one mean constraint without making the sparse matrix dense.
        operator: sparse.spmatrix | spla.LinearOperator = spla.LinearOperator(
            matrix.shape,
            matvec=lambda vector: matrix @ vector + float(np.mean(vector)),
            dtype=np.dtype(np.float64),
        )
    else:
        operator = matrix
    diagonal = np.maximum(np.abs(matrix.diagonal()), 1.0e-12)
    preconditioner = spla.LinearOperator(
        (vector_size, vector_size), matvec=lambda x: x / diagonal, dtype=np.dtype(np.float64)
    )
    iterations = 0

    def callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    rhs_vector = rhs.reshape(-1)
    if reaction < 0.0:
        solution_vector, status = spla.minres(
            operator,
            rhs_vector,
            M=preconditioner,
            rtol=rtol,
            maxiter=maxiter,
            callback=callback,
            check=False,
        )
    else:
        solution_vector, status = spla.cg(
            operator,
            rhs_vector,
            M=preconditioner,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            callback=callback,
        )
    solution = unpack(np.asarray(solution_vector, dtype=np.float64))
    if nullspace:
        solution -= float(np.mean(solution))
        if not periodic:
            solution = _bounded_embed(solution[1:-1, 1:-1], shape, boundary)
    residual = matrix @ np.asarray(solution_vector, dtype=np.float64)
    relative = float(
        np.linalg.norm(residual - rhs_vector)
        / max(np.linalg.norm(rhs_vector), np.finfo(np.float64).eps)
    )
    info = SolverInfo(status == 0 and np.isfinite(relative), iterations, relative)
    if not info.converged:
        raise RuntimeError(
            f"elliptic solver failed: status={status}, iterations={iterations}, "
            f"relative_residual={relative:.6g}"
        )
    return solution, info


def crank_nicolson_diffusion(
    values: FloatArray,
    diffusivity: float,
    dt: float,
    dx: float,
    dy: float,
    boundary: str,
    *,
    rtol: float = 1.0e-9,
) -> tuple[np.ndarray, SolverInfo]:
    """Advance diffusion with Fourier exactness or bounded Crank--Nicolson."""

    state = np.asarray(values, dtype=np.float64)
    boundary = normalize_boundary(boundary)
    if diffusivity < 0.0 or dt < 0.0:
        raise ValueError("diffusivity and dt must be non-negative")
    if dt == 0.0 or diffusivity == 0.0:
        return state.copy(), SolverInfo(True, 0, 0.0)
    if boundary == "periodic":
        height, width = state.shape
        ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
        kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
        kkx, kky = np.meshgrid(kx, ky)
        wave2 = kkx**2 + kky**2
        result = np.fft.ifft2(np.fft.fft2(state) * np.exp(-diffusivity * dt * wave2)).real
        return result, SolverInfo(True, 1, 0.0)
    operator_at_old = _variable_operator(state, np.ones_like(state), boundary, dx, dy)
    rhs = state - 0.5 * diffusivity * dt * operator_at_old
    result, info = solve_elliptic(
        rhs,
        boundary,
        dx,
        dy,
        coefficient=np.full_like(state, 0.5 * diffusivity * dt),
        reaction=1.0,
        rtol=rtol,
    )
    apply_scalar_boundary(result, boundary)
    return result, info


def advance_reaction_diffusion(
    values: FloatArray,
    diffusivity: float,
    reaction_rate: float,
    frame_dt: float,
    dx: float,
    dy: float,
    boundary: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Advance one saved Allen--Cahn interval with Strang splitting."""

    state = np.asarray(values, dtype=np.float64).copy()
    boundary = normalize_boundary(boundary)
    if diffusivity < 0.0 or reaction_rate < 0.0 or frame_dt < 0.0:
        raise ValueError("reaction-diffusion coefficients and time must be non-negative")
    substeps = max(1, int(np.ceil(reaction_rate * frame_dt / 0.18)))
    dt = frame_dt / substeps
    maximum_iterations = 0
    maximum_residual = 0.0
    for _ in range(substeps):
        half_decay = np.exp(-reaction_rate * dt)
        denominator = np.sqrt(
            np.maximum(
                state**2 + (1.0 - state**2) * half_decay,
                np.finfo(np.float64).eps,
            )
        )
        state = state / denominator
        state, solver = crank_nicolson_diffusion(state, diffusivity, dt, dx, dy, boundary)
        maximum_iterations = max(maximum_iterations, solver.iterations)
        maximum_residual = max(maximum_residual, solver.relative_residual)
        denominator = np.sqrt(
            np.maximum(
                state**2 + (1.0 - state**2) * half_decay,
                np.finfo(np.float64).eps,
            )
        )
        state = state / denominator
        apply_scalar_boundary(state, boundary)
    return state, {
        "substeps": substeps,
        "linear_solver_iterations_max": maximum_iterations,
        "linear_solver_relative_residual_max": maximum_residual,
    }


def _spectral_advection(values: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = values.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
    kkx, kky = np.meshgrid(kx, ky)
    transform = np.fft.fft2(values)
    cutoff_x, cutoff_y = width // 3, height // 3
    modes_x = np.fft.fftfreq(width) * width
    modes_y = np.fft.fftfreq(height) * height
    dealias = (np.abs(modes_y)[:, None] <= cutoff_y) & (np.abs(modes_x)[None, :] <= cutoff_x)
    grad_x = np.fft.ifft2(1j * kkx * transform).real
    grad_y = np.fft.ifft2(1j * kky * transform).real
    nonlinear = values * (grad_x + grad_y)
    return np.fft.ifft2(np.fft.fft2(nonlinear) * dealias).real


def _rusanov_burgers_advection(
    values: np.ndarray, dx: float, dy: float, boundary: str
) -> np.ndarray:
    """Return conservative Burgers advection using Rusanov face fluxes."""

    state = np.asarray(values, dtype=np.float64)
    if boundary == "periodic":
        center = state
        east = np.roll(state, -1, axis=-1)
        west = np.roll(state, 1, axis=-1)
        north = np.roll(state, -1, axis=-2)
        south = np.roll(state, 1, axis=-2)
    else:
        padded = np.pad(state, 1, mode="edge")
        center = padded[1:-1, 1:-1]
        east, west = padded[1:-1, 2:], padded[1:-1, :-2]
        north, south = padded[2:, 1:-1], padded[:-2, 1:-1]

    def flux(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        physical = 0.25 * (left * left + right * right)
        speed = np.maximum(np.abs(left), np.abs(right))
        return physical - 0.5 * speed * (right - left)

    east_flux = flux(center, east)
    west_flux = flux(west, center)
    north_flux = flux(center, north)
    south_flux = flux(south, center)
    return -((east_flux - west_flux) / dx + (north_flux - south_flux) / dy)


def advance_burgers(
    values: FloatArray,
    viscosity: float,
    frame_dt: float,
    dx: float,
    dy: float,
    boundary: str,
    *,
    advection_scheme: str | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Advance the 2-D scalar Burgers extension with BC-consistent splitting."""

    state = np.asarray(values, dtype=np.float64).copy()
    boundary = normalize_boundary(boundary)
    scheme = str(advection_scheme or ("spectral" if boundary == "periodic" else "rusanov")).lower()
    if scheme not in {"spectral", "rusanov"}:
        raise ValueError("advection_scheme must be spectral or rusanov")
    if scheme == "spectral" and boundary != "periodic":
        raise ValueError("spectral Burgers advection requires a periodic boundary")
    if viscosity < 0.0 or frame_dt < 0.0:
        raise ValueError("viscosity and frame_dt must be non-negative")
    remaining = float(frame_dt)
    substeps = 0
    max_courant = 0.0
    maximum_diffusion_iterations = 0
    while remaining > np.finfo(np.float64).eps * max(1.0, frame_dt):
        speed = max(float(np.max(np.abs(state))), 1.0e-12)
        stable_dt = 0.35 / (speed * (1.0 / dx + 1.0 / dy))
        dt = min(remaining, stable_dt)
        courant = speed * dt * (1.0 / dx + 1.0 / dy)
        max_courant = max(max_courant, courant)
        if scheme == "spectral":
            k1 = -_spectral_advection(state, dx, dy)
            predictor = state + dt * k1
            k2 = -_spectral_advection(predictor, dx, dy)
            state = state + 0.5 * dt * (k1 + k2)
        else:
            k1 = _rusanov_burgers_advection(state, dx, dy, boundary)
            predictor = state + dt * k1
            apply_scalar_boundary(predictor, boundary)
            k2 = _rusanov_burgers_advection(predictor, dx, dy, boundary)
            state = state + 0.5 * dt * (k1 + k2)
            apply_scalar_boundary(state, boundary)
        state, info = crank_nicolson_diffusion(state, viscosity, dt, dx, dy, boundary)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Burgers integrator produced non-finite values")
        remaining = max(0.0, remaining - dt)
        substeps += 1
        if substeps > 10000:
            raise RuntimeError("Burgers CFL controller exceeded 10,000 substeps")
        maximum_diffusion_iterations = max(maximum_diffusion_iterations, info.iterations)
    return state, {
        "substeps": substeps,
        "max_courant": max_courant,
        "max_diffusion_iterations": maximum_diffusion_iterations,
    }


def advance_periodic_vorticity(
    vorticity: FloatArray,
    forcing: FloatArray,
    viscosity: float,
    frame_dt: float,
    dx: float,
    dy: float,
    *,
    internal_dt: float = 1.0e-4,
) -> tuple[np.ndarray, int]:
    """FNO-style 2/3-dealiased pseudospectral CN vorticity update."""

    omega = np.asarray(vorticity, dtype=np.float64)
    height, width = omega.shape
    steps = max(1, int(np.ceil(frame_dt / internal_dt)))
    dt = frame_dt / steps
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
    kkx, kky = np.meshgrid(kx, ky)
    wave2 = kkx**2 + kky**2
    safe_wave2 = wave2.copy()
    safe_wave2[0, 0] = 1.0
    modes_x = np.fft.fftfreq(width) * width
    modes_y = np.fft.fftfreq(height) * height
    dealias = (np.abs(modes_y)[:, None] <= height // 3) & (np.abs(modes_x)[None, :] <= width // 3)
    omega_hat = np.fft.fft2(omega)
    forcing_hat = np.fft.fft2(np.asarray(forcing, dtype=np.float64))
    for _ in range(steps):
        psi_hat = omega_hat / safe_wave2
        psi_hat[0, 0] = 0.0
        velocity_x = np.fft.ifft2(1j * kky * psi_hat).real
        velocity_y = np.fft.ifft2(-1j * kkx * psi_hat).real
        omega_x = np.fft.ifft2(1j * kkx * omega_hat).real
        omega_y = np.fft.ifft2(1j * kky * omega_hat).real
        nonlinear_hat = np.fft.fft2(velocity_x * omega_x + velocity_y * omega_y)
        nonlinear_hat *= dealias
        numerator = (
            -dt * nonlinear_hat
            + dt * forcing_hat
            + (1.0 - 0.5 * dt * viscosity * wave2) * omega_hat
        )
        omega_hat = numerator / (1.0 + 0.5 * dt * viscosity * wave2)
    result = np.fft.ifft2(omega_hat).real
    return result - float(np.mean(result)), steps


def solve_bounded_streamfunction(vorticity: FloatArray, dx: float, dy: float) -> np.ndarray:
    """Solve ``-laplace(psi)=omega`` with homogeneous wall streamfunction.

    The type-I sine transform diagonalizes the exact second-order Dirichlet
    operator on the rectangular interior.  This is both faster and more
    reproducible than rebuilding a Krylov matrix at every flow substep.
    """

    omega = np.asarray(vorticity, dtype=np.float64)
    height, width = omega.shape
    if height < 4 or width < 4:
        raise ValueError("bounded streamfunction solve requires at least 4x4 nodes")
    interior = omega[1:-1, 1:-1]
    modes_y = np.arange(1, height - 1, dtype=np.float64)
    modes_x = np.arange(1, width - 1, dtype=np.float64)
    eigen_y = 4.0 * np.sin(np.pi * modes_y / (2.0 * (height - 1))) ** 2 / dy**2
    eigen_x = 4.0 * np.sin(np.pi * modes_x / (2.0 * (width - 1))) ** 2 / dx**2
    transformed = dstn(interior, type=1, norm="ortho")
    psi = np.zeros_like(omega)
    psi[1:-1, 1:-1] = idstn(
        transformed / (eigen_y[:, None] + eigen_x[None, :]),
        type=1,
        norm="ortho",
    )
    return psi


def bounded_velocity_from_vorticity(
    vorticity: FloatArray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return divergence-compatible velocity and streamfunction on a rectangle."""

    psi = solve_bounded_streamfunction(vorticity, dx, dy)
    velocity_x = np.zeros_like(psi)
    velocity_y = np.zeros_like(psi)
    velocity_x[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dy)
    velocity_y[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)
    return velocity_x, velocity_y, psi


def apply_vorticity_wall_boundary(
    vorticity: np.ndarray,
    streamfunction: np.ndarray,
    boundary: str,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Apply no-slip Thom or free-slip vorticity boundary values."""

    boundary = normalize_boundary(boundary)
    if boundary == "neumann":
        vorticity[[0, -1], :] = 0.0
        vorticity[:, [0, -1]] = 0.0
        return vorticity
    if boundary != "dirichlet":
        raise ValueError("bounded vorticity rectangle supports dirichlet or neumann")
    vorticity[0, 1:-1] = -2.0 * streamfunction[1, 1:-1] / dy**2
    vorticity[-1, 1:-1] = -2.0 * streamfunction[-2, 1:-1] / dy**2
    vorticity[1:-1, 0] = -2.0 * streamfunction[1:-1, 1] / dx**2
    vorticity[1:-1, -1] = -2.0 * streamfunction[1:-1, -2] / dx**2
    vorticity[0, 0] = 0.5 * (vorticity[0, 1] + vorticity[1, 0])
    vorticity[0, -1] = 0.5 * (vorticity[0, -2] + vorticity[1, -1])
    vorticity[-1, 0] = 0.5 * (vorticity[-1, 1] + vorticity[-2, 0])
    vorticity[-1, -1] = 0.5 * (vorticity[-1, -2] + vorticity[-2, -1])
    return vorticity


def _bounded_vorticity_rhs(
    vorticity: np.ndarray,
    viscosity: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    velocity_x, velocity_y, psi = bounded_velocity_from_vorticity(vorticity, dx, dy)
    omega_x = np.zeros_like(vorticity)
    omega_y = np.zeros_like(vorticity)
    laplace_omega = np.zeros_like(vorticity)
    omega_x[1:-1, 1:-1] = (vorticity[1:-1, 2:] - vorticity[1:-1, :-2]) / (2.0 * dx)
    omega_y[1:-1, 1:-1] = (vorticity[2:, 1:-1] - vorticity[:-2, 1:-1]) / (2.0 * dy)
    laplace_omega[1:-1, 1:-1] = (
        vorticity[1:-1, 2:] - 2.0 * vorticity[1:-1, 1:-1] + vorticity[1:-1, :-2]
    ) / dx**2 + (vorticity[2:, 1:-1] - 2.0 * vorticity[1:-1, 1:-1] + vorticity[:-2, 1:-1]) / dy**2
    rhs = -(velocity_x * omega_x + velocity_y * omega_y) + viscosity * laplace_omega
    speed = float(np.max(np.hypot(velocity_x, velocity_y)))
    return rhs, speed, psi


def advance_bounded_vorticity(
    vorticity: FloatArray,
    viscosity: float,
    frame_dt: float,
    dx: float,
    dy: float,
    boundary: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Advance rectangular wall flow with SSP-RK2 vorticity--streamfunction."""

    boundary = normalize_boundary(boundary)
    if boundary not in {"dirichlet", "neumann"}:
        raise ValueError("bounded vorticity integrator requires dirichlet or neumann")
    omega = np.asarray(vorticity, dtype=np.float64).copy()
    _, initial_speed, psi = _bounded_vorticity_rhs(omega, viscosity, dx, dy)
    apply_vorticity_wall_boundary(omega, psi, boundary, dx, dy)
    advective_dt = 0.30 * min(dx, dy) / max(initial_speed, 1.0e-8)
    diffusive_dt = 0.20 * min(dx, dy) ** 2 / max(viscosity, 1.0e-12)
    substeps = max(1, int(np.ceil(frame_dt / min(advective_dt, diffusive_dt))))
    dt = frame_dt / substeps
    maximum_courant = 0.0
    for _ in range(substeps):
        rhs0, speed0, _ = _bounded_vorticity_rhs(omega, viscosity, dx, dy)
        predictor = omega + dt * rhs0
        _, _, predictor_psi = _bounded_vorticity_rhs(predictor, viscosity, dx, dy)
        apply_vorticity_wall_boundary(predictor, predictor_psi, boundary, dx, dy)
        rhs1, speed1, _ = _bounded_vorticity_rhs(predictor, viscosity, dx, dy)
        omega += 0.5 * dt * (rhs0 + rhs1)
        _, _, psi = _bounded_vorticity_rhs(omega, viscosity, dx, dy)
        apply_vorticity_wall_boundary(omega, psi, boundary, dx, dy)
        maximum_courant = max(maximum_courant, max(speed0, speed1) * dt / min(dx, dy))
    velocity_x, velocity_y, _ = bounded_velocity_from_vorticity(omega, dx, dy)
    du_dx = np.zeros_like(omega)
    dv_dy = np.zeros_like(omega)
    du_dx[1:-1, 1:-1] = (velocity_x[1:-1, 2:] - velocity_x[1:-1, :-2]) / (2.0 * dx)
    dv_dy[1:-1, 1:-1] = (velocity_y[2:, 1:-1] - velocity_y[:-2, 1:-1]) / (2.0 * dy)
    divergence = du_dx + dv_dy
    scale = np.sqrt(np.mean(du_dx[1:-1, 1:-1] ** 2)) + np.sqrt(np.mean(dv_dy[1:-1, 1:-1] ** 2))
    return omega, {
        "substeps": substeps,
        "max_courant": maximum_courant,
        "divergence_loss_normalized_solver": float(
            np.sqrt(np.mean(divergence[1:-1, 1:-1] ** 2)) / max(float(scale), np.finfo(float).eps)
        ),
    }


def _cell_to_faces(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = velocity.shape
    u = np.empty((height, width + 1), dtype=np.float64)
    v = np.empty((height + 1, width), dtype=np.float64)
    u[:, 1:-1] = 0.5 * (velocity[:, :-1, 0] + velocity[:, 1:, 0])
    u[:, 0], u[:, -1] = velocity[:, 0, 0], velocity[:, -1, 0]
    v[1:-1, :] = 0.5 * (velocity[:-1, :, 1] + velocity[1:, :, 1])
    v[0, :], v[-1, :] = velocity[0, :, 1], velocity[-1, :, 1]
    return u, v


def _faces_to_cell(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack((0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1] + v[1:])), axis=-1)


def _pressure_matrix(fluid: np.ndarray, dx: float, dy: float) -> sparse.csr_matrix:
    height, width = fluid.shape
    index = -np.ones((height, width), dtype=np.int64)
    index[fluid] = np.arange(int(np.count_nonzero(fluid)))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, col in np.argwhere(fluid):
        center = int(index[row, col])
        diagonal = 0.0
        for dr, dc, weight in (
            (0, -1, 1.0 / dx**2),
            (0, 1, 1.0 / dx**2),
            (-1, 0, 1.0 / dy**2),
            (1, 0, 1.0 / dy**2),
        ):
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < height and 0 <= cc < width and fluid[rr, cc]:
                rows.append(center)
                cols.append(int(index[rr, cc]))
                data.append(-weight)
                diagonal += weight
        rows.append(center)
        cols.append(center)
        data.append(diagonal)
    return sparse.csr_matrix((data, (rows, cols)), shape=(int(np.count_nonzero(fluid)),) * 2)


def project_mac(
    u: np.ndarray,
    v: np.ndarray,
    fluid: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, SolverInfo]:
    """Project staggered face velocities onto the discrete divergence-free space."""

    projected_u, projected_v, _, info = project_mac_with_pressure(u, v, fluid, dt, dx, dy)
    return projected_u, projected_v, info


def project_mac_with_pressure(
    u: np.ndarray,
    v: np.ndarray,
    fluid: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SolverInfo]:
    """Project MAC faces and return the cell-centred pressure correction.

    The pressure is part of the bounded Navier--Stokes ground truth.  Keeping it
    makes the momentum residual independently auditable instead of inferring a
    hidden pressure from the stored velocity after generation.
    """

    divergence = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
    rhs = -divergence[fluid] / dt
    compatibility_error = abs(float(np.mean(rhs)))
    compatibility_scale = max(float(np.sqrt(np.mean(rhs * rhs))), np.finfo(float).eps)
    if compatibility_error / compatibility_scale > 1.0e-8:
        raise RuntimeError(
            "pressure projection received incompatible boundary flux: "
            f"normalized_mean_rhs={compatibility_error / compatibility_scale:.6g}"
        )
    rhs -= float(np.mean(rhs))
    matrix = _pressure_matrix(fluid, dx, dy)
    diagonal = np.maximum(matrix.diagonal(), 1.0e-12)
    preconditioner = sparse.diags(1.0 / diagonal)
    iterations = 0

    def callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    pressure_vector, status = spla.cg(
        matrix,
        rhs,
        M=preconditioner,
        rtol=1.0e-9,
        atol=0.0,
        maxiter=4000,
        callback=callback,
    )
    pressure = np.zeros(fluid.shape, dtype=np.float64)
    pressure[fluid] = pressure_vector
    open_u = fluid[:, :-1] & fluid[:, 1:]
    open_v = fluid[:-1, :] & fluid[1:, :]
    u[:, 1:-1][open_u] -= dt * (pressure[:, 1:] - pressure[:, :-1])[open_u] / dx
    v[1:-1, :][open_v] -= dt * (pressure[1:, :] - pressure[:-1, :])[open_v] / dy
    residual = matrix @ pressure_vector - rhs
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).eps))
    info = SolverInfo(status == 0 and np.isfinite(relative), iterations, relative)
    if not info.converged:
        raise RuntimeError(
            f"pressure projection failed: status={status}, relative_residual={relative:.6g}"
        )
    return u, v, pressure, info


def _mac_solid_cells(geometry: np.ndarray, boundary: str) -> np.ndarray:
    """Return the cells excluded from a topology-specific MAC solve."""

    solid = np.asarray(geometry, dtype=np.float64) > 0.5
    solid = solid.copy()
    if normalize_boundary(boundary) == "robin":
        # B3 is a channel: top/bottom and the embedded object are solid while
        # the left/right columns remain open to the prescribed balanced flow.
        solid[:, [0, -1]] = False
        solid[[0, -1], :] = True
    return solid


def _mac_blocked_faces(solid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocked_u = np.zeros((solid.shape[0], solid.shape[1] + 1), dtype=bool)
    blocked_v = np.zeros((solid.shape[0] + 1, solid.shape[1]), dtype=bool)
    blocked_u[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
    blocked_v[1:-1, :] = solid[:-1, :] | solid[1:, :]
    return blocked_u, blocked_v


def _channel_profile(height: int, inflow_speed: float) -> np.ndarray:
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    profile = 4.0 * float(inflow_speed) * y * (1.0 - y)
    # These two stored rows are wall proxies, so the corner convention is
    # unambiguous and exactly reproducible in the quality evaluator.
    profile[[0, -1]] = 0.0
    return profile


def apply_mac_boundary(
    u: np.ndarray,
    v: np.ndarray,
    solid: np.ndarray,
    boundary: str,
    *,
    inflow_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the velocity boundary directly on staggered MAC faces."""

    boundary = normalize_boundary(boundary)
    if boundary == "periodic":
        raise ValueError("bounded MAC boundary cannot be periodic")
    blocked_u, blocked_v = _mac_blocked_faces(solid)
    u[blocked_u] = 0.0
    v[blocked_v] = 0.0
    if boundary == "dirichlet":
        u[[0, -1], :] = 0.0
        v[:, [0, -1]] = 0.0
        u[:, [0, -1]] = 0.0
        v[[0, -1], :] = 0.0
    elif boundary == "neumann":
        # Free slip: zero normal velocity and zero normal derivative of the
        # tangential component.  Set tangential proxies first so corners retain
        # the zero-normal convention.
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]
        u[:, [0, -1]] = 0.0
        v[[0, -1], :] = 0.0
    elif boundary == "robin":
        profile = _channel_profile(u.shape[0], inflow_speed)
        u[[0, -1], :] = 0.0
        v[:, [0, -1]] = 0.0
        v[[0, -1], :] = 0.0
        u[:, 0] = profile
        u[:, -1] = profile
    else:  # pragma: no cover - normalization rejects unsupported values
        raise ValueError(f"unsupported MAC boundary {boundary!r}")
    return u, v


def encode_mac_state(u: np.ndarray, v: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Encode left/bottom MAC faces and cell pressure as ``[H,W,3]``."""

    height, width = pressure.shape
    if u.shape != (height, width + 1) or v.shape != (height + 1, width):
        raise ValueError("MAC face shapes do not match pressure cells")
    return np.stack((u[:, :-1], v[:-1, :], pressure), axis=-1)


def decode_mac_state(
    state: np.ndarray,
    geometry: np.ndarray,
    boundary: str,
    *,
    inflow_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode ``[H,W,3]`` and reconstruct the known right/top boundary faces."""

    encoded = np.asarray(state, dtype=np.float64)
    if encoded.ndim != 3 or encoded.shape[-1] != 3:
        raise ValueError("bounded MAC state must have shape [H,W,3]")
    height, width, _ = encoded.shape
    u = np.zeros((height, width + 1), dtype=np.float64)
    v = np.zeros((height + 1, width), dtype=np.float64)
    u[:, :-1] = encoded[..., 0]
    v[:-1, :] = encoded[..., 1]
    pressure = encoded[..., 2].copy()
    solid = _mac_solid_cells(geometry, boundary)
    apply_mac_boundary(u, v, solid, boundary, inflow_speed=inflow_speed)
    return u, v, pressure, solid


def initial_bounded_mac_state(
    velocity: np.ndarray,
    geometry: np.ndarray,
    dx: float,
    dy: float,
    boundary: str,
    *,
    inflow_speed: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Project an initial collocated field and encode auditable MAC ground truth."""

    u, v = _cell_to_faces(np.asarray(velocity, dtype=np.float64))
    solid = _mac_solid_cells(geometry, boundary)
    apply_mac_boundary(u, v, solid, boundary, inflow_speed=inflow_speed)
    fluid = ~solid
    u, v, _, projection = project_mac_with_pressure(u, v, fluid, 1.0, dx, dy)
    apply_mac_boundary(u, v, solid, boundary, inflow_speed=inflow_speed)
    divergence = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
    scale = np.sqrt(np.mean(((u[:, 1:] - u[:, :-1]) / dx)[fluid] ** 2)) + np.sqrt(
        np.mean(((v[1:, :] - v[:-1, :]) / dy)[fluid] ** 2)
    )
    return encode_mac_state(u, v, np.zeros_like(geometry, dtype=np.float64)), {
        "pressure_iterations_max": projection.iterations,
        "pressure_relative_residual_max": projection.relative_residual,
        "divergence_loss_normalized_solver": float(
            np.sqrt(np.mean(divergence[fluid] ** 2)) / max(float(scale), np.finfo(float).eps)
        ),
    }


def advance_bounded_mac_state(
    state: FloatArray,
    geometry: FloatArray,
    viscosity: float,
    frame_dt: float,
    dx: float,
    dy: float,
    boundary: str,
    *,
    inflow_speed: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Advance a bounded velocity-pressure state on one MAC grid.

    Advection is semi-Lagrangian, diffusion is bounded Crank--Nicolson, and
    every substep ends with a pressure projection on the same face topology
    that is stored.  Observation masks are intentionally absent from this API.
    """

    u, v, pressure, solid = decode_mac_state(state, geometry, boundary, inflow_speed=inflow_speed)
    fluid = ~solid
    cell_velocity = _faces_to_cell(u, v)
    speed = max(float(np.max(np.linalg.norm(cell_velocity[fluid], axis=-1))), 1.0e-8)
    substeps = max(1, int(np.ceil(frame_dt * speed / (0.35 * min(dx, dy)))))
    dt = frame_dt / substeps
    maximum_iterations = 0
    maximum_residual = 0.0
    for _ in range(substeps):
        cell_velocity = _faces_to_cell(u, v)
        advected = np.empty_like(cell_velocity)
        for channel in range(2):
            advected[..., channel] = semi_lagrangian(
                cell_velocity[..., channel],
                cell_velocity[..., 0],
                cell_velocity[..., 1],
                dt,
                dx,
                dy,
                boundary,
            )
            advected[..., channel], _ = crank_nicolson_diffusion(
                advected[..., channel], viscosity, dt, dx, dy, "neumann"
            )
        apply_velocity_boundary(
            advected,
            boundary,
            geometry=np.asarray(geometry)[..., None],
            inflow_speed=inflow_speed,
        )
        advected[solid] = 0.0
        u, v = _cell_to_faces(advected)
        apply_mac_boundary(u, v, solid, boundary, inflow_speed=inflow_speed)
        u, v, pressure, projection = project_mac_with_pressure(u, v, fluid, dt, dx, dy)
        apply_mac_boundary(u, v, solid, boundary, inflow_speed=inflow_speed)
        maximum_iterations = max(maximum_iterations, projection.iterations)
        maximum_residual = max(maximum_residual, projection.relative_residual)
    divergence_x = (u[:, 1:] - u[:, :-1]) / dx
    divergence_y = (v[1:, :] - v[:-1, :]) / dy
    divergence = divergence_x + divergence_y
    gradient_scale = np.sqrt(np.mean(divergence_x[fluid] ** 2)) + np.sqrt(
        np.mean(divergence_y[fluid] ** 2)
    )
    return encode_mac_state(u, v, pressure), {
        "substeps": substeps,
        "pressure_iterations_max": maximum_iterations,
        "pressure_relative_residual_max": maximum_residual,
        "divergence_loss_normalized_solver": float(
            np.sqrt(np.mean(divergence[fluid] ** 2))
            / max(float(gradient_scale), np.finfo(float).eps)
        ),
    }


_LBM_DIRECTIONS = np.asarray(
    ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)),
    dtype=np.int64,
)
_LBM_WEIGHTS = np.asarray((4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36))
_LBM_OPPOSITE = np.asarray((0, 3, 4, 1, 2, 7, 8, 5, 6), dtype=np.int64)
_LBM_SOUND_SPEED_SQUARED = 1.0 / 3.0


def lbm_step_parameters(
    frame_dt: float,
    viscosity: float,
    dx: float,
    inflow_speed: float,
) -> tuple[int, float, float]:
    """Choose a stable physical time step and BGK relaxation time."""

    diffusive_limit = 0.20 * dx**2 / max(viscosity, 1.0e-12)
    advective_limit = 0.08 * dx / max(inflow_speed, 1.0e-8)
    substeps = max(1, int(np.ceil(frame_dt / min(diffusive_limit, advective_limit))))
    dt = frame_dt / substeps
    lattice_viscosity = viscosity * dt / dx**2
    relaxation_time = 0.5 + 3.0 * lattice_viscosity
    if not 0.5001 < relaxation_time < 1.95:
        raise RuntimeError(f"LBM relaxation time {relaxation_time:.6g} is outside the stable range")
    return substeps, dt, relaxation_time


def _lbm_equilibrium(
    density: np.ndarray, velocity_x: np.ndarray, velocity_y: np.ndarray
) -> np.ndarray:
    velocity_squared = velocity_x**2 + velocity_y**2
    equilibrium = np.empty(density.shape + (9,), dtype=np.float64)
    for index, (cx, cy) in enumerate(_LBM_DIRECTIONS):
        cu = cx * velocity_x + cy * velocity_y
        equilibrium[..., index] = (
            _LBM_WEIGHTS[index] * density * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * velocity_squared)
        )
    return equilibrium


def initialize_lbm_distributions(
    velocity: np.ndarray,
    pressure: np.ndarray,
    dt: float,
    dx: float,
) -> np.ndarray:
    """Lift physical velocity/pressure fields into D2Q9 equilibrium populations."""

    state = np.asarray(velocity, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    lattice_scale = dt / dx
    density = 1.0 + pressure / (_LBM_SOUND_SPEED_SQUARED * (dx / dt) ** 2)
    if np.any(density <= 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("LBM density reconstructed from pressure must be finite and positive")
    return _lbm_equilibrium(
        density,
        state[..., 0] * lattice_scale,
        state[..., 1] * lattice_scale,
    )


def lbm_macroscopic(
    distributions: np.ndarray,
    solid: np.ndarray,
    dt: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density = np.sum(distributions, axis=-1)
    safe_density = np.maximum(density, np.finfo(float).eps)
    velocity_x_lattice = np.sum(distributions * _LBM_DIRECTIONS[:, 0], axis=-1) / safe_density
    velocity_y_lattice = np.sum(distributions * _LBM_DIRECTIONS[:, 1], axis=-1) / safe_density
    velocity_x = velocity_x_lattice * dx / dt
    velocity_y = velocity_y_lattice * dx / dt
    velocity_x[solid] = 0.0
    velocity_y[solid] = 0.0
    reference_density = float(np.mean(density[~solid]))
    pressure = (density - reference_density) * _LBM_SOUND_SPEED_SQUARED * (dx / dt) ** 2
    pressure[solid] = 0.0
    return velocity_x, velocity_y, pressure


def advance_lbm_channel(
    distributions: np.ndarray,
    geometry: FloatArray,
    substeps: int,
    dt: float,
    dx: float,
    relaxation_time: float,
    *,
    inflow_speed: float,
    body_force_x: float,
) -> np.ndarray:
    """Advance a body-force-driven, streamwise-periodic obstacle channel."""

    populations = np.asarray(distributions, dtype=np.float64).copy()
    solid = _mac_solid_cells(np.asarray(geometry), "robin")
    del inflow_speed  # retained in the public signature for protocol provenance
    lattice_acceleration_x = float(body_force_x) * dt**2 / dx
    for _ in range(int(substeps)):
        density = np.sum(populations, axis=-1)
        safe_density = np.maximum(density, np.finfo(float).eps)
        ux = np.sum(populations * _LBM_DIRECTIONS[:, 0], axis=-1) / safe_density
        uy = np.sum(populations * _LBM_DIRECTIONS[:, 1], axis=-1) / safe_density
        ux[solid] = 0.0
        uy[solid] = 0.0
        equilibrium = _lbm_equilibrium(density, ux, uy)
        post_collision = populations - (populations - equilibrium) / relaxation_time
        force_x = density * lattice_acceleration_x
        prefactor = 1.0 - 0.5 / relaxation_time
        for index, (cx, cy) in enumerate(_LBM_DIRECTIONS):
            cu = cx * ux + cy * uy
            forcing = _LBM_WEIGHTS[index] * prefactor * (3.0 * (cx - ux) + 9.0 * cu * cx) * force_x
            post_collision[..., index] += forcing
        streamed = np.zeros_like(populations)
        fluid_source = ~solid
        for index, (cx, cy) in enumerate(_LBM_DIRECTIONS):
            valid = fluid_source.copy()
            neighbor_solid = np.roll(np.roll(solid, -int(cy), axis=0), -int(cx), axis=1)
            reflected = valid & neighbor_solid
            transmitted = valid & ~neighbor_solid
            moving = np.where(transmitted, post_collision[..., index], 0.0)
            streamed[..., index] += np.roll(np.roll(moving, int(cy), axis=0), int(cx), axis=1)
            streamed[..., int(_LBM_OPPOSITE[index])][reflected] += post_collision[..., index][
                reflected
            ]
        density = np.sum(streamed, axis=-1)
        streamed[solid] = _lbm_equilibrium(
            np.ones_like(density),
            np.zeros_like(density),
            np.zeros_like(density),
        )[solid]
        populations = streamed
    return populations


@lru_cache(maxsize=4)
def _periodic_channel_pressure_solver(
    height: int,
    width: int,
    solid_bytes: bytes,
    dx: float,
    dy: float,
) -> tuple[Any, np.ndarray]:
    solid = np.frombuffer(solid_bytes, dtype=np.bool_).reshape(height, width)
    fluid = ~solid
    index = -np.ones((height, width), dtype=np.int64)
    index[fluid] = np.arange(int(np.count_nonzero(fluid)))
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row, col in np.argwhere(fluid):
        center = int(index[row, col])
        diagonal = 0.0
        for rr, cc, weight in (
            (row, (col - 1) % width, 1.0 / dx**2),
            (row, (col + 1) % width, 1.0 / dx**2),
            (row - 1, col, 1.0 / dy**2),
            (row + 1, col, 1.0 / dy**2),
        ):
            if 0 <= rr < height and fluid[rr, cc]:
                rows.append(center)
                cols.append(int(index[rr, cc]))
                values.append(-weight)
                diagonal += weight
        rows.append(center)
        cols.append(center)
        values.append(diagonal)
    matrix = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(int(np.count_nonzero(fluid)),) * 2,
    ).tolil()
    # Fix the arbitrary pressure gauge while preserving a symmetric SPD system.
    matrix[0, :] = 0.0
    matrix[:, 0] = 0.0
    matrix[0, 0] = 1.0
    solver = spla.factorized(matrix.tocsc())
    return solver, index


def project_periodic_channel_velocity(
    velocity: np.ndarray,
    geometry: FloatArray,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Project a periodic-x obstacle channel and encode MAC velocity plus pressure."""

    state = np.asarray(velocity, dtype=np.float64)
    solid = _mac_solid_cells(np.asarray(geometry), "robin")
    fluid = ~solid
    height, width = solid.shape
    u = np.zeros((height, width + 1), dtype=np.float64)
    v = np.zeros((height + 1, width), dtype=np.float64)
    u[:, 1:-1] = 0.5 * (state[:, :-1, 0] + state[:, 1:, 0])
    periodic_u = 0.5 * (state[:, -1, 0] + state[:, 0, 0])
    u[:, 0] = periodic_u
    u[:, -1] = periodic_u
    v[1:-1, :] = 0.5 * (state[:-1, :, 1] + state[1:, :, 1])
    blocked_u, blocked_v = _mac_blocked_faces(solid)
    u[blocked_u] = 0.0
    v[blocked_v] = 0.0
    divergence = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
    rhs_values = -divergence[fluid] / dt
    compatibility = abs(float(np.mean(rhs_values))) / max(
        float(np.sqrt(np.mean(rhs_values**2))), np.finfo(float).eps
    )
    if compatibility > 1.0e-8:
        raise RuntimeError(f"periodic-channel projection flux is incompatible: {compatibility:.6g}")
    rhs_values -= float(np.mean(rhs_values))
    solver, index = _periodic_channel_pressure_solver(
        height, width, np.ascontiguousarray(solid).tobytes(), float(dx), float(dy)
    )
    rhs_values[0] = 0.0
    pressure_values = np.asarray(solver(rhs_values), dtype=np.float64)
    pressure = np.zeros((height, width), dtype=np.float64)
    pressure[fluid] = pressure_values
    open_u = fluid[:, :-1] & fluid[:, 1:]
    open_v = fluid[:-1, :] & fluid[1:, :]
    u[:, 1:-1][open_u] -= dt * (pressure[:, 1:] - pressure[:, :-1])[open_u] / dx
    v[1:-1, :][open_v] -= dt * (pressure[1:, :] - pressure[:-1, :])[open_v] / dy
    periodic_open = fluid[:, -1] & fluid[:, 0]
    u[:, 0][periodic_open] -= dt * (pressure[:, 0] - pressure[:, -1])[periodic_open] / dx
    u[:, -1] = u[:, 0]
    u[blocked_u] = 0.0
    v[blocked_v] = 0.0
    divergence_x = (u[:, 1:] - u[:, :-1]) / dx
    divergence_y = (v[1:, :] - v[:-1, :]) / dy
    divergence = divergence_x + divergence_y
    scale = np.sqrt(np.mean(divergence_x[fluid] ** 2)) + np.sqrt(np.mean(divergence_y[fluid] ** 2))
    cell_velocity = _faces_to_cell(u, v)
    cell_velocity[solid] = 0.0
    residual = float(
        np.sqrt(np.mean(divergence[fluid] ** 2)) / max(float(scale), np.finfo(float).eps)
    )
    return (
        encode_mac_state(u, v, pressure),
        cell_velocity,
        {
            "divergence_loss_normalized_solver": residual,
            "pressure_compatibility_error": compatibility,
        },
    )


def _periodic_channel_rhs(
    velocity: np.ndarray,
    geometry: FloatArray,
    viscosity: float,
    body_force_x: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Centered finite-difference momentum right-hand side in a periodic channel."""

    state = np.asarray(velocity, dtype=np.float64)
    solid = _mac_solid_cells(np.asarray(geometry), "robin")
    result = np.zeros_like(state)
    for component in range(2):
        value = state[..., component]
        grad_x = (np.roll(value, -1, axis=1) - np.roll(value, 1, axis=1)) / (2.0 * dx)
        grad_y = (np.roll(value, -1, axis=0) - np.roll(value, 1, axis=0)) / (2.0 * dy)
        laplace = (np.roll(value, -1, axis=1) - 2.0 * value + np.roll(value, 1, axis=1)) / dx**2 + (
            np.roll(value, -1, axis=0) - 2.0 * value + np.roll(value, 1, axis=0)
        ) / dy**2
        result[..., component] = (
            -(state[..., 0] * grad_x + state[..., 1] * grad_y) + viscosity * laplace
        )
    result[..., 0] += float(body_force_x)
    result[solid] = 0.0
    return result


def advance_projected_channel_velocity(
    velocity: np.ndarray,
    geometry: FloatArray,
    viscosity: float,
    frame_dt: float,
    dx: float,
    dy: float,
    *,
    body_force_x: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Advance a periodic-x obstacle channel with SSP-RK2 and MAC projection.

    The predictor uses the same second-order spatial operator reported by the
    quality layer.  Both Runge--Kutta stages are pressure projected, so no-slip
    walls/obstacle faces and incompressibility are enforced by the numerical
    operator rather than by overwriting a completed periodic solution.
    """

    state = np.asarray(velocity, dtype=np.float64).copy()
    solid = _mac_solid_cells(np.asarray(geometry), "robin")
    state[solid] = 0.0
    speed = max(float(np.max(np.linalg.norm(state, axis=-1))), 1.0e-10)
    spacing = min(dx, dy)
    advective_dt = 0.30 * spacing / speed
    diffusive_dt = np.inf if viscosity <= 0.0 else 0.18 * spacing**2 / float(viscosity)
    stable_dt = min(advective_dt, diffusive_dt)
    substeps = max(1, int(np.ceil(float(frame_dt) / stable_dt)))
    dt = float(frame_dt) / substeps
    max_divergence = 0.0
    max_compatibility = 0.0
    encoded: np.ndarray | None = None
    for _ in range(substeps):
        rhs0 = _periodic_channel_rhs(state, geometry, viscosity, body_force_x, dx, dy)
        _, stage, first_projection = project_periodic_channel_velocity(
            state + dt * rhs0, geometry, dt, dx, dy
        )
        rhs1 = _periodic_channel_rhs(stage, geometry, viscosity, body_force_x, dx, dy)
        encoded, state, second_projection = project_periodic_channel_velocity(
            0.5 * (state + stage + dt * rhs1), geometry, dt, dx, dy
        )
        max_divergence = max(
            max_divergence,
            float(first_projection["divergence_loss_normalized_solver"]),
            float(second_projection["divergence_loss_normalized_solver"]),
        )
        max_compatibility = max(
            max_compatibility,
            float(first_projection["pressure_compatibility_error"]),
            float(second_projection["pressure_compatibility_error"]),
        )
    if encoded is None:  # pragma: no cover - substeps is always positive
        raise RuntimeError("channel advance performed no substeps")
    return (
        encoded,
        state,
        {
            "substeps": substeps,
            "internal_time_step": dt,
            "max_courant": speed * dt / spacing,
            "divergence_loss_normalized_solver": max_divergence,
            "pressure_compatibility_error": max_compatibility,
        },
    )


@lru_cache(maxsize=8)
def _masked_streamfunction_solver(
    height: int,
    width: int,
    solid_bytes: bytes,
    dx: float,
    dy: float,
) -> tuple[Any, np.ndarray]:
    """Factor ``-laplace`` on an arbitrary wall/obstacle fluid domain."""

    solid = np.frombuffer(solid_bytes, dtype=np.bool_).reshape(height, width)
    fluid = ~solid
    index = -np.ones((height, width), dtype=np.int64)
    index[fluid] = np.arange(int(np.count_nonzero(fluid)))
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    diagonal = 2.0 / dx**2 + 2.0 / dy**2
    for row, col in np.argwhere(fluid):
        center = int(index[row, col])
        rows.append(center)
        cols.append(center)
        values.append(diagonal)
        for rr, cc, weight in (
            (row, col - 1, -1.0 / dx**2),
            (row, col + 1, -1.0 / dx**2),
            (row - 1, col, -1.0 / dy**2),
            (row + 1, col, -1.0 / dy**2),
        ):
            if 0 <= rr < height and 0 <= cc < width and fluid[rr, cc]:
                rows.append(center)
                cols.append(int(index[rr, cc]))
                values.append(weight)
    matrix = sparse.csc_matrix(
        (values, (rows, cols)),
        shape=(int(np.count_nonzero(fluid)),) * 2,
    )
    return spla.factorized(matrix), index


def solve_masked_streamfunction(
    vorticity: np.ndarray,
    geometry: FloatArray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover divergence-free velocity on a wall/obstacle domain."""

    omega = np.asarray(vorticity, dtype=np.float64)
    solid = np.asarray(geometry, dtype=np.float64) > 0.5
    fluid = ~solid
    if not np.any(fluid):
        raise ValueError("masked streamfunction domain contains no fluid cells")
    solver, index = _masked_streamfunction_solver(
        omega.shape[0],
        omega.shape[1],
        np.ascontiguousarray(solid).tobytes(),
        float(dx),
        float(dy),
    )
    streamfunction = np.zeros_like(omega)
    streamfunction[fluid] = np.asarray(solver(omega[fluid]), dtype=np.float64)
    velocity_x = (np.roll(streamfunction, -1, axis=0) - np.roll(streamfunction, 1, axis=0)) / (
        2.0 * dy
    )
    velocity_y = -(np.roll(streamfunction, -1, axis=1) - np.roll(streamfunction, 1, axis=1)) / (
        2.0 * dx
    )
    velocity_x[solid] = 0.0
    velocity_y[solid] = 0.0
    return velocity_x, velocity_y, streamfunction


def apply_masked_vorticity_boundary(
    vorticity: np.ndarray,
    streamfunction: np.ndarray,
    geometry: FloatArray,
    dx: float,
    dy: float,
) -> None:
    """Apply Thom's no-slip wall vorticity on outer and obstacle solid cells."""

    solid = np.asarray(geometry, dtype=np.float64) > 0.5
    fluid = ~solid
    accumulated = np.zeros_like(vorticity, dtype=np.float64)
    counts = np.zeros_like(vorticity, dtype=np.int16)
    for row_shift, col_shift, spacing in (
        (-1, 0, dy),
        (1, 0, dy),
        (0, -1, dx),
        (0, 1, dx),
    ):
        neighbor_fluid = np.roll(np.roll(fluid, -row_shift, axis=0), -col_shift, axis=1)
        neighbor_psi = np.roll(np.roll(streamfunction, -row_shift, axis=0), -col_shift, axis=1)
        interface = solid & neighbor_fluid
        accumulated[interface] += -2.0 * neighbor_psi[interface] / spacing**2
        counts[interface] += 1
    boundary_solid = solid & (counts > 0)
    vorticity[solid] = 0.0
    vorticity[boundary_solid] = accumulated[boundary_solid] / counts[boundary_solid]


def advance_masked_vorticity(
    vorticity: np.ndarray,
    geometry: FloatArray,
    forcing: np.ndarray,
    viscosity: float,
    frame_dt: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """SSP-RK2 vorticity solve on a wall/obstacle domain."""

    state = np.asarray(vorticity, dtype=np.float64).copy()
    solid = np.asarray(geometry, dtype=np.float64) > 0.5
    fluid = ~solid

    def rhs(current: np.ndarray) -> tuple[np.ndarray, float]:
        velocity_x, velocity_y, psi = solve_masked_streamfunction(current, geometry, dx, dy)
        apply_masked_vorticity_boundary(current, psi, geometry, dx, dy)
        grad_x = (np.roll(current, -1, axis=1) - np.roll(current, 1, axis=1)) / (2.0 * dx)
        grad_y = (np.roll(current, -1, axis=0) - np.roll(current, 1, axis=0)) / (2.0 * dy)
        laplace = (
            np.roll(current, -1, axis=1) - 2.0 * current + np.roll(current, 1, axis=1)
        ) / dx**2 + (
            np.roll(current, -1, axis=0) - 2.0 * current + np.roll(current, 1, axis=0)
        ) / dy**2
        result = -velocity_x * grad_x - velocity_y * grad_y + viscosity * laplace
        result += np.asarray(forcing, dtype=np.float64)
        result[solid] = 0.0
        speed = float(np.max(np.sqrt(velocity_x[fluid] ** 2 + velocity_y[fluid] ** 2)))
        return result, speed

    _, initial_speed = rhs(state.copy())
    spacing = min(dx, dy)
    advective_dt = np.inf if initial_speed <= 1.0e-12 else 0.30 * spacing / initial_speed
    diffusive_dt = np.inf if viscosity <= 0.0 else 0.18 * spacing**2 / viscosity
    substeps = max(1, int(np.ceil(frame_dt / min(advective_dt, diffusive_dt))))
    dt = frame_dt / substeps
    max_speed = initial_speed
    for _ in range(substeps):
        first_rhs, first_speed = rhs(state.copy())
        stage = state + dt * first_rhs
        _, stage_psi = solve_masked_streamfunction(stage, geometry, dx, dy)[1:]
        apply_masked_vorticity_boundary(stage, stage_psi, geometry, dx, dy)
        second_rhs, second_speed = rhs(stage.copy())
        state = 0.5 * state + 0.5 * (stage + dt * second_rhs)
        _, state_psi = solve_masked_streamfunction(state, geometry, dx, dy)[1:]
        apply_masked_vorticity_boundary(state, state_psi, geometry, dx, dy)
        max_speed = max(max_speed, first_speed, second_speed)
    return state, {
        "substeps": substeps,
        "internal_time_step": dt,
        "max_courant": max_speed * dt / spacing,
        "divergence_loss_normalized_solver": 0.0,
    }


def advance_bounded_velocity(
    velocity: FloatArray,
    geometry: FloatArray,
    viscosity: float,
    frame_dt: float,
    dx: float,
    dy: float,
    boundary: str,
    *,
    inflow_speed: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Advance bounded incompressible flow with a MAC pressure projection."""

    state = np.asarray(velocity, dtype=np.float64).copy()
    solid = np.asarray(geometry, dtype=np.float64) > 0.5
    # Outer walls are boundary faces, not solid cells in the projection domain.
    fluid = ~solid
    fluid[[0, -1], :] = True
    fluid[:, [0, -1]] = True
    speed = max(float(np.max(np.linalg.norm(state, axis=-1))), 1.0e-8)
    substeps = max(1, int(np.ceil(frame_dt * speed / (0.45 * min(dx, dy)))))
    dt = frame_dt / substeps
    max_pressure_iterations = 0
    max_pressure_residual = 0.0
    for _ in range(substeps):
        advected = np.empty_like(state)
        advected[..., 0] = semi_lagrangian(
            state[..., 0], state[..., 0], state[..., 1], dt, dx, dy, boundary
        )
        advected[..., 1] = semi_lagrangian(
            state[..., 1], state[..., 0], state[..., 1], dt, dx, dy, boundary
        )
        for channel in range(2):
            advected[..., channel], _ = crank_nicolson_diffusion(
                advected[..., channel], viscosity, dt, dx, dy, "neumann"
            )
        apply_velocity_boundary(
            advected,
            boundary,
            geometry=geometry[..., None],
            inflow_speed=inflow_speed,
        )
        advected[solid] = 0.0
        u, v = _cell_to_faces(advected)
        blocked_u = np.zeros_like(u, dtype=bool)
        blocked_v = np.zeros_like(v, dtype=bool)
        blocked_u[:, 1:-1] = solid[:, :-1] | solid[:, 1:]
        blocked_v[1:-1, :] = solid[:-1, :] | solid[1:, :]
        u[blocked_u] = 0.0
        v[blocked_v] = 0.0
        u, v, projection = project_mac(u, v, fluid, dt, dx, dy)
        u[blocked_u] = 0.0
        v[blocked_v] = 0.0
        state = _faces_to_cell(u, v)
        apply_velocity_boundary(
            state,
            boundary,
            geometry=geometry[..., None],
            inflow_speed=inflow_speed,
        )
        state[solid] = 0.0
        max_pressure_iterations = max(max_pressure_iterations, projection.iterations)
        max_pressure_residual = max(max_pressure_residual, projection.relative_residual)
    du_dx, _ = gradient(state[..., 0], boundary, dx, dy)
    _, dv_dy = gradient(state[..., 1], boundary, dx, dy)
    divergence_rms = float(np.sqrt(np.mean((du_dx + dv_dy)[fluid] ** 2)))
    velocity_gradient_rms = float(
        np.sqrt(np.mean(du_dx[fluid] ** 2)) + np.sqrt(np.mean(dv_dy[fluid] ** 2))
    )
    return state, {
        "substeps": substeps,
        "pressure_iterations_max": max_pressure_iterations,
        "pressure_relative_residual_max": max_pressure_residual,
        "divergence_loss_normalized_solver": divergence_rms
        / max(velocity_gradient_rms, np.finfo(float).eps),
    }


__all__ = [
    "SolverInfo",
    "advance_bounded_mac_state",
    "advance_bounded_vorticity",
    "advance_bounded_velocity",
    "advance_burgers",
    "advance_periodic_vorticity",
    "apply_vorticity_wall_boundary",
    "advance_reaction_diffusion",
    "advance_lbm_channel",
    "crank_nicolson_diffusion",
    "bounded_velocity_from_vorticity",
    "decode_mac_state",
    "encode_mac_state",
    "initial_bounded_mac_state",
    "initialize_lbm_distributions",
    "lbm_macroscopic",
    "lbm_step_parameters",
    "project_mac",
    "project_mac_with_pressure",
    "solve_elliptic",
    "solve_bounded_streamfunction",
]
