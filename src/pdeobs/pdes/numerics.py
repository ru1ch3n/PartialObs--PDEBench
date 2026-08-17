"""Boundary-consistent numerical kernels used by the dataset generators.

The routines in this module deliberately separate periodic Fourier solvers
from bounded finite-difference/finite-volume solvers.  A periodic update must
never be followed by a cosmetic boundary overwrite: the boundary protocol is
part of the discrete operator that is solved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import linalg as spla

from .common import (
    apply_scalar_boundary,
    apply_velocity_boundary,
    gradient,
    laplacian,
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

    def matvec(vector: np.ndarray) -> np.ndarray:
        values = unpack(vector)
        result = _variable_operator(values, coeff, boundary, dx, dy) + reaction * values
        packed = pack(result)
        if nullspace:
            packed = packed + float(np.mean(vector))
        return packed

    operator = spla.LinearOperator(
        (vector_size, vector_size), matvec=matvec, dtype=np.dtype(np.float64)
    )
    diagonal = (
        2.0 * coeff[1:-1, 1:-1] / (dx * dx)
        + 2.0 * coeff[1:-1, 1:-1] / (dy * dy)
        + abs(float(reaction))
    )
    if periodic:
        diagonal = (
            2.0 * coeff / (dx * dx) + 2.0 * coeff / (dy * dy) + abs(float(reaction))
        )
    diagonal = np.maximum(diagonal.reshape(-1), 1.0e-12)
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
    residual = pack(_variable_operator(solution, coeff, boundary, dx, dy) + reaction * solution)
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
    operator_at_old = _variable_operator(
        state, np.ones_like(state), boundary, dx, dy
    )
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


def _spectral_advection(values: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = values.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(height, d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(width, d=dx)
    kkx, kky = np.meshgrid(kx, ky)
    transform = np.fft.fft2(values)
    cutoff_x, cutoff_y = width // 3, height // 3
    modes_x = np.fft.fftfreq(width) * width
    modes_y = np.fft.fftfreq(height) * height
    dealias = (np.abs(modes_y)[:, None] <= cutoff_y) & (
        np.abs(modes_x)[None, :] <= cutoff_x
    )
    grad_x = np.fft.ifft2(1j * kkx * transform).real
    grad_y = np.fft.ifft2(1j * kky * transform).real
    nonlinear = values * (grad_x + grad_y)
    return np.fft.ifft2(np.fft.fft2(nonlinear) * dealias).real


def advance_burgers(
    values: FloatArray,
    viscosity: float,
    frame_dt: float,
    dx: float,
    dy: float,
    boundary: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Advance the 2-D scalar Burgers extension with BC-consistent splitting."""

    state = np.asarray(values, dtype=np.float64).copy()
    boundary = normalize_boundary(boundary)
    speed = max(float(np.max(np.abs(state))), 1.0e-8)
    substeps = max(1, int(np.ceil(frame_dt * speed / (0.35 * min(dx, dy)))))
    dt = frame_dt / substeps
    max_courant = 0.0
    maximum_diffusion_iterations = 0
    for _ in range(substeps):
        courant = float(np.max(np.abs(state))) * dt / min(dx, dy)
        max_courant = max(max_courant, courant)
        if boundary == "periodic":
            k1 = -_spectral_advection(state, dx, dy)
            predictor = state + dt * k1
            k2 = -_spectral_advection(predictor, dx, dy)
            state = state + 0.5 * dt * (k1 + k2)
        else:
            gx, gy = gradient(state, boundary, dx, dy)
            k1 = -state * (gx + gy)
            predictor = state + dt * k1
            apply_scalar_boundary(predictor, boundary)
            pgx, pgy = gradient(predictor, boundary, dx, dy)
            k2 = -predictor * (pgx + pgy)
            state = state + 0.5 * dt * (k1 + k2)
            apply_scalar_boundary(state, boundary)
        state, info = crank_nicolson_diffusion(
            state, viscosity, dt, dx, dy, boundary
        )
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
    internal_dt: float = 1.0e-3,
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
    dealias = (np.abs(modes_y)[:, None] <= height // 3) & (
        np.abs(modes_x)[None, :] <= width // 3
    )
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
        omega_hat[~dealias] = 0.0
    result = np.fft.ifft2(omega_hat).real
    return result - float(np.mean(result)), steps


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

    divergence = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
    rhs = -divergence[fluid] / dt
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
    return u, v, info


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
    "advance_bounded_velocity",
    "advance_burgers",
    "advance_periodic_vorticity",
    "crank_nicolson_diffusion",
    "project_mac",
    "solve_elliptic",
]
