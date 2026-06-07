"""Pressure estimation from time-resolved acceleration data.

This module ports the finite-difference pressure solver in ``PressureSolve.jl``
to Python.  Inputs follow the same array layout:

* ``mask`` has shape ``(nx, ny, nz)``.
* ``acceleration`` has shape ``(3, nx, ny, nz, nt)`` or ``(3, nx, ny, nz)``
  for a single time point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsmr
from scipy.ndimage import correlate1d

Scheme = Literal[0, 1, 2]
PA_TO_MMHG = 0.00750062


@dataclass(frozen=True)
class FiniteDifferenceSystem:
    """Sparse masked finite-difference operator and mask indexing metadata."""

    matrix: csr_matrix
    mask_indices: np.ndarray
    mask_shape: tuple[int, int, int]
    order: Literal["F", "C"] = "F"


def compute_laplacian_3d(field: np.ndarray, res: np.ndarray) -> np.ndarray:
    """Computes the discrete 3D Laplacian of a scalar field.

    Accounts for physical grid spacing. Uses 'reflect' at boundaries
    to prevent artificial gradients at the edge of the bounding box.
    """
    laplacian = np.zeros_like(field)
    weights = np.array([1.0, -2.0, 1.0])

    for axis, dx in enumerate(res):
        # Apply 1D second-derivative filter along the current axis
        laplacian += correlate1d(
            field,
            weights / (dx ** 2),
            axis=axis,
            mode='reflect'
        )

    return laplacian


def make_fd_matrix(
    mask: np.ndarray,
    res: tuple[float, float, float] | list[float] | np.ndarray,
    *,
    scheme: Scheme = 1,
    dtype: np.dtype | type = np.float64,
    order: Literal["F", "C"] = "F",
) -> FiniteDifferenceSystem:
    """Build the sparse masked finite-difference matrix.

    ``scheme`` matches the Julia implementation:

    * ``0``: first-order forward/backward differences.
    * ``1``: second-order central differences where possible.
    * ``2``: fourth-order central differences where possible, second-order
      one-sided differences near boundaries, and first-order otherwise.

    The default Fortran ordering matches Julia's ``findall``/linear indexing.
    """

    if scheme not in (0, 1, 2):
        raise ValueError("Only schemes 0, 1, and 2 are supported")

    dtype = np.dtype(dtype)
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 3:
        raise ValueError("mask must have shape (nx, ny, nz)")

    res_arr = np.asarray(res, dtype=dtype)
    if res_arr.shape != (3,):
        raise ValueError("res must contain exactly three spatial resolutions")
    if np.any(res_arr <= 0):
        raise ValueError("spatial resolutions must be positive")

    mask_indices = np.flatnonzero(mask_bool.ravel(order=order))
    n_mask = int(mask_indices.size)
    shape = mask_bool.shape

    index_map = np.full(shape, -1, dtype=np.int64, order=order)
    index_map.ravel(order=order)[mask_indices] = np.arange(n_mask, dtype=np.int64)
    coords = np.column_stack(np.unravel_index(mask_indices, shape, order=order))
    current = np.arange(n_mask, dtype=np.int64)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    def neighbor(axis: int, step: int) -> np.ndarray:
        ncoords = coords.copy()
        ncoords[:, axis] += step
        valid = (0 <= ncoords[:, axis]) & (ncoords[:, axis] < shape[axis])
        out = np.full(n_mask, -1, dtype=np.int64)
        if np.any(valid):
            out[valid] = index_map[tuple(ncoords[valid].T)]
        return out

    def add(row: np.ndarray, col: np.ndarray, value: float) -> None:
        rows.append(row.astype(np.int64, copy=False))
        cols.append(col.astype(np.int64, copy=False))
        data.append(np.full(row.shape, value, dtype=dtype))

    for axis, spacing in enumerate(res_arr):
        backward = neighbor(axis, -1)
        forward = neighbor(axis, 1)
        row_base = axis * n_mask
        matrix_row = row_base + current

        if scheme == 2:
            backward_2 = neighbor(axis, -2)
            forward_2 = neighbor(axis, 2)

            central4 = (
                (forward > -1)
                & (backward > -1)
                & (forward_2 > -1)
                & (backward_2 > -1)
            )
            forward2 = ~central4 & (forward > -1) & (forward_2 > -1)
            backward2 = ~central4 & ~forward2 & (backward > -1) & (backward_2 > -1)
            forward1 = ~central4 & ~forward2 & ~backward2 & (forward > -1)
            backward1 = (
                ~central4 & ~forward2 & ~backward2 & ~forward1 & (backward > -1)
            )

            rows_c = matrix_row[central4]
            add(rows_c, forward_2[central4], -1.0 / (12.0 * spacing))
            add(rows_c, forward[central4], 8.0 / (12.0 * spacing))
            add(rows_c, backward[central4], -8.0 / (12.0 * spacing))
            add(rows_c, backward_2[central4], 1.0 / (12.0 * spacing))

            rows_f2 = matrix_row[forward2]
            add(rows_f2, forward_2[forward2], -1.0 / (2.0 * spacing))
            add(rows_f2, forward[forward2], 4.0 / (2.0 * spacing))
            add(rows_f2, current[forward2], -3.0 / (2.0 * spacing))

            rows_b2 = matrix_row[backward2]
            add(rows_b2, backward_2[backward2], 1.0 / (2.0 * spacing))
            add(rows_b2, backward[backward2], -4.0 / (2.0 * spacing))
            add(rows_b2, current[backward2], 3.0 / (2.0 * spacing))

            rows_f1 = matrix_row[forward1]
            add(rows_f1, forward[forward1], 1.0 / spacing)
            add(rows_f1, current[forward1], -1.0 / spacing)

            rows_b1 = matrix_row[backward1]
            add(rows_b1, current[backward1], 1.0 / spacing)
            add(rows_b1, backward[backward1], -1.0 / spacing)
            continue

        central = (forward > -1) & (backward > -1) & (scheme == 1)
        forward1 = ~central & (forward > -1)
        backward1 = ~central & ~forward1 & (backward > -1)

        rows_c = matrix_row[central]
        add(rows_c, forward[central], 0.5 / spacing)
        add(rows_c, backward[central], -0.5 / spacing)

        rows_f = matrix_row[forward1]
        add(rows_f, forward[forward1], 1.0 / spacing)
        add(rows_f, current[forward1], -1.0 / spacing)

        rows_b = matrix_row[backward1]
        add(rows_b, current[backward1], 1.0 / spacing)
        add(rows_b, backward[backward1], -1.0 / spacing)

    if rows:
        row_arr = np.concatenate(rows)
        col_arr = np.concatenate(cols)
        data_arr = np.concatenate(data)
    else:
        row_arr = np.array([], dtype=np.int64)
        col_arr = np.array([], dtype=np.int64)
        data_arr = np.array([], dtype=dtype)

    matrix = coo_matrix(
        (data_arr, (row_arr, col_arr)),
        shape=(3 * n_mask, n_mask),
        dtype=dtype,
    ).tocsr()
    return FiniteDifferenceSystem(matrix, mask_indices, shape, order)


def solve_pressure(
    mask: np.ndarray,
    acceleration: np.ndarray,
    res: tuple[float, float, float] | list[float] | np.ndarray,
    *,
    velocity: np.ndarray = None,  # Maintain backward compatibility
    rho: float = 1060.0,
    mu: float = 0.004,
    scheme: Scheme = 1,
    maxiter: int = 3000,
    atol: float = 1e-6,
    btol: float = 1e-6,
    dtype: np.dtype | type = np.float64,
    return_mmhg: bool = False,
) -> np.ndarray:
    """Estimate pressure fields from acceleration using sparse LSMR solves.

    Time-resolved acceleration must have shape ``(3, nx, ny, nz, nt)`` and
    returns pressure with shape ``(nx, ny, nz, nt)``.  A single time point can
    be passed as ``(3, nx, ny, nz)`` and returns pressure with shape
    ``(nx, ny, nz)``.
    """

    dtype = np.dtype(dtype)
    accel = np.asarray(acceleration, dtype=dtype)
    mask_arr = np.asarray(mask, dtype=bool)


    single_time = accel.ndim == 4
    if velocity is not None:
        velocity = np.asarray(velocity, dtype=dtype)
        if velocity.shape != accel.shape:
            raise ValueError("velocity shape must exactly match acceleration shape")

    if accel.ndim not in (4, 5):
        raise ValueError(
            "acceleration must have shape (3, nx, ny, nz) or (3, nx, ny, nz, nt)"
        )
    if accel.shape[0] != 3:
        raise ValueError("first acceleration dimension must contain x, y, z")
    if accel.shape[1:4] != mask_arr.shape:
        raise ValueError("acceleration spatial shape must match mask")

    fd = make_fd_matrix(mask_arr, res, scheme=scheme, dtype=dtype)
    n_mask = fd.mask_indices.size
    n_times = 1 if single_time else accel.shape[4]
    pressure = np.zeros((*mask_arr.shape, n_times), dtype=dtype, order="F")
    flat_pressure = pressure.reshape((-1, n_times), order=fd.order)

    rho_value = dtype.type(rho)
    for t in range(n_times):
        rhs = np.empty(3 * n_mask, dtype=dtype)
        res_arr = np.asarray(res, dtype=dtype)
        mu_value = dtype.type(mu)

        for dim in range(3):
            # Extract 3D fields for the current dimension and time step
            if single_time:
                a_comp = accel[dim]
                v_comp = velocity[dim] if velocity is not None else None
            else:
                a_comp = accel[dim, :, :, :, t]
                v_comp = velocity[dim, :, :, :, t] if velocity is not None else None

            # Start with the inertial force:
            force_comp = a_comp * rho_value

            # Add the viscous force if velocity data is provided
            if v_comp is not None:
                laplacian_v = compute_laplacian_3d(v_comp, res_arr)
                force_comp += mu_value * laplacian_v

            # Flatten, extract only valid mask indices, and assign to RHS vector
            force_flat = force_comp.ravel(order=fd.order)
            rhs_slice = rhs[dim * n_mask: (dim + 1) * n_mask]
            rhs_slice[:] = force_flat[fd.mask_indices]

        solution = lsmr(fd.matrix, rhs, maxiter=maxiter, atol=atol, btol=btol)[0]
        flat_pressure[fd.mask_indices, t] = solution

    if return_mmhg:
        pressure *= dtype.type(PA_TO_MMHG)
    if single_time:
        return pressure[:, :, :, 0]
    return pressure


def pressure_pa_to_mmhg(pressure_pa: np.ndarray) -> np.ndarray:
    """Convert pressure from pascals to millimeters of mercury."""

    return np.asarray(pressure_pa) * PA_TO_MMHG