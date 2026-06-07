"""Convert velocity fields to acceleration through the material derivative."""

from __future__ import annotations
import numpy as np
from pressure_estimation import Scheme, make_fd_matrix


def velocity_to_acceleration(
    velocity: np.ndarray,
    res: tuple[float, float, float] | list[float] | np.ndarray,
    dt: float,
    *,
    mask: np.ndarray | None = None,
    spatial_scheme: Scheme = 1,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Compute ``Dv/Dt = dv/dt + (v . grad)v`` from velocity data.

    ``velocity`` may be time-resolved with shape ``(3, nx, ny, nz, nt)`` or a
    single steady time point with shape ``(3, nx, ny, nz)``.  The returned
    acceleration has the same rank: 5D input returns ``(3, nx, ny, nz, nt)``,
    while 4D input returns ``(3, nx, ny, nz)``.

    Spatial derivatives are evaluated only inside ``mask``.  If no mask is
    supplied, the full spatial domain is used.  The temporal derivative uses
    central differences in the interior and first-order one-sided differences
    at the first and last time points.  For a single time point, ``dv/dt`` is
    zero and only the advective term is returned.
    """

    dtype = np.dtype(dtype)
    vel = np.asarray(velocity, dtype=dtype)

    single_time = vel.ndim == 4
    if vel.ndim not in (4, 5):
        raise ValueError("velocity must have shape (3, nx, ny, nz) or (3, nx, ny, nz, nt)")
    if vel.shape[0] != 3:
        raise ValueError("first velocity dimension must contain x, y, z")
    if dt <= 0:
        raise ValueError("dt must be positive")

    spatial_shape = vel.shape[1:4]
    if mask is None:
        mask_arr = np.ones(spatial_shape, dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != spatial_shape:
            raise ValueError("mask shape must match velocity spatial shape")

    fd = make_fd_matrix(mask_arr, res, scheme=spatial_scheme, dtype=dtype)
    n_mask = fd.mask_indices.size
    n_times = 1 if single_time else vel.shape[4]

    velocity_masked = np.empty((3, n_mask, n_times), dtype=dtype)
    for dim in range(3):
        for t in range(n_times):
            if single_time:
                component = vel[dim].ravel(order=fd.order)
            else:
                component = vel[dim, :, :, :, t].ravel(order=fd.order)
            velocity_masked[dim, :, t] = component[fd.mask_indices]

    acceleration = np.zeros((3, *spatial_shape, n_times), dtype=dtype, order="F")
    flat_acceleration = acceleration.reshape((3, -1, n_times), order=fd.order)
    temporal = _temporal_derivative(velocity_masked, dtype.type(dt), dtype=dtype)

    for t in range(n_times):
        advective = np.zeros((3, n_mask), dtype=dtype)
        velocity_t = velocity_masked[:, :, t]

        for component in range(3):
            gradient = fd.matrix @ velocity_t[component]
            gradient = gradient.reshape((3, n_mask))
            advective[component] = np.sum(velocity_t * gradient, axis=0)

        flat_acceleration[:, fd.mask_indices, t] = temporal[:, :, t] + advective

    if single_time:
        return acceleration[:, :, :, :, 0]
    return acceleration


def _temporal_derivative(
    velocity_masked: np.ndarray,
    dt: float,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    derivative = np.zeros_like(velocity_masked, dtype=dtype)
    n_times = velocity_masked.shape[2]

    if n_times == 1:
        return derivative
    if n_times == 2:
        derivative[:, :, :] = (velocity_masked[:, :, 1:2] - velocity_masked[:, :, 0:1]) / dt
        return derivative

    derivative[:, :, 0] = (velocity_masked[:, :, 1] - velocity_masked[:, :, 0]) / dt
    derivative[:, :, -1] = (velocity_masked[:, :, -1] - velocity_masked[:, :, -2]) / dt
    derivative[:, :, 1:-1] = (
        velocity_masked[:, :, 2:] - velocity_masked[:, :, :-2]
    ) / (2.0 * dt)
    return derivative