import pyvista as pv
import numpy as np
import os
import glob
import re

from pressure_estimation import solve_pressure

BASE_DIRECTORY = "/scratch/users/mmegally/healthy_U_bend/24-procs"
RHO = 1060.0  # Blood density [kg/m^3]
MU = 0.004  # Blood dynamic viscosity [Pa*s]
ARRAY_NAME = 'Material_Acceleration_pre_vti'
VEL_ARRAY_NAME = 'Velocity'
OUTPUT_ARRAY = 'Pressure_mmHg_DerivFirst_viscid'


def get_ref_num(filepath):
    """Extracts the timestep number for accurate chronological sorting."""
    match = re.search(r'_ref(\d+)\.vti$', filepath)
    return int(match.group(1)) if match else -1


def process_folder(out_dir, sr_cm):
    """3-Pass Memory-Efficient 5D Pressure Solver"""
    folder_name = os.path.basename(out_dir)
    vti_files = sorted(glob.glob(os.path.join(out_dir, "*.vti")), key=get_ref_num)
    nt = len(vti_files)

    if nt == 0:
        return

    # Dynamic Batching based on spatial resolution
    if sr_cm <= 0.05:
        BATCH_SIZE = 20  # Heaviest memory load, strict batching
    elif sr_cm <= 0.15:
        BATCH_SIZE = 50  # Medium load, moderate batching
    else:
        BATCH_SIZE = nt  # Light load, process whole folder at once

    # Split files into chunks
    for batch_start in range(0, nt, BATCH_SIZE):
        batch_files = vti_files[batch_start: batch_start + BATCH_SIZE]
        current_batch_size = len(batch_files)

        print(f"  [{folder_name}] Processing frames {batch_start} to {batch_start + current_batch_size - 1}...")

        # PASS 1: Read metadata and extract arrays
        grid0 = pv.read(batch_files[0])
        nx, ny, nz = grid0.dimensions
        mask_flat = grid0.point_data['vtkValidPointMask'].astype(bool)
        mask_3d = mask_flat.reshape((nx, ny, nz), order='F')
        del grid0

        accel_5d = np.zeros((3, nx, ny, nz, current_batch_size), dtype=np.float64)
        vel_5d = np.zeros((3, nx, ny, nz, current_batch_size), dtype=np.float64)

        for t, vti_path in enumerate(batch_files):
            grid = pv.read(vti_path)
            accel_flat = grid.point_data[ARRAY_NAME]
            vel_flat = grid.point_data[VEL_ARRAY_NAME]
            for i in range(3):
                accel_5d[i, :, :, :, t] = accel_flat[:, i].reshape((nx, ny, nz), order='F')
                vel_5d[i, :, :, :, t] = vel_flat[:, i].reshape((nx, ny, nz), order='F')

        res_m = [sr_cm / 100.0] * 3
        accel_5d_m = accel_5d / 100.0
        vel_5d_m = vel_5d / 100.0

        # PASS 2: Solve the Pressure
        estimated_pressure_4d = solve_pressure(
            mask=mask_3d,
            acceleration=-accel_5d_m,
            velocity=vel_5d_m,
            res=res_m,
            rho=RHO,
            mu=MU,
            scheme=1,
            return_mmhg=True
        )
        del accel_5d, accel_5d_m, vel_5d, vel_5d_m

        # PASS 3: Stream results back to disk
        for t, vti_path in enumerate(batch_files):
            grid = pv.read(vti_path)
            pressure_3d_t = estimated_pressure_4d[:, :, :, t]

            # Shift the pressures strictly within the mask:
            min_pressure = np.min(pressure_3d_t[mask_3d])
            pressure_3d_t[mask_3d] -= min_pressure

            # Save:
            grid.point_data[OUTPUT_ARRAY] = pressure_3d_t.ravel(order='F')
            grid.save(vti_path)


if __name__ == "__main__":
    print("Starting DERIVATIVE FIRST pressure solver...")

    search_pattern = os.path.join(BASE_DIRECTORY, "Outputs_MatAccel_TR*_SR*")
    output_dirs = glob.glob(search_pattern)

    if not output_dirs:
        print("No output directories found for Derivative First.")

    for out_dir in output_dirs:
        match = re.search(r'SR([0-9.]+)', out_dir)
        if match:
            sr_cm = float(match.group(1))
            process_folder(out_dir, sr_cm)

    print("\nDerivative First pipelines executed successfully!")