import pyvista as pv
import numpy as np
import os
import glob
import re


def compute_material_acceleration_vtu(vtu_dir):
    """
    Reads 4D Flow .vtu files, extracts the native solver acceleration,
    computes the advective term using mesh shape functions, and
    saves the total material acceleration back to the files.
    """
    print(f"\n--- Processing Directory: {vtu_dir} ---")

    # Find and sort all .vtu files:
    search_pattern = os.path.join(vtu_dir, "result_*.vtu")
    vtu_files = glob.glob(search_pattern)

    if not vtu_files:
        print(f"No files matching 'result_*.vtu' found in {vtu_dir}.")
        return

    # Sort files by their timestamp:
    def extract_time(filepath):
        match = re.search(r'result_(\d+)\.vtu$', filepath)
        return int(match.group(1)) if match else 0

    vtu_files.sort(key=extract_time)
    nt = len(vtu_files)

    # Base dataset just to get number of points
    base_mesh = pv.read(vtu_files[0])
    n_points = base_mesh.n_points
    print(f"Found {nt} timeframes. Unstructured grid with {n_points} points.")

    print("Computing material acceleration using native solver acceleration...")
    for i, filepath in enumerate(vtu_files):

        # Load the mesh
        mesh = pv.read(filepath)

        # Extract native Velocity and Acceleration:
        v = mesh.point_data['Velocity']
        eulerian_accel = mesh.point_data['Acceleration']

        # Compute Spatial Gradient using mesh shape functions
        grad_mesh = mesh.compute_derivative(scalars='Velocity', gradient=True)
        G = grad_mesh.point_data['gradient']

        # Compute Advective Term: (v . grad)v
        advective = np.zeros_like(v)  # set shape for advective data

        # X-component: u(du/dx) + v(du/dy) + w(du/dz)
        advective[:, 0] = v[:, 0] * G[:, 0] + v[:, 1] * G[:, 1] + v[:, 2] * G[:, 2]
        # Y-component: u(dv/dx) + v(dv/dy) + w(dv/dz)
        advective[:, 1] = v[:, 0] * G[:, 3] + v[:, 1] * G[:, 4] + v[:, 2] * G[:, 5]
        # Z-component: u(dw/dx) + v(dw/dy) + w(dw/dz)
        advective[:, 2] = v[:, 0] * G[:, 6] + v[:, 1] * G[:, 7] + v[:, 2] * G[:, 8]

        # Total Material Acceleration:
        material_acceleration = eulerian_accel + advective

        # Save back to the current .vtu file
        mesh.point_data['Material_Acceleration_pre_vti'] = material_acceleration
        mesh.save(filepath)

        if (i + 1) % 25 == 0 or i == nt - 1:
            print(f"Processed and saved {i + 1}/{nt} frames...")

    print("Success!")


if __name__ == "__main__":
    # Update to your actual scratch directory path
    BASE_DIRECTORY = "/scratch/users/mmegally/healthy_U_bend/24-procs"

    compute_material_acceleration_vtu(
        vtu_dir=BASE_DIRECTORY
    )

