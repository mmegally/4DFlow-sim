import pyvista as pv
import numpy as np
import os
import glob
import re

from vel_accel import velocity_to_acceleration


def compute_material_acceleration_for_directory(vti_dir, tr_ms, sr_cm):
    """
    Reads a directory of 4D Flow .vti files, constructs the 5D velocity array in batches,
    computes the material derivative, and saves the acceleration back to the files.
    """
    print(f"\n--- Processing Directory: {vti_dir} ---")

    # Find and sort all .vti files temporally:
    search_pattern = os.path.join(vti_dir, "*.vti")
    vti_files = glob.glob(search_pattern)

    if not vti_files:
        print(f"No .vti files found in {vti_dir}.")
        return

    # Sort files by their reference timestep to ensure correct temporal ordering:
    def extract_ref_time(filepath):
        match = re.search(r'_ref(\d+)\.vti$', filepath)
        return int(match.group(1)) if match else 0

    vti_files.sort(key=extract_ref_time)

    # Extract Grid Dimensions from the first file:
    base_mesh = pv.read(vti_files[0])
    nx, ny, nz = base_mesh.dimensions
    nt = len(vti_files)

    print(f"Found {nt} timeframes. Grid dimensions: {nx}x{ny}x{nz}")

    # Velocity is in cm/s: convert TR [ms] to [s]:
    dt_seconds = tr_ms / 1000.0

    print("Computing material acceleration via rolling temporal window...")
    for i in range(nt):
        # Determine the 3-frame window for the central difference:
        if i == 0:
            window_indices = [0, 1, min(2, nt - 1)]
            target_out_idx = 0
        elif i == nt - 1:
            window_indices = [max(0, nt - 3), nt - 2, nt - 1]
            target_out_idx = 2
        else:
            window_indices = [i - 1, i, i + 1]
            target_out_idx = 1

        # Construct the chunked 5D Velocity Array
        velocity_chunk = np.zeros((3, nx, ny, nz, len(window_indices)), dtype=np.float64)

        for chunk_idx, file_idx in enumerate(window_indices):
            mesh = pv.read(vti_files[file_idx])
            # Extract velocity, transpose to (3, n_points), then reshape
            vel_flat = mesh.point_data['Velocity'].T
            velocity_chunk[:, :, :, :, chunk_idx] = vel_flat.reshape((3, nx, ny, nz), order='F')

        # Compute Material Derivative with Charles' library:
        accel_chunk = velocity_to_acceleration(
            velocity=velocity_chunk,
            res=(sr_cm, sr_cm, sr_cm),
            dt=dt_seconds,
            spatial_scheme=1
        )

        # Extract the specific timestep (3, nx, ny, nz)
        accel_3d = accel_chunk[:, :, :, :, target_out_idx]

        # Flatten back to (n_points, 3) for PyVista
        accel_flat = accel_3d.reshape((3, -1), order='F').T

        # Save Back to .vti
        curr_mesh = pv.read(vti_files[i])

        # Name output 'Material_Acceleration' to avoid overwriting Eulerian acceleration:
        curr_mesh.point_data['Material_Acceleration_post_vti'] = accel_flat
        curr_mesh.save(vti_files[i])

        # Progress tracking (every 25 frames):
        if (i + 1) % 25 == 0 or i == nt - 1:
            print(f"Processed and saved {i + 1}/{nt} frames...")

    print("Success!")


if __name__ == "__main__":
    BASE_DIRECTORY = "/scratch/users/mmegally/healthy_U_bend/24-procs"

    # Processing all SR with TR 2 ms
    TR_TEST = 2
    SR_LIST = [0.1, 0.15, 0.20, 0.25, 0.30]

    for SR_TEST in SR_LIST:
        target_folder = os.path.join(BASE_DIRECTORY, f"Outputs_TR{int(TR_TEST)}_SR{SR_TEST}")

        # Run the integration
        compute_material_acceleration_for_directory(
            vti_dir=target_folder,
            tr_ms=TR_TEST,
            sr_cm=SR_TEST
        )
