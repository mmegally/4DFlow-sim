import pyvista as pv
import numpy as np
import os


def preload_cfd_arrays(base_dir, start_t, end_t, step_t, arrays_to_process):
    """
    Reads the VTU files once and caches the numpy arrays in memory.
    """
    print(f"\n Pre-loading CFD data into memory ({start_t} to {end_t})")
    data_cache = {}
    base_mesh = None

    for t in range(start_t, end_t + step_t, step_t):
        t_str = f"{t:03d}" if t < 100 else str(t)
        filename = f"result_{t_str}.vtu"
        full_path = os.path.join(base_dir, filename)

        if os.path.exists(full_path):
            mesh = pv.read(full_path)
            if base_mesh is None:
                base_mesh = mesh.copy()
            data_cache[t] = {arr: mesh.point_data[arr].copy() for arr in arrays_to_process}
        else:
            print(f"Warning: {filename} could not be found.")

    print(f"Successfully loaded {len(data_cache)} CFD frames into memory.")
    return data_cache, base_mesh


def process_resolution_set(
        base_dir,
        data_cache,
        base_mesh,
        tr_ms,
        sr_cm,
        data_start,
        data_end,
        dt_cfd_ms,
        arrays_to_process
):
    """
    Averages cached data temporally within strictly available bounds,
    and resamples spatially.
    """
    # MODIFICATION: Changed output directory name to prevent overwriting old data
    out_dir = os.path.join(base_dir, f"Outputs_MatAccel_TR{int(tr_ms)}_SR{sr_cm}")
    os.makedirs(out_dir, exist_ok=True)

    # Determine where to start (temporally) based on range of data:
    frames_to_average = int(tr_ms / dt_cfd_ms)
    half_window = frames_to_average // 2
    step = int(dt_cfd_ms)

    time_offset = half_window * step

    # The earliest and latest times we can center on without exceeding our data bounds
    safe_ref_start = data_start + time_offset
    safe_ref_end = data_end - time_offset

    print(f"\n Processing Set: Temporal = {tr_ms} ms | Spatial = {sr_cm} cm")
    print(f"Safe Reference Window: {safe_ref_start} ms to {safe_ref_end} ms")

    # Spatial Grid Generation
    bounds = base_mesh.bounds
    nx = int(np.ceil((bounds[1] - bounds[0]) / sr_cm)) + 1
    ny = int(np.ceil((bounds[3] - bounds[2]) / sr_cm)) + 1
    nz = int(np.ceil((bounds[5] - bounds[4]) / sr_cm)) + 1

    target_grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(sr_cm, sr_cm, sr_cm),
        origin=(bounds[0], bounds[2], bounds[4])
    )

    # Calculate Reference Timesteps using np.arange to ensure exact tr_ms spacing
    reference_timesteps = np.arange(safe_ref_start, safe_ref_end + 1, tr_ms).astype(int)

    saved_count = 0
    for ref_t in reference_timesteps:
        start_t = ref_t - time_offset
        end_t = ref_t + time_offset

        # Collect keys strictly within our standard window
        valid_keys = [t for t in range(start_t, end_t + step, step) if t in data_cache]

        if not valid_keys:
            continue

        actual_frames = len(valid_keys)

        averaged_data = {
            arr: np.zeros_like(data_cache[valid_keys[0]][arr], dtype=np.float64)
            for arr in arrays_to_process
        }

        # Accumulate
        for t_key in valid_keys:
            for arr in arrays_to_process:
                averaged_data[arr] += data_cache[t_key][arr]

        # Average and apply to base mesh
        for arr in arrays_to_process:
            base_mesh.point_data[arr] = averaged_data[arr] / actual_frames

        # Resample and save
        resampled_grid = target_grid.sample(base_mesh, pass_cell_data=False)
        output_filename = os.path.join(out_dir, f"simulated_4dflow_TR{int(tr_ms)}_SR{sr_cm}_ref{ref_t}.vti")
        resampled_grid.save(output_filename)
        saved_count += 1

    print(f"Completed! Saved {saved_count} .vti files to {out_dir}")


if __name__ == "__main__":
    BASE_DIRECTORY = "/scratch/users/mmegally/healthy_U_bend/24-procs"

    DATA_START = 3440
    DATA_END = 4300
    CFD_DT = 2.0

    # Added material acceleration array
    ARRAYS = ['Velocity', 'Acceleration', 'Pressure', 'Material_Acceleration_pre_vti']

    temporal_resolutions_ms = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    spatial_resolutions_cm = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    try:
        # Preload the entire available cycle strictly
        cached_data, base_geometry = preload_cfd_arrays(
            base_dir=BASE_DIRECTORY,
            start_t=DATA_START,
            end_t=DATA_END,
            step_t=int(CFD_DT),
            arrays_to_process=ARRAYS
        )

        if not cached_data:
            raise ValueError("No data loaded. Check your base directory and file naming.")

        # Experiment 1: Vary Temporal Resolution
        print("\n" + "=" * 40 + "\nRUNNING EXPERIMENT 1: VARYING TEMPORAL\n" + "=" * 40)
        fixed_sr = 0.05
        for tr in temporal_resolutions_ms:
            process_resolution_set(
                base_dir=BASE_DIRECTORY, data_cache=cached_data, base_mesh=base_geometry,
                tr_ms=tr, sr_cm=fixed_sr, data_start=DATA_START, data_end=DATA_END,
                dt_cfd_ms=CFD_DT, arrays_to_process=ARRAYS
            )

        # Experiment 2: Vary Spatial Resolution
        print("\n" + "=" * 40 + "\nRUNNING EXPERIMENT 2: VARYING SPATIAL\n" + "=" * 40)
        fixed_tr = 2.0
        for sr in spatial_resolutions_cm:
            if sr == 0.05:  # Skip sr = 0.5mm and tr = 2ms (already ran)
                continue

            process_resolution_set(
                base_dir=BASE_DIRECTORY, data_cache=cached_data, base_mesh=base_geometry,
                tr_ms=fixed_tr, sr_cm=sr, data_start=DATA_START, data_end=DATA_END,
                dt_cfd_ms=CFD_DT, arrays_to_process=ARRAYS
            )

        print("\nAll parameter sweeps finished successfully!")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")