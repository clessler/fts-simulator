'''
Runs the FTS simulation across a sweep of frequencies (local_config.sweep_freqs).
Ray tracing (scan_fts) is performed once; interferogram generation is repeated
per frequency to avoid redundant computation. Outputs saved to
local_config.output_path_by_freq/<freq_GHz>GHz.npz.
'''
# imports (geometry-independent)
import time
import argparse
import importlib
import os
import pickle
import numpy as np
import ray_tracing as rt

_GEO_MODULES = {'lat': 'geometry_files.so_coupling_optics_TR_geometry', 'sat': 'geometry_files.SAT_TR_geometry', 'act': 'geometry_files.ACT_FTS_TR_geometry'}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run FTS frequency sweep for a single detector position.')
    parser.add_argument('--num-rays', type=int, default=500, help='Number of rays to use in the simulation (default: 500)')
    parser.add_argument('--geometry', default=None,
                        help='Geometry to use: preset key ("lat", "sat", "act") or a dotted module path '
                             '(e.g. "geometry_files.so_lat_iris_test_geo"). Overrides local_config.geometry '
                             'if set; defaults to "lat" if neither is specified.')
    args = parser.parse_args()
    num_rays = args.num_rays

    try:
        import local_config as cfg
    except ImportError:
        raise ImportError(
            "local_config.py not found. Copy local_config.example.py to "
            "local_config.py and fill in your machine-specific paths/parameters."
        )

    geometry_setting = args.geometry or getattr(cfg, 'geometry', 'lat')
    geo_module_path = _GEO_MODULES.get(geometry_setting, geometry_setting)

    # Set env var before importing fts_utils so workers inherit it and _default_geo is correct
    os.environ['FTS_GEO_MODULE'] = geo_module_path
    import fts_utils  # reads FTS_GEO_MODULE to select geometry

    geo = importlib.import_module(geo_module_path)

    beam_data = np.loadtxt(cfg.beam_file)
    xpos, ypos, theta_bound = cfg.xpos, cfg.ypos, cfg.theta_bound
    FTS_throw, FTS_step = cfg.FTS_throw, cfg.FTS_step
    freqs = cfg.sweep_freqs

    detector_names = getattr(cfg, 'detector_chain', ['source'])
    detectors = [getattr(geo, name) for name in detector_names]

    out_dir = cfg.output_path_by_freq.format(num_rays=num_rays, **vars(cfg))
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    starting_rays = rt.generate_rays_from_beam_data(
        geo.fp, [xpos, ypos], beam_data=beam_data, theta_bound=theta_bound,
        method='random', num_rays=num_rays, direction_sign='auto',
        direction_reference=geo.stop)

    print("Starting ray trace (scan_fts)...")
    ray_outputs = fts_utils.scan_fts(starting_rays, FTS_throw, FTS_step, debug=True)

    if cfg.save_ray_outputs_by_freq:
        ray_outputs_path = os.path.join(out_dir, f'ray_outputs_{num_rays}r.pkl')
        with open(ray_outputs_path, 'wb') as f:
            pickle.dump(ray_outputs, f)
        print(f"Ray outputs saved to {ray_outputs_path}")

    for freq in freqs:
        freq_ghz = freq / 1e9
        print(f"\nGenerating interferogram for {freq_ghz:.4g} GHz...")
        interferogram, dm_positions = fts_utils.generate_interferogram( # type: ignore
            ray_outputs, detectors, np.array([freq]), FTS_throw, FTS_step, return_maps=False)

        out_file = os.path.join(out_dir, cfg.output_filename_by_freq.format(freq_ghz=freq_ghz, num_rays=num_rays, **vars(cfg)))
        np.savez(out_file, interferogram=interferogram, dm_positions=dm_positions, freq_hz=freq)
        print(f"  Saved to {out_file}")

    print(f"\nAll done. Total time: {time.time() - t0:.2f} seconds")
