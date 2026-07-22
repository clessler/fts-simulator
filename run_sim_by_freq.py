'''
Runs the FTS simulation across a sweep of frequencies (local_config.sweep_freqs).
Ray tracing (scan_fts) is performed once; interferogram generation is repeated
per frequency to avoid redundant computation. Outputs saved to
sim_outputs/by_freq/<num_rays>r/<freq_GHz>GHz.npz.
'''
# imports (geometry-independent)
import time
import argparse
import importlib
import os
import pickle
import numpy as np
import ray_tracing as rt

_GEO_MODULES = {'lat': 'so_coupling_optics_TR_geometry', 'sat': 'SAT_TR_geometry', 'act': 'ACT_FTS_TR_geometry'}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run FTS frequency sweep for a single detector position.')
    parser.add_argument('--num-rays', type=int, default=500, help='Number of rays to use in the simulation (default: 500)')
    parser.add_argument('--geometry', choices=['lat', 'sat', 'act'], default='lat',
                        help='Geometry to use: "lat" for so_coupling_optics_TR_geometry (default), "sat" for SAT_TR_geometry, "act" for ACT FTS geometry')
    args = parser.parse_args()
    num_rays = args.num_rays

    # Set env var before importing fts_utils so workers inherit it and _default_geo is correct
    os.environ['FTS_GEO_MODULE'] = _GEO_MODULES[args.geometry]
    import fts_utils  # reads FTS_GEO_MODULE to select geometry

    geo = importlib.import_module(_GEO_MODULES[args.geometry])

    try:
        import local_config as cfg
    except ImportError:
        raise ImportError(
            "local_config.py not found. Copy local_config.example.py to "
            "local_config.py and fill in your machine-specific paths/parameters."
        )

    beam_data = np.loadtxt(cfg.beam_file)
    xpos, ypos, theta_bound = cfg.xpos, cfg.ypos, cfg.theta_bound
    FTS_throw, FTS_step = cfg.FTS_throw, cfg.FTS_step
    freqs = cfg.sweep_freqs

    out_dir = f'sim_outputs/by_freq/{num_rays}r'
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    starting_rays = rt.generate_rays_from_beam_data(
        geo.fp, [xpos, ypos], beam_data=beam_data, theta_bound=theta_bound,
        method='random', num_rays=num_rays, direction_sign='auto',
        direction_reference=geo.stop)

    print("Starting ray trace (scan_fts)...")
    ray_outputs = fts_utils.scan_fts(starting_rays, FTS_throw, FTS_step, debug=True)

    ray_outputs_path = os.path.join(out_dir, f'ray_outputs_{num_rays}r.pkl')
    with open(ray_outputs_path, 'wb') as f:
        pickle.dump(ray_outputs, f)
    print(f"Ray outputs saved to {ray_outputs_path}")

    for freq in freqs:
        freq_ghz = freq / 1e9
        print(f"\nGenerating interferogram for {freq_ghz:.4g} GHz...")
        interferogram, dm_positions = fts_utils.generate_interferogram( # type: ignore
            ray_outputs, geo.source, np.array([freq]), FTS_throw, FTS_step, return_maps=False)

        out_file = os.path.join(out_dir, f'{freq_ghz:.4g}GHz.npz')
        np.savez(out_file, interferogram=interferogram, dm_positions=dm_positions, freq_hz=freq)
        print(f"  Saved to {out_file}")

    print(f"\nAll done. Total time: {time.time() - t0:.2f} seconds")
