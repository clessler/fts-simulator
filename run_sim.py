'''
Script to to simulate the FTS scanning & produce an interferogram for 1 detector position.

If num_rays is not specified, defaults to 500 rays.
Run like so (for 1000 rays): python run_sim.py --num-rays 1000

Inputs: 
- FTS parameters (e.g. scan speed, sampling rate, etc.)
- Detector position (e.g. x, y coordinates)
- Beam data
- Number of rays to use
- File describing the geometry of the FTS & optical system

Outputs:
- Interferogram for the specified detector position
- Array of mirror positions corresponding to the interferogram
- Array of intensity maps at the source for each mirror position (optional)
'''
# imports (geometry-independent)
import argparse
import importlib
import os
import time
import pickle
import numpy as np
import ray_tracing as rt

_GEO_MODULES = {'lat': 'geometry_files.so_coupling_optics_TR_geometry', 'sat': 'geometry_files.SAT_TR_geometry', 'act': 'geometry_files.ACT_FTS_TR_geometry'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FTS simulation for a single detector position.')
    parser.add_argument('--num-rays', type=int, default=500, help='Number of rays to use in the simulation (default: 500)')
    parser.add_argument('--geometry', default=None,
                        help='Geometry to use: preset key ("lat", "sat", "act") or a dotted module path '
                             '(e.g. "geometry_files.so_lat_iris_test_geo"). Overrides local_config.geometry '
                             'if set; defaults to "lat" if neither is specified.')
    args = parser.parse_args()
    num_rays = args.num_rays

    # machine/run-specific parameters (beam file, detector position, scan
    # params, frequencies) live in local_config.py, which is gitignored —
    # see local_config.example.py for the template.
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
    freqs, weights = cfg.freqs, cfg.weights

    t0 = time.time()

    # generate starting rays
    starting_rays = rt.generate_rays_from_beam_data(
        geo.fp, [xpos, ypos], beam_data=beam_data, theta_bound=theta_bound, method='random', num_rays=num_rays, direction_sign='auto', direction_reference=geo.stop)

    # scan the FTS
    ray_outputs = fts_utils.scan_fts(starting_rays, FTS_throw, FTS_step, debug=True)

    # generate interferogram (most time-intensive)
    interferogram, dm_positions, source_maps = fts_utils.generate_interferogram(ray_outputs, geo.source, freqs, FTS_throw, FTS_step, return_maps=True, weights=weights) # type: ignore

    # save outputs (set local_config.output_path/output_filename to change where/what files get saved)
    os.makedirs(cfg.output_path, exist_ok=True)
    output_file = os.path.join(cfg.output_path, cfg.output_filename.format(num_rays=num_rays, **vars(cfg)))

    np.savez(output_file, interferogram=interferogram, dm_positions=dm_positions, source_maps=source_maps,
             input_freqs=freqs) # add input_weights=weights if weights is not None, ie if running on a passband input

    # save ray_outputs so generate_interferogram can be re-run at different frequencies
    if cfg.save_ray_outputs:
        with open(os.path.join(cfg.output_path, f'ray_outputs_{num_rays}r.pkl'), 'wb') as f:
            pickle.dump(ray_outputs, f)

    t1 = time.time()
    print(f"Done! Total simulation time: {t1 - t0:.2f} seconds")