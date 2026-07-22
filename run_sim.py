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

_GEO_MODULES = {'lat': 'so_coupling_optics_TR_geometry', 'sat': 'SAT_TR_geometry', 'act': 'ACT_FTS_TR_geometry'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FTS simulation for a single detector position.')
    parser.add_argument('--num-rays', type=int, default=500, help='Number of rays to use in the simulation (default: 500)')
    parser.add_argument('--geometry', choices=['lat', 'sat', 'act'], default='lat',
                        help='Geometry to use: "lat" for so_coupling_optics_TR_geometry (default), "sat" for SAT_TR_geometry, "act" for ACT FTS geometry')
    args = parser.parse_args()
    num_rays = args.num_rays

    # Set env var before importing fts_utils so workers inherit it and _default_geo is correct
    os.environ['FTS_GEO_MODULE'] = _GEO_MODULES[args.geometry]
    import fts_utils  # reads FTS_GEO_MODULE to select geometry

    geo = importlib.import_module(_GEO_MODULES[args.geometry])

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

    # save outputs (change this to change where the output files get saved)
    os.makedirs('sim_outputs/act', exist_ok=True)
    output_file = f'sim_outputs/act/150GHz_{num_rays}r.npz'

    np.savez(output_file, interferogram=interferogram, dm_positions=dm_positions, source_maps=source_maps,
             input_freqs=freqs) # add input_weights=weights if weights is not None, ie if running on a passband input

    # save ray_outputs so generate_interferogram can be re-run at different frequencies
    with open(f'sim_outputs/act/ray_outputs_{num_rays}r.pkl', 'wb') as f:
        pickle.dump(ray_outputs, f)

    t1 = time.time()
    print(f"Done! Total simulation time: {t1 - t0:.2f} seconds")