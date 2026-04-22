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
# imports
import argparse
import time
import os
import pickle
import numpy as np
import ray_tracing as rt
import fts_utils
import so_coupling_optics_TR_geometry as geo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FTS simulation for a single detector position.')
    parser.add_argument('--num-rays', type=int, default=500, help='Number of rays to use in the simulation (default: 500)')
    args = parser.parse_args()
    num_rays = args.num_rays

    # input beam file here
    # note: might want to auto-configure so that different beam files get used depending on my command-line argument (e.g. which frequency band I'm simulating for)
    # insert your beam file path here:
    beam_file = '/home/clessler/instrument_model/instrument_hardware/simulated_beam/simulated_feedhorn_beams/MF/150_GHz_weighted.txt'
    beam_data = np.loadtxt(beam_file)

    # set starting position & beam angle bounds for ray generation
    xpos, ypos = -53.947736,1.34448473
    theta_bound = 15  # degrees, bounds for ray angles relative to the optical axis

    # set FTS scan parameters
    FTS_throw = 20 # mm, total (positive) mirror throw
    FTS_step = 0.1 # mm, step size for mirror position

    # input frequency or array of frequencies here
    freqs = np.array([150e9])  # Hz, example frequency for simulation (150 GHz)

    t0 = time.time()

    # generate starting rays
    starting_rays = rt.generate_rays_from_beam_data(
        geo.fp, [xpos, ypos], beam_data=beam_data, theta_bound=theta_bound, method='random', num_rays=num_rays, direction_sign='auto', direction_reference=geo.stop)

    # scan the FTS
    ray_outputs = fts_utils.scan_fts(starting_rays, FTS_throw, FTS_step, debug=True)

    # generate interferogram (most time-intensive)
    interferogram, dm_positions, source_maps = fts_utils.generate_interferogram(ray_outputs, geo.source, freqs, FTS_throw, FTS_step, return_maps=True) # type: ignore

    # save outputs
    os.makedirs('sim_outputs', exist_ok=True)
    output_file = f'sim_outputs/150GHz_{num_rays}r.npz'

    np.savez(output_file, interferogram=interferogram, dm_positions=dm_positions, source_maps=source_maps)

    # save ray_outputs so generate_interferogram can be re-run at different frequencies
    with open(f'sim_outputs/ray_outputs_{num_rays}r.pkl', 'wb') as f:
        pickle.dump(ray_outputs, f)

    t1 = time.time()
    print(f"Done! Total simulation time: {t1 - t0:.2f} seconds")