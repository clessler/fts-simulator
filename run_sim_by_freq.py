'''
Runs the FTS simulation for each frequency in np.linspace(128e9, 168e9, 10).
Ray tracing (scan_fts) is performed once; interferogram generation is repeated
per frequency to avoid redundant computation. Outputs saved to
sim_outputs/by_freq/500r/<freq_GHz>GHz.npz.
'''
import time
import argparse
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

    beam_file = '/home/clessler/instrument_model/instrument_hardware/simulated_beam/simulated_feedhorn_beams/MF/150_GHz_weighted.txt'
    beam_data = np.loadtxt(beam_file)

    xpos, ypos = -53.947736, 1.34448473
    theta_bound = 15

    FTS_throw = 20
    FTS_step = 0.1

    freqs = np.linspace(128e9, 168e9, 10)

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
