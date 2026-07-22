'''
Template for local, machine/run-specific parameters used by run_sim.py.

Copy this file to local_config.py (gitignored, not tracked) and fill in
your real paths/values. That way tweaking a beam file path or a scan
parameter for a local run never shows up as a change to tracked code.
'''
import numpy as np

# --- beam data ---
# don't have teraflash beam profile so use feedhorn profile (approx. Gaussian) for now
beam_file = '/path/to/instrument_model/instrument_hardware/simulated_beam/simulated_feedhorn_beams/MF/150_GHz_weighted.txt'

# --- detector position & beam angle bounds ---
# lat params
# xpos, ypos = -53.947736, 1.34448473
# theta_bound = 15  # degrees, bounds for ray angles relative to the optical axis

# sat params
# xpos, ypos = 0, 0
# theta_bound = 25  # degrees, bounds for ray angles relative to the optical axis

# act params
xpos, ypos = 0, 0
theta_bound = 12  # degrees

# --- FTS scan parameters ---
# sim params for MF scans
# FTS_throw = 20  # mm, total (positive) mirror throw
# FTS_step = 0.1  # mm, step size for mirror position

# realistic MF scan params
FTS_throw = 37.5
FTS_step = 0.15

# --- input frequency or array of frequencies ---
freqs = np.array([150e9])  # monochromatic 150 GHz
weights = None  # for monochromatic input, all rays have equal weight

# broadband: load MF2 passband and use non-zero entries as weighted input
# passband_file = '/path/to/bolocalc-so-model/V4r0/V4r0_Goal/SAT/bands/detectors/MF_2.txt'
# passband_data = np.loadtxt(passband_file)
# passband_freqs = passband_data[:, 0] * 1e9  # GHz -> Hz
# passband_weights = passband_data[:, 1]
# mask = passband_weights > 0
# subsample_fac = 2  # sub-sampling to speed this up
# freqs = passband_freqs[mask][::subsample_fac]
# weights = passband_weights[mask][::subsample_fac]

# --- frequency sweep (used by run_sim_by_freq.py) ---
sweep_freqs = np.linspace(128e9, 168e9, 10)
