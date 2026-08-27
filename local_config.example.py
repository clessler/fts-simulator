'''
Template for local, machine/run-specific parameters used by run_sim.py.

Copy this file to local_config.py (gitignored, not tracked) and fill in
your real paths/values for every simulation run.
'''
import numpy as np

''' --- geometry --- '''
# The optical geometry to use for the simulation run. Either a preset key ('lat', 'sat', 'act' —
# same as the --geometry CLI choices) or a full dotted module path for a
#  geometry file not in the preset list, e.g.
# 'geometry_files.so_lat_iris_test_geo'.
# Overridden by --geometry on the CLI if that flag is passed.
geometry = 'geometry_files.your_geometry_file'

# Ordered list of Detector attribute names on the geometry module, defining the
# propagation chain from the ray-traced eval_plane through zero or more
# intermediate irises/aperture stops to the final source plane. Each name is
# resolved via getattr(geo, name); all listed Detectors must share the same
# `resolution`. Defaults to ['source'] (no iris) if unset.
# detector_chain = ['source']
#
# One iris:
# detector_chain = ['iris_detector', 'source']
#
# Multiple irises (geometry module must define each Detector object):
# detector_chain = ['iris_detector_1', 'iris_detector_2', 'source']

# If True, save a separate source map for every detector in detector_chain
# (keyed 'source_maps_<name>' in the output .npz) instead of just the final
# one. Useful for comparing the field before/after each iris. Defaults to
# False (single 'source_maps' key, matching today's output format).
# save_stage_maps = False

''' --- beam data --- '''
# add in your beam intensity profile here
beam_file = '/path/to/instrument_model/simulated_beam.txt'

''' --- detector position & beam angle bounds --- '''
# fill in the x/y position from which to launch rays and the cone-angle to launch them into
# lat params
xpos, ypos = -53.947736, 1.34448473
theta_bound = 15  # degrees, bounds for ray angles relative to the optical axis

# sat params
# xpos, ypos = 0, 0
# theta_bound = 25

# act params
# xpos, ypos = 0, 0
# theta_bound = 12

''' --- FTS scan parameters --- '''
# add the parameters of the FTS scan you want to simulate
FTS_throw = 37.5 # mm (this means the FTS scans over +/- 37.5mm)
FTS_step = 0.15

''' --- input frequency or array of frequencies --- '''
freqs = np.array([150e9])  # monochromatic 150 GHz
weights = None  # for monochromatic input, all rays have equal weight

# sample for simulating a broadband detector: load in passband and use non-zero entries as weighted input
# passband_file = '/path/to/bolocalc-so-model/V4r0/V4r0_Goal/SAT/bands/detectors/MF_2.txt'
# passband_data = np.loadtxt(passband_file)
# passband_freqs = passband_data[:, 0] * 1e9  # GHz -> Hz
# passband_weights = passband_data[:, 1]
# mask = passband_weights > 0
# subsample_fac = 2  # sub-sampling to speed this up
# freqs = passband_freqs[mask][::subsample_fac]
# weights = passband_weights[mask][::subsample_fac]

''' --- frequency sweep (used by run_sim_by_freq.py) --- '''
# if running run_sim_by_freq, add the evaluation frequencies you want to use here
sweep_freqs = np.linspace(128e9, 168e9, 10)

''' --- output --- '''
# directory run_sim.py saves its interferogram/ray_outputs files into
output_path = 'sim_outputs/your_geometry'

# filename run_sim.py uses for its interferogram outputs. {num_rays} is filled in at run time.
output_filename = '150GHz_{num_rays}r.npz'

# whether run_sim.py also saves ray_outputs (allows
# generate_interferogram be re-run at different frequencies without
# re-tracing rays) - these files can be large, so turn off if not needed
save_ray_outputs = True

# directory run_sim_by_freq.py saves each frequency's interferogram output into
output_path_by_freq = 'sim_outputs/by_freq/{num_rays}r'

# filename run_sim_by_freq.py uses for each frequency's interferogram output ({freq_ghz} filled in at run time)
output_filename_by_freq = '{freq_ghz:.4g}GHz.npz'

# whether run_sim_by_freq.py also saves ray outputs (as ray_outputs_<num_rays>r.pkl)
save_ray_outputs_by_freq = True
