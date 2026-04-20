# Code to simulate the scanning of the FTS.
# FTS, OT & coupling optics geometry defined in so_coupling_optics_TR_geometry.py
# Includes utilities to trace rays through the full optical system.

import ray_tracing as rt
import numpy as np
import copy
import so_coupling_optics_TR_geometry as geo
import diffraction as diff
from tqdm import tqdm
import time
from multiprocessing import Pool
# from so_coupling_optics_TR_geometry import optical_system, dm, ell_3, ell_4, ell_5, ell_6

c = 2.998e11 # speed of light in mm/s


'''Utilities for FTS scanning and interferogram generation.'''

def _nan_ray_record():
    return [np.nan, np.nan, np.full(3, np.nan), np.full(3, np.nan), np.nan]


def _as_ray_matrix(rays):
    arr = np.asarray(rays, dtype=object)
    if arr.size == 0:
        return np.empty((0, 5), dtype=object)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _pad_rays_to_rectangular(ray_lists):
    mats = [_as_ray_matrix(rays) for rays in ray_lists]
    n_pos = len(mats)
    max_rays = max((m.shape[0] for m in mats), default=0)
    if max_rays == 0:
        return np.empty((n_pos, 0, 5), dtype=object)

    pad_row = np.array(_nan_ray_record(), dtype=object).reshape(1, 5)
    padded = []
    for rays_arr in mats:
        n_pad = max_rays - rays_arr.shape[0]
        if n_pad > 0:
            rays_arr = np.vstack((rays_arr, np.repeat(pad_row, n_pad, axis=0)))
        padded.append(rays_arr)
    return np.stack(padded, axis=0)

# Function to produce an optical system with just the FTS and with the dm at a specified position
def get_optical_system_with_dm_pos(pos):
    dm_new = copy.copy(geo.dm)  # shallow copy; rotation matrices are unchanged since only origin moves
    dm_new.origin = (0, pos, 0)
    nodes = dict(geo.optical_system["nodes"])  # shallow copy of node dict
    nodes["pre_fts"] = {"element": [], "next": {"default": "wg_1"}}
    nodes["ell_3"] = {"element": [geo.ell_3, dm_new, geo.ell_5], "next": {"default": "wg_3"}}
    nodes["ell_4"] = {"element": [geo.ell_4, dm_new, geo.ell_6], "next": {"default": "wg_3"}}
    return {
        "entry": "pre_fts",
        "allowed_branch_sequences": geo.optical_system["allowed_branch_sequences"],
        "nodes": nodes,
    }


def _power_at_pos(pos_rays, detector, freqs):
    source_map, _, _ = detector.intensity_map(pos_rays, freqs, plot=False)
    return float(np.sum(source_map))


def _power_and_map_at_pos(pos_rays, detector, freqs):
    source_map, _, _ = detector.intensity_map(pos_rays, freqs, plot=False)
    return float(np.sum(source_map)), source_map


def _trace_at_pos(pos, prefts, separate_by_path):
    return rt.trace_optical_system(
        get_optical_system_with_dm_pos(pos),
        starting_rays=prefts,
        plot=False,
        return_all=separate_by_path,
    )


def scan_fts(starting_rays, fts_throw, fts_step, separate_by_path=False, return_path_ids=False, debug=False, n_workers=None):
    dm_positions = np.arange(-fts_throw, fts_throw + fts_step, fts_step)
    # trace rays through pre-FTS once. Then iterate with those as the starting rays
    t0 = 0.0
    if debug:
        t0 = time.time()
    prefts = rt.trace_optical_system(geo.optical_system["nodes"]["pre_fts"]["element"], starting_rays=starting_rays, plot=False)
    if debug:
        print("Finished tracing through pre-FTS optics.")
        print(f"Time taken: {time.time() - t0:.2f} seconds.")

    args = [(pos, prefts, separate_by_path) for pos in dm_positions]
    with Pool(processes=n_workers) as pool:
        outputs = list(tqdm(pool.starmap(_trace_at_pos, args), total=len(dm_positions), desc="Scanning FTS"))

    if not separate_by_path:
        return _pad_rays_to_rectangular(outputs)

    rays_by_path_by_pos = [out.get("rays_by_path", {}) for out in outputs]
    path_ids = list(dict.fromkeys(pid for rbp in rays_by_path_by_pos for pid in rbp))
    if len(path_ids) == 0:
        empty = np.empty((0, len(dm_positions), 0, 5), dtype=object)
        return (empty, np.array([], dtype=object)) if return_path_ids else empty

    stacked = np.stack(
        [_pad_rays_to_rectangular([rbp.get(pid, []) for rbp in rays_by_path_by_pos]) for pid in path_ids],
        axis=0,
    )
    return (stacked, np.array(path_ids, dtype=object)) if return_path_ids else stacked

def generate_interferogram(final_rays, detector, freqs, fts_throw, fts_step, n_workers=None, return_maps=False):
    dm_positions = np.arange(-fts_throw, fts_throw + fts_step, fts_step)
    args = [(pos, detector, freqs) for pos in final_rays]
    with Pool(processes=n_workers) as pool:
        if return_maps:
            results = list(tqdm(pool.starmap(_power_and_map_at_pos, args), total=len(dm_positions), desc="Generating interferogram"))
            power_values, source_maps = zip(*results)
            return np.array(power_values, dtype=float), dm_positions, np.array(source_maps, dtype=float)
        power_values = list(tqdm(pool.starmap(_power_at_pos, args), total=len(dm_positions), desc="Generating interferogram"))
    return np.array(power_values, dtype=float), dm_positions

'''Utilities for generating spectra from interferograms.'''

def generate_spectrum(interferogram, fts_step_size):
    # Fourier transform interferogram
    windowed_interferogram = np.hanning(int(np.shape(interferogram)[
        0])) * interferogram
    S = np.fft.rfft(windowed_interferogram) # real-valued FFT
    fft = np.abs(S)
    fft_freqs = np.fft.rfftfreq(len(interferogram), d=(4 * fts_step_size)/c)
    return fft_freqs, fft