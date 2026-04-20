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

def sum_rays_at_fp(final_rays, freqs, fts_throw, fts_step):
    dm_positions = np.arange(-fts_throw, fts_throw + fts_step, fts_step)
    final_arr = np.asarray(final_rays, dtype=object)

    if final_arr.ndim == 4:
        path_interferograms = np.array(
            [sum_rays_at_fp(final_arr[p], freqs, fts_throw, fts_step)[0] for p in range(final_arr.shape[0])],
            dtype=float,
        )
        return path_interferograms, dm_positions

    interferogram = np.zeros(final_arr.shape[0], dtype=float)
    for i, pos_data in enumerate(final_arr):
        pos_arr = _as_ray_matrix(pos_data)
        if pos_arr.shape[0] == 0:
            continue

        theta = np.asarray(pos_arr[:, 0], dtype=float)
        intensity = np.asarray(pos_arr[:, 1], dtype=float)
        distance = np.asarray(pos_arr[:, 4], dtype=float)

        valid = np.isfinite(theta) & np.isfinite(intensity) & np.isfinite(distance)
        if not np.any(valid):
            continue

        theta = theta[valid]
        intensity = intensity[valid]
        distance = distance[valid]
        amp = np.sqrt(intensity)
        ex0 = amp * np.cos(theta)
        ey0 = amp * np.sin(theta)

        for freq in freqs:
            wavelength = c / freq
            phase = np.exp(1j * 2 * np.pi * distance / wavelength)
            ex = ex0 * phase
            ey = ey0 * phase
            interferogram[i] += np.square(np.abs(np.sum(ex))) + np.square(np.abs(np.sum(ey)))
    return interferogram, dm_positions

def sum_rays_at_fp_test(final_rays, freqs, fts_throw, fts_step):
    # Coherently sum different paths for each ray index, then add ray powers.
    # Input should be output from scan_fts(..., separate_by_path=True):
    #   shape (n_paths, n_dm_positions, n_rays_max, 5)
    dm_positions = np.arange(-fts_throw, fts_throw + fts_step, fts_step)
    arr = np.asarray(final_rays, dtype=object)

    if arr.ndim != 4:
        raise ValueError(
            "sum_rays_at_fp_test expects final_rays from scan_fts(..., separate_by_path=True) "
            "with shape (n_paths, n_positions, n_rays, 5)."
        )

    n_paths, n_positions, n_rays_max, _ = arr.shape
    if n_positions != dm_positions.size:
        raise ValueError(
            "Mismatch between final_rays mirror-position axis and requested (fts_throw, fts_step)."
        )

    interferogram = np.zeros(n_positions, dtype=float)

    for i in range(n_positions):
        # Sum one mirror position at a time.
        rays_at_pos = arr[:, i, :, :]  # (n_paths, n_rays_max, 5)

        for freq in freqs:
            wavelength = c / freq
            power_this_freq = 0.0

            # Each ray index represents one launched ray; combine its path contributions coherently.
            for r in range(n_rays_max):
                ray_entries = rays_at_pos[:, r, :]  # (n_paths, 5)

                theta = np.asarray(ray_entries[:, 0], dtype=float)
                intensity = np.asarray(ray_entries[:, 1], dtype=float)
                distance = np.asarray(ray_entries[:, 4], dtype=float)

                valid = np.isfinite(theta) & np.isfinite(intensity) & np.isfinite(distance)
                if not np.any(valid):
                    continue

                theta = theta[valid]
                intensity = intensity[valid]
                distance = distance[valid]

                amp = np.sqrt(intensity)
                ex0 = amp * np.cos(theta)
                ey0 = amp * np.sin(theta)
                phase = np.exp(1j * 2 * np.pi * distance / wavelength)

                ex_sum = np.sum(ex0 * phase)
                ey_sum = np.sum(ey0 * phase)

                # Incoherently add each launched ray's power contribution.
                power_this_freq += np.abs(ex_sum) ** 2 + np.abs(ey_sum) ** 2

            interferogram[i] += power_this_freq

    return interferogram, dm_positions

'''Utilities for generating spectra from interferograms.'''

def generate_spectrum(interferogram, fts_step_size):
    # Fourier transform interferogram
    windowed_interferogram = np.hanning(int(np.shape(interferogram)[
        0])) * interferogram
    S = np.fft.rfft(windowed_interferogram) # real-valued FFT
    fft = np.abs(S)
    fft_freqs = np.fft.rfftfreq(len(interferogram), d=(4 * fts_step_size)/c)
    return fft_freqs, fft