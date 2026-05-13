# Code to simulate the scanning of the FTS.
# Geometry module is selected via the FTS_GEO_MODULE env var (set before importing this module).
# Defaults to so_coupling_optics_TR_geometry if the env var is not set.

import os as _os
import importlib as _importlib
import ray_tracing as rt
import numpy as np
import copy
_default_geo = _importlib.import_module(_os.environ.get('FTS_GEO_MODULE', 'so_coupling_optics_TR_geometry'))
import diffraction as diff
from tqdm import tqdm
import sys
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

# Function to produce an optical system with just the FTS and with the central mirror at a specified position
def get_optical_system_with_dm_pos(pos):
    m_new = copy.copy(_default_geo.central_mirror)  # shallow copy; rotation matrices are unchanged since only origin moves
    m_new.origin = (0, pos, 0)
    nodes = dict(_default_geo.optical_system["nodes"])  # shallow copy of node dict
    nodes["pre_fts"] = {"element": [], "next": {"default": "wg_1"}}
    nodes["ell_3"] = {"element": [_default_geo.ell_3, m_new, _default_geo.ell_5], "next": {"default": "wg_3"}}
    nodes["ell_4"] = {"element": [_default_geo.ell_4, m_new, _default_geo.ell_6], "next": {"default": "wg_3"}}
    return {
        "entry": "pre_fts",
        "allowed_branch_sequences": _default_geo.optical_system["allowed_branch_sequences"],
        "nodes": nodes,
    }

# Plotting function to produce a full copy of optical system with the central mirror at a specified position
def get_full_optical_system_with_dm_pos(pos):
    m_new = copy.copy(_default_geo.central_mirror)  # shallow copy; rotation matrices are unchanged since only origin moves
    m_new.origin = (0, pos, 0)
    nodes = dict(_default_geo.optical_system["nodes"])  # shallow copy of node dict
    # nodes["pre_fts"] = {"element": [], "next": {"default": "wg_1"}}
    nodes["ell_3"] = {"element": [_default_geo.ell_3, m_new, _default_geo.ell_5], "next": {"default": "wg_3"}}
    nodes["ell_4"] = {"element": [_default_geo.ell_4, m_new, _default_geo.ell_6], "next": {"default": "wg_3"}}
    return {
        "entry": "pre_fts",
        "allowed_branch_sequences": _default_geo.optical_system["allowed_branch_sequences"],
        "nodes": nodes,
    }

def _has_valid_rays(pos_rays):
    if len(pos_rays) == 0:
        return False
    return not all(np.isnan(r[0]) for r in pos_rays)


def _power_at_pos(pos_rays, detector, freqs, ray_chunk_size=32, weights=None):
    if not _has_valid_rays(pos_rays):
        return 0.0
    source_map, _, _ = detector.intensity_map(pos_rays, freqs, weights=weights, plot=False, ray_chunk_size=ray_chunk_size)
    return float(np.nansum(source_map))


def _power_and_map_at_pos(pos_rays, detector, freqs, ray_chunk_size=32, weights=None):
    if not _has_valid_rays(pos_rays):
        return 0.0, None
    source_map, _, _ = detector.intensity_map(pos_rays, freqs, weights=weights, plot=False, ray_chunk_size=ray_chunk_size)
    return float(np.nansum(source_map)), source_map


def _trace_at_pos(pos, prefts, separate_by_path, return_full_history=False):
    return rt.trace_optical_system(
        get_optical_system_with_dm_pos(pos),
        starting_rays=prefts,
        plot=False,
        return_all=separate_by_path,
        return_full_history=return_full_history,
    )


def _trace_at_pos_star(args): return _trace_at_pos(*args)
def _power_at_pos_star(args): return _power_at_pos(*args)
def _power_and_map_at_pos_star(args): return _power_and_map_at_pos(*args)


def _prepend_prefts_histories(outputs, pos_to_prefts_hist):
    """Prepend pre-FTS path histories to each FTS terminal ray's history in-place.

    Each FTS terminal ray's history begins at the pre-FTS output position.
    We look up that position in pos_to_prefts_hist and prepend the full
    pre-FTS path, skipping the duplicate junction point.
    """
    for out in outputs:
        hists = out.get("ray_histories") if isinstance(out, dict) else None
        if not hists:
            continue
        for i, h in enumerate(hists):
            if h is None or len(h) == 0:
                continue
            key = tuple(h[0].tolist())
            pre = pos_to_prefts_hist.get(key)
            if pre is not None:
                hists[i] = list(pre) + h[1:]  # h[0] duplicates pre[-1]; drop it


def scan_fts(starting_rays, fts_throw, fts_step, separate_by_path=False, return_path_ids=False, return_full_history=False, debug=False, n_workers=None):
    dm_positions = np.arange(-fts_throw, fts_throw + fts_step, fts_step)
    # trace rays through pre-FTS once. Then iterate with those as the starting rays
    t0 = 0.0
    if debug:
        t0 = time.time()

    prefts_elements = _default_geo.optical_system["nodes"]["pre_fts"]["element"]
    if return_full_history:
        prefts_result = rt.trace_optical_system(prefts_elements, starting_rays=starting_rays, plot=False, return_full_history=True)
        prefts = prefts_result["final_rays"]
        pos_to_prefts_hist = {
            tuple(np.asarray(ray[2], dtype=float).tolist()): hist
            for ray, hist in zip(prefts, prefts_result["ray_histories"])
        }
    else:
        prefts = rt.trace_optical_system(prefts_elements, starting_rays=starting_rays, plot=False)
        pos_to_prefts_hist = None

    if debug:
        print("Finished tracing through pre-FTS optics.")
        print(f"Time taken: {time.time() - t0:.2f} seconds.")

    args = [(pos, prefts, separate_by_path, return_full_history) for pos in dm_positions]
    with Pool(processes=n_workers) as pool:
        outputs = list(tqdm(pool.imap(_trace_at_pos_star, args), total=len(dm_positions), desc="Scanning FTS", file=sys.stderr, dynamic_ncols=True))

    if return_full_history:
        _prepend_prefts_histories(outputs, pos_to_prefts_hist)

    if not separate_by_path:
        if return_full_history:
            histories_by_pos = [out["ray_histories"] for out in outputs]
            rays_by_pos = [out["final_rays"] for out in outputs]
            return _pad_rays_to_rectangular(rays_by_pos), histories_by_pos
        return _pad_rays_to_rectangular(outputs)

    rays_by_path_by_pos = [out.get("rays_by_path", {}) for out in outputs]
    path_ids = list(dict.fromkeys(pid for rbp in rays_by_path_by_pos for pid in rbp))
    histories_by_pos = [out["ray_histories"] for out in outputs] if return_full_history else None

    if len(path_ids) == 0:
        empty = np.empty((0, len(dm_positions), 0, 5), dtype=object)
        if return_full_history:
            return (empty, np.array([], dtype=object), histories_by_pos) if return_path_ids else (empty, histories_by_pos)
        return (empty, np.array([], dtype=object)) if return_path_ids else empty

    stacked = np.stack(
        [_pad_rays_to_rectangular([rbp.get(pid, []) for rbp in rays_by_path_by_pos]) for pid in path_ids],
        axis=0,
    )
    if return_full_history:
        return (stacked, np.array(path_ids, dtype=object), histories_by_pos) if return_path_ids else (stacked, histories_by_pos)
    return (stacked, np.array(path_ids, dtype=object)) if return_path_ids else stacked

def generate_interferogram(final_rays, detector, freqs, fts_throw, fts_step, n_workers=None, return_maps=False, ray_chunk_size=32, weights=None):
    dm_positions = np.arange(-fts_throw, fts_throw + fts_step, fts_step)
    args = [(pos, detector, freqs, ray_chunk_size, weights) for pos in final_rays]
    with Pool(processes=n_workers) as pool:
        if return_maps:
            results = list(tqdm(pool.imap(_power_and_map_at_pos_star, args), total=len(dm_positions), desc="Generating interferogram", file=sys.stderr, dynamic_ncols=True))
            power_values, source_maps = zip(*results)
            map_shape = next((m.shape for m in source_maps if m is not None), None)
            source_maps = [m if m is not None else np.zeros(map_shape) for m in source_maps]
            return np.array(power_values, dtype=float), dm_positions, np.array(source_maps, dtype=float)
        power_values = list(tqdm(pool.imap(_power_at_pos_star, args), total=len(dm_positions), desc="Generating interferogram", file=sys.stderr, dynamic_ncols=True))
    return np.array(power_values, dtype=float), dm_positions

'''Utilities for generating spectra from interferograms.'''

def generate_spectrum(interferogram, fts_step_size, normalize=True, normalize_cutoff=None, return_cutoff = False):
    # Fourier transform interferogram
    windowed_interferogram = np.hanning(int(np.shape(interferogram)[
        0])) * interferogram
    S = np.fft.rfft(windowed_interferogram) # real-valued FFT
    fft = np.abs(S)
    fft_freqs = np.fft.rfftfreq(len(interferogram), d=(4 * fts_step_size)/c)
    if normalize:
        if normalize_cutoff is not None:
            mask = fft_freqs >= normalize_cutoff
            norm_val = np.max(fft[mask]) if np.any(mask) else np.max(fft)
        else:
            norm_val = np.max(fft)
        fft /= norm_val
    if return_cutoff and normalize_cutoff is not None:
        return fft_freqs[mask], fft[mask] # type: ignore
    return fft_freqs, fft