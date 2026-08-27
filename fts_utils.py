# Code to simulate the scanning of the FTS.
# Geometry module is selected via the FTS_GEO_MODULE env var (set before importing this module).
# Defaults to geometry_files.so_coupling_optics_TR_geometry if the env var is not set.

import os as _os
import importlib as _importlib
import ray_tracing as rt
import numpy as np
import copy
_default_geo = _importlib.import_module(_os.environ.get('FTS_GEO_MODULE', 'geometry_files.so_coupling_optics_TR_geometry'))
import diffraction as diff
from tqdm import tqdm
import sys
import time
from multiprocessing import Pool
# from geometry_files.so_coupling_optics_TR_geometry import optical_system, dm, ell_3, ell_4, ell_5, ell_6

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


def _complex_field_at_pos(pos_rays, detector, freqs, ray_chunk_size=32):
    if not _has_valid_rays(pos_rays):
        return None, None
    Ex, Ey, _, _ = detector.intensity_map(pos_rays, freqs, weights=None, plot=False, ray_chunk_size=ray_chunk_size, return_field=True)
    return Ex, Ey


def _trace_at_pos(pos, prefts, separate_by_path, return_full_history=False):
    return rt.trace_optical_system(
        get_optical_system_with_dm_pos(pos),
        starting_rays=prefts,
        plot=False,
        return_all=separate_by_path,
        return_full_history=return_full_history,
    )


def _trace_at_pos_star(args): return _trace_at_pos(*args)
def _complex_field_at_pos_star(args): return _complex_field_at_pos(*args)


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

def generate_complex_field_maps(final_rays, detector, freqs, fts_throw, fts_step, n_workers=None, ray_chunk_size=32):
    '''Parallelized Huygens-Fresnel stage: ray-traced rays -> complex (Ex, Ey) field at
    `detector`'s plane, for each DM position. This is the only stage worth running under
    Pool -- downstream propagation through any further detectors (see _propagate_field_chain)
    is cheap FFT work and is done serially.'''
    dm_positions = np.arange(-fts_throw, fts_throw + fts_step, fts_step)
    args = [(pos, detector, freqs, ray_chunk_size) for pos in final_rays]
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(_complex_field_at_pos_star, args), total=len(dm_positions), desc="Generating complex field maps", file=sys.stderr, dynamic_ncols=True))
    x_local, y_local = detector.grid_axes()
    field_shape = (len(freqs), len(y_local), len(x_local))
    Ex = np.stack([e if e is not None else np.zeros(field_shape, dtype=complex) for e, _ in results])
    Ey = np.stack([e if e is not None else np.zeros(field_shape, dtype=complex) for _, e in results])
    return Ex, Ey, dm_positions, x_local, y_local

def _propagate_field_chain(Ex, Ey, detector_chain, freqs, weights=None, pad_factor=2, max_batch_bytes=2e9):
    '''propagate a (N_pos, N_freq, Ny, Nx) complex field -- already at
    detector_chain[0]'s plane -- through the rest of detector_chain via batched
    angular-spectrum hops.'''
    w = np.asarray(weights) if weights is not None else np.ones(len(freqs))

    def _weighted_map(Ex, Ey, det):
        m = np.sum(w[None, :, None, None] * (np.abs(Ex)**2 + np.abs(Ey)**2), axis=1)
        m[..., det.aperture_mask()] = np.nan
        return m

    n_pos, n_freq = Ex.shape[:2]
    n_hops = len(detector_chain) - 1
    stage_maps = [_weighted_map(Ex, Ey, detector_chain[0])]
    for hop_i, (prev, cur) in enumerate(zip(detector_chain[:-1], detector_chain[1:])):
        x_out, y_out = cur.grid_axes()
        Ny_pad = int(np.ceil(max(Ex.shape[-2], len(y_out)) * pad_factor))
        Nx_pad = int(np.ceil(max(Ex.shape[-1], len(x_out)) * pad_factor))
        bytes_per_pos = n_freq * Ny_pad * Nx_pad * 16 * 2  # complex128, Ex + Ey
        chunk_size = max(1, int(max_batch_bytes // bytes_per_pos))
        n_chunks = -(-n_pos // chunk_size)  # ceil division

        Ex_chunks, Ey_chunks = [], []
        starts = tqdm(range(0, n_pos, chunk_size), total=n_chunks,
                       desc=f"Propagating through iris {hop_i + 1}/{n_hops}",
                       file=sys.stderr, dynamic_ncols=True)
        for start in starts:
            sl = slice(start, start + chunk_size)
            Ex_c, Ey_c, _, _ = cur.intensity_map_from_field(Ex[sl], Ey[sl], prev, freqs, weights=None, pad_factor=pad_factor, return_field=True)
            Ex_chunks.append(Ex_c)
            Ey_chunks.append(Ey_c)
        Ex, Ey = np.concatenate(Ex_chunks), np.concatenate(Ey_chunks)

        if cur is not detector_chain[-1]:  # intermediate physical stop: clip before propagating further
            mask = cur.aperture_mask()
            Ex[..., mask] = 0.0
            Ey[..., mask] = 0.0
        stage_maps.append(_weighted_map(Ex, Ey, cur))
    return stage_maps

def generate_interferogram(final_rays, detector, freqs, fts_throw, fts_step, n_workers=None, return_maps=False, return_all_stage_maps=False, ray_chunk_size=32, weights=None, pad_factor=2, max_batch_bytes=2e9):
    chain = tuple(detector) if isinstance(detector, (list, tuple)) else (detector,)
    Ex, Ey, dm_positions, x_local, y_local = generate_complex_field_maps(final_rays, chain[0], freqs, fts_throw, fts_step, n_workers=n_workers, ray_chunk_size=ray_chunk_size)
    stage_maps = _propagate_field_chain(Ex, Ey, chain, freqs, weights=weights, pad_factor=pad_factor, max_batch_bytes=max_batch_bytes)
    power_values = np.nansum(stage_maps[-1], axis=(-2, -1))
    if return_all_stage_maps:
        return power_values, dm_positions, stage_maps       # list, one array per detector_chain entry
    if return_maps:
        return power_values, dm_positions, stage_maps[-1]   # final detector only, same shape as today
    return power_values, dm_positions

'''Utilities for generating spectra from interferograms.'''

# add in source-dependence through exponent alpha
def generate_spectrum(interferogram, fts_step_size, normalize=True, normalize_cutoff=None, return_cutoff = False, include_source=False, source_alpha=2):
    # Fourier transform interferogram
    windowed_interferogram = np.hanning(int(np.shape(interferogram)[
        0])) * interferogram
    S = np.fft.rfft(windowed_interferogram) # real-valued FFT
    fft = np.abs(S)
    fft_freqs = np.fft.rfftfreq(len(interferogram), d=(4 * fts_step_size)/c)

    mask = (fft_freqs >= normalize_cutoff) if normalize_cutoff is not None else np.ones(len(fft_freqs), dtype=bool)

    if include_source:
        # apply source-dependence through power law with exponent alpha
        # keep the highest value in the spectrum the same as before
        norm_freq = fft_freqs[mask][np.argmax(fft[mask])] if np.any(mask) else fft_freqs[np.argmax(fft)]
        fft *= fft_freqs**source_alpha/(norm_freq**source_alpha)

    if normalize:
        norm_val = np.max(fft[mask]) if np.any(mask) else np.max(fft)
        fft /= norm_val
    if return_cutoff and normalize_cutoff is not None:
        return fft_freqs[mask], fft[mask]
    return fft_freqs, fft