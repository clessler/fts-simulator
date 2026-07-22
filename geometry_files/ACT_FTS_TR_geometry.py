import ray_tracing as rt
import diffraction as diff
import numpy as np

# focal plane (receiver)
fp = rt.Object(origin = (-816.917, -515.171, 0), diam = 25, gx_local = (0.1927092, 0, 0.98125591), gy_local = (0.98125591, 0, -0.1927092), gz_local = (0, 1, 0))

# assume surfaces are in time-forward order (surf1 first, surf2 second), matching k values
# Source: fts_coupling_optics_geo.py, which is in INCHES (not cm).
#   lens3_surf1: c=-0.2260 in^-1, k=-0.9603, a4=+0.0005 in^-3, diam=5.112 in
#   lens3_surf2: c=-0.0913 in^-1, k=-6.539,  a4=-0.0004 in^-3, diam=5.089 in

# Convert in^-1 -> mm^-1: divide by 25.4.  Convert in^-3 -> mm^-3: divide by 25.4^3.

L3 = rt.DoubleAsphericLens(c_1 = 0.2260/25.4, k_1 = -0.9603, a2_1 = 0, a4_1 = 0.0005/25.4**3, a6_1 = 0, a8_1 = 0, c_2 = -0.0913/25.4, k_2 = -6.539, a2_2 = 0, a4_2 = -0.0004/25.4**3, a6_2 = 0, a8_2 = 0, sign_1 = 1, sign_2 = 1, diam = 129.255, thickness = 29.646, origin=(-552.029, -567.191, 0), gx_local = (-0.1927092, 0, -0.98125591), gy_local = (-0.98125591, 0, 0.1927092), gz_local = (0, 1, 0), n_lens=1.517)

stop=L3 

# bs
bs = rt.FlatMirror(diam=178, origin=(-300.162, -620.262, -0.494), gx_local=(0.83011758, 0,  -0.55758838), gy_local=(0.55758838, 0, 0.83011758), gz_local=(0,-1,0))

# L2
L2 = rt.DoubleAsphericLens(c_1 = -0.3571/25.4, k_1 = 0, a2_1 = 0, a4_1 = 0, a6_1 = 0, a8_1 = 0, c_2 = -0.4878/25.4, k_2 = 0, a2_2 = 0, a4_2 = 0, a6_2 = 0, a8_2 = 0, sign_1 = 1, sign_2 = 1, diam = 59.167, thickness = 53.34, origin=(-278.441, -494.754, 0), gx_local = (0.98125591, 0, -0.1927092), gy_local = (-0.1927092, 0, -0.98125591), gz_local = (0, 1, 0), n_lens=1.517)

# L1
L1 = rt.DoubleAsphericLens(c_1 = 0.1176/25.4, k_1 = 0, a2_1 = 0, a4_1 = 0, a6_1 = 0, a8_1 = 0, c_2 = -0.2222/25.4, k_2 = -1.8567, a2_2 = 0, a4_2 = 0.0005/25.4**3, a6_2 = 0, a8_2 = 0, sign_1 = 1, sign_2 = 1, diam = 109.183, thickness = 23.528, origin=(-236.582, -281.617, 0), gx_local = (0.98125591, 0, -0.1927092), gy_local = (-0.1927092, 0, -0.98125591), gz_local = (0, 1, 0), n_lens=1.517)

coupling_optics = [fp, L3, bs, L2, L1]

# FTS (ACT version)
def ab_to_ck(a, b):
    c = a/(b**2)
    k = (b**2 - a**2)/a**2
    return c, k

c_ellipse, k_ellipse = ab_to_ck(452.323, 461.188)
mirror_diam = 84.6

ell_1 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = mirror_diam, origin=(-135, 226.162, 0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0), sign=1)

ell_2 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = mirror_diam, origin=(-135, -226.162, 0), gx_local = (-1,0,0), gy_local = (0,0,1), gz_local = (0,-1,0), sign=1)

ell_3 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = mirror_diam, origin=(-45, 226.162, 0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0), sign=1)

ell_4 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = mirror_diam, origin=(-45, -226.162, 0), gx_local = (-1,0,0), gy_local = (0,0,1), gz_local = (0,-1,0), sign=1)

ell_5 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = mirror_diam, origin=(45, 226.162, 0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0), sign=1)

ell_6 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = mirror_diam, origin=(45, -226.162, 0), gx_local = (-1,0,0), gy_local = (0,0,1), gz_local = (0,-1,0), sign=1)

ell_7 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = mirror_diam, origin=(135, 226.162, 0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0), sign=1)

ell_8 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = mirror_diam, origin=(135, -226.162, 0), gx_local = (-1,0,0), gy_local = (0,0,1), gz_local = (0,-1,0), sign=1)

wg_1 = rt.WireGrid(pol_axis = np.pi/4, origin = (-180, 0,0), diam = 80, gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

wg_2 = rt.WireGrid(pol_axis = 0, origin = (-90, 0,0), diam = 80, gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

wg_3 = rt.WireGrid(pol_axis = np.pi/2, origin = (90, 0,0), diam = 80, gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

wg_4 = rt.WireGrid(pol_axis = np.pi/4, origin = (180, 0,0), diam = 80, gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

central_mirror = rt.DihedralMirror(m=1, b=0.5, diam=mirror_diam, origin= (0,0,0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

# iris
# Where to put this?
iris = rt.FocalPlane(origin=(247.445, 339.020, 0), diam=76.2, gx_local=(0, -0.9807794, 0.19511988), gy_local=(0, 0.19511988, 0.9807794), gz_local=(-1, 0, 0))

# source
source = diff.Detector(origin=(294.526, 575.683, 0), diam=120, gx_local=(0, -0.9807794, 0.19511988), gy_local=(0, 0.19511988, 0.9807794), gz_local=(-1, 0, 0), resolution = 1)

# full optical system with branching in FTS

allowed_branch_sequences = [
    ("wg_1:T", "ell_1", "wg_2:T", "ell_4", "wg_3:R", "ell_8", "wg_4:T", "eval_plane"),
    ("wg_1:T", "ell_1", "wg_2:R", "ell_3", "wg_3:T", "ell_8", "wg_4:T", "eval_plane"),
    ("wg_1:R", "ell_2", "wg_2:T", "ell_3", "wg_3:R", "ell_7", "wg_4:R", "eval_plane"),
    ("wg_1:R", "ell_2", "wg_2:R", "ell_4", "wg_3:T", "ell_7", "wg_4:R", "eval_plane"),
]

# main optical system for analysis
optical_system = {
    "entry": "pre_fts",
    "allowed_branch_sequences": allowed_branch_sequences,
    "nodes": {
        # Optics tube + coupling optics only
        "pre_fts": {
            "element": coupling_optics,
            "next": {"default": "wg_1"},
        },

        # Wire grid optical node
        "wg_1": {
            "element": wg_1,
            "next": {}, # routing comes from allowed_branch_sequences
        },

        "ell_1": {
            "element": [ell_1],
            "next": {"default": "wg_2"},
        },

        "ell_2": {
            "element": [ell_2],
            "next": {"default": "wg_2"},
        },

        'wg_2': {
            "element": wg_2,
            "next": {},
        },

        "ell_3": {"element": [ell_3, central_mirror, ell_5], "next": {"default": "wg_3"}}, 
        "ell_4": {"element": [ell_4, central_mirror, ell_6], "next": {"default": "wg_3"}}, 
        "wg_3": {"element": wg_3, "next": {}},
        "ell_7": {"element": [ell_7], "next": {"default": "wg_4"}},
        "ell_8": {"element": [ell_8], "next": {"default": "wg_4"}},
        "wg_4": {"element": wg_4, "next": {}},
        "eval_plane": {"element": [iris], "next": {"default": None}},
    },
}