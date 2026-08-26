import ray_tracing as rt
import diffraction as diff
import numpy as np

'''Optics tube'''
# focal plane
fp = rt.Object(origin = (999.274, -1631.118, -18.564), diam = 140*2, gx_local = (0.76049382, 0.02756666, 0.64875977), gy_local = (0.64832329,  0.02350535, -0.76100224), gz_local = (-0.03621495,  0.99934402,  0))

# Lens 3
L3 = rt.DoubleAsphericLens(c_1 = -1/604.446, k_1 = -3.177, a2_1 = -2.367e-5, a4_1 = 1e-10, a6_1 = 6.566e-15, a8_1 = 0, c_2 = -1/6608.297, k_2 = 414.143, a2_2 = 2.982e-6, a4_2 = -3.995e-10, a6_2 = -3.181e-16, a8_2 = 0, sign_1 = -1, sign_2 = -1, diam = 179*2, thickness = 31.552, origin=(858.569, -1466.026, -18.525), gx_local=(0.76049382, 0.02756666, 0.64875977), gy_local=(0.64832329,  0.02350535, -0.76100224), gz_local=(-0.03621495,  0.99934402,  0), n_lens=3.38)

# aperture stop
stop = rt.Aperture(origin = (702.866, -1283.388, -18.528), diam = 68*2, gx_local = (0.76049382, 0.02756666, 0.64875977), gy_local = (0.64832329,  0.02350535, -0.76100224), gz_local = (-0.03621495,  0.99934402,  0))

# Lens 2
L2 = rt.AsphericLens(c = 1/871.246, k = -10.201, a_2 = 1.148e-4, a_4 = -1.104e-10, a_6 = 2.97e-14, a_8 = -5.98e-19, diam = 168*2, thickness = 25.825, origin=(626.147, -1193.396, -18.525), gx_local=(-0.76049382,  0.02756666, -0.64875977), gy_local=(-0.64832329,  0.02350535,  0.76100224), gz_local=(0.03621495, 0.99934402, 0), n_lens=3.38)

# Lens 1
L1 = rt.AsphericLens(c = 1/1510.613, k = 12.097, a_2 = 5.877e-5, a_4 = -1.769e-9, a_6 = 1.31e-15, a_8 = -1.995e-19, diam = 189*2, thickness = 18.68, origin=(374.498, -898.213, -18.525), gx_local=(-0.76049382,  0.02756666, -0.64875977), gy_local=(-0.64832329,  0.02350535,  0.76100224), gz_local=(0.03621495, 0.99934402, 0), n_lens=3.38)

# 4K and 40K filters
filter_4K = rt.FlatLens(diam=188*2, thickness=2.1, origin = (349.728, -869.158, -18.528), gx_local = (0.76049382, 0.02756666, 0.64875977), gy_local = (0.64832329,  0.02350535, -0.76100224), gz_local = (-0.03621495,  0.99934402,  0), n_lens=1.56)

filter_40K = rt.FlatLens(diam=179.5*2, thickness=10, origin = (322.091, -836.74, -18.528), gx_local = (0.76049382, 0.02756666, 0.64875977), gy_local = (0.64832329,  0.02350535, -0.76100224), gz_local = (-0.03621495,  0.99934402,  0), n_lens=1.56)

# alumina
alumina = rt.FlatLens(diam=190*2, thickness=7, origin = (291.404, -800.745, -18.528), gx_local = (0.76049382, 0.02756666, 0.64875977), gy_local = (0.64832329,  0.02350535, -0.76100224), gz_local = (-0.03621495,  0.99934402,  0), n_lens=3.14)

# 80K and 300K filters
filter_80K = rt.FlatLens(diam=190*2, thickness=1, origin = (280.376, -787.808, -18.528), gx_local = (0.76049382, 0.02756666, 0.64875977), gy_local = (0.64832329,  0.02350535, -0.76100224), gz_local = (-0.03621495,  0.99934402,  0), n_lens=1.56)

filter_300K = rt.FlatLens(diam=200*2, thickness=10, origin = (241.429, -742.166, -18.46), gx_local = (0.76049382, 0.02756666, 0.64875977), gy_local = (0.64832329,  0.02350535, -0.76100224), gz_local = (-0.03621495,  0.99934402,  0), n_lens=1.56)

OT = [fp, L3, stop, L2, L1, filter_4K, filter_40K, alumina, filter_80K, filter_300K]

'''End of optics tube; start of coupling optics'''

# M1
M1 = rt.AsphericMirror(c = -1/435.938, k = -0.738, a_2 = 4.287e-4, a_4 = 1.533e-10, diam = 850*2, origin=(-480.504, -898.4073, -565.7307), gx_local = (0.46653284, -0.58807357,  0.66069402), gy_local = (0.88386309,  0.33837118, -0.32294113), gz_local = (-0.03370144,  0.73463137,  0.67762893), sign=-1)

# M2
M2 = rt.AsphericMirror(c = 1/364.517, k = -1.142, a_2 = -6.617e-4, a_4 = -3.28e-10, diam = 800*2, origin=(311.7475, -573.4117, -167.995), gx_local = (-0.91269546, 0.06536967, -0.40337799), gy_local = (-0.40716047,  -0.06154007,  0.91128106), gz_local = (0.03474005,  0.99596131, 0.08279011), sign=1)

# beamsplitter
bs = rt.FlatMirror(diam=300*2, origin=(-328.964, -450.069, -10.925), gx_local=(-0.24351913, -0.01813357,  0.96972656), gy_local=(0.96976326, 0.01202771, 0.24375102), gz_local=(-0.01608971,  0.99976324,  0.01464845))

'''End of coupling optics; start of FTS optics'''
# useful function to switch from semi-major and minor axes to c and k
def ab_to_ck(a, b):
    c = a/(b**2)
    k = (b**2 - a**2)/a**2
    return c, k

c_ellipse, k_ellipse = ab_to_ck(564.751, 577.787)

ell_1 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = 122.042, origin=(-183.063, 282.375, 0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0), sign=1)

# note this might be wrong!! Might want to use different basis vectors but keep the sign...
ell_2 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = 122.042, origin=(-183.063, -282.375, 0), gx_local = (-1,0,0), gy_local = (0,0,1), gz_local = (0,-1,0), sign=1)

ell_3 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = 122.042, origin=(-61.021, 282.375, 0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0), sign=1)

ell_4 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = 122.042, origin=(-61.021, -282.375, 0), gx_local = (-1,0,0), gy_local = (0,0,1), gz_local = (0,-1,0), sign=1)

ell_5 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = 122.042, origin=(61.021, 282.375, 0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0), sign=1)

ell_6 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = 122.042, origin=(61.021, -282.375, 0), gx_local = (-1,0,0), gy_local = (0,0,1), gz_local = (0,-1,0), sign=1)

ell_7 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = 122.042, origin=(183.063, 282.375, 0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0), sign=1)

ell_8 = rt.AsphericMirror(c = c_ellipse, k = k_ellipse, a_2 = 0, a_4 = 0, diam = 122.042, origin=(183.063, -282.375, 0), gx_local = (-1,0,0), gy_local = (0,0,1), gz_local = (0,-1,0), sign=1)

wg_1 = rt.WireGrid(pol_axis = np.pi/4, origin = (-246.11, 0,0), diam = 115.57, gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

wg_2 = rt.WireGrid(pol_axis = 0, origin = (-124.11, 0,0), diam = 115.57, gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

wg_3 = rt.WireGrid(pol_axis = np.pi/2, origin = (124.11, 0,0), diam = 115.57, gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

wg_4 = rt.WireGrid(pol_axis = np.pi/4, origin = (246.11, 0,0), diam = 115.57, gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

# old geometry
# dm = rt.DihedralMirror(m=1, b=0.5, diam=124, origin= (0,0,0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))

# flat mirror (SO FTS turns out to have a flat mirror rather than a dihedral one)
# NOTE: might want to give this a thickness, but this is unlikely to matter
central_mirror = rt.FlatMirror(diam=124, origin=(0,0,0), gx_local=(1,0,0), gy_local=(0,0,-1), gz_local=(0,1,0))

# considered modelling the FTS as a non-sequential system; but this will most likely be slower
# FTS_system = rt.NSSystem([wg_1, ell_1, ell_2])

'''End of FTS optics'''

'''Iris/focal plane'''
# origin in local coords: -238.577, 0, 368.267
iris = rt.FocalPlane(origin=(310.981, 309.566, 0), diam=76.2, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0))

'''Evaluation plane for Kirchoff-propagating to the source'''
eval_plane = rt.FocalPlane(origin = (251.836, 35.884, 0.0), diam = 100, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0))

'''Full system with allowed paths through FTS'''

# Optional branch-routing table for graph mode.
# Format per tuple: ("node:port", "next_node", "node:port", "next_node", ...)
# Example below means:
#   wg_1:T -> ell_1, then wg_2:T -> ell_4

# Note: commented out branches that go to zero due to anti-aligned wire grids
allowed_branch_sequences = [
    # ("wg_1:T", "ell_1", "wg_2:T", "ell_4", "wg_3:T", "ell_7", "wg_4:R", "iris"),
    ("wg_1:T", "ell_1", "wg_2:T", "ell_4", "wg_3:R", "ell_8", "wg_4:T", "eval_plane"),
    ("wg_1:T", "ell_1", "wg_2:R", "ell_3", "wg_3:T", "ell_8", "wg_4:T", "eval_plane"),
    # ("wg_1:T", "ell_1", "wg_2:R", "ell_3", "wg_3:R", "ell_7", "wg_4:R", "iris"),
    # ("wg_1:R", "ell_2", "wg_2:T", "ell_3", "wg_3:T", "ell_8", "wg_4:T", "iris"),
    ("wg_1:R", "ell_2", "wg_2:T", "ell_3", "wg_3:R", "ell_7", "wg_4:R", "eval_plane"),
    ("wg_1:R", "ell_2", "wg_2:R", "ell_4", "wg_3:T", "ell_7", "wg_4:R", "eval_plane"),
    # ("wg_1:R", "ell_2", "wg_2:R", "ell_4", "wg_3:R", "ell_8", "wg_4:T", "iris"),
]

# main optical system for analysis
optical_system = {
    "entry": "pre_fts",
    "allowed_branch_sequences": allowed_branch_sequences,
    "nodes": {
        # Optics tube + coupling optics only
        "pre_fts": {
            "element": OT+[M1, M2, bs],
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
        "eval_plane": {"element": [eval_plane], "next": {"default": None}},
    },
}

# source detector object
source = diff.Detector(origin=(359.266, 533.008, 0), diam=120, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0), resolution = 1)

# iris as a detector object (for beam-mapping at the iris plane)
iris_detector = diff.Detector(origin=(310.981, 309.566, 0), diam=76.2, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0), resolution=1)