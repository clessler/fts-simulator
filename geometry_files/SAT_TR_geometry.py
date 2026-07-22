import ray_tracing as rt
import numpy as np
import diffraction as diff

'''Optics tube'''
# focal plane
fp = rt.Object(origin=(-199.369, 206.892, -3069.356), diam=180.97*2, gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444))

# LPE 3
LPE_3 = rt.FlatLens(diam=185.15*2, thickness=4.5, origin = (-199.658, 205.572, -3054.417), gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444), n_lens=1.50145)

# L3
# cm to mm for a2: *e-1, for a4: *e-3, for a6: *e-5, for a8: *e-7
L3 = rt.DoubleAsphericLens(c_1 = -1/399.78, k_1 = -18.19, a2_1 = 2.29e-4, a4_1 = 5.008e-9, a6_1 = 1.945e-14, a8_1 = 0, c_2 = -1/402.35, k_2 = -30.006, a2_2 = 4.059e-5, a4_2 = 5.001e-9, a6_2 = 5.007e-14, a8_2 = 0, sign_1 = -1, sign_2 = -1, diam = 220*2, thickness = 29.66, origin=(-200.11, 203.372, -3028.461), gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444), n_lens=3.36)

# L2 (diff. coord frame; inverted z)
L2 = rt.DoubleAsphericLens(c_1 = 1/818.92, k_1 = -30, a2_1 = 1.565e-4, a4_1 = 3.063e-9, a6_1 = -5.002e-14, a8_1 = 0, c_2 = 1/417.33, k_2 = -1.046, a2_2 = -8.382e-4, a4_2 = -1.39e-9, a6_2 = 1.847e-15, a8_2 = 0, sign_1 = 1, sign_2 = 1, diam = 224*2, thickness = 46.97, origin=(-203.619, 187.127, -2842.612), gx_local=(0.97743859, -0.2103797, 0.01881997), gy_local=(-0.21122056, -0.97355259, 0.08707023), gz_local=(0, -0.0890804, -0.99602444), n_lens=3.36)

# LPE 2
LPE_2 = rt.FlatLens(diam=240*2, thickness=4.5, origin = (-204.414, 183.572, -2802.778), gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444), n_lens=1.50145)

# LPE 1
LPE_1 = rt.FlatLens(diam=240*2, thickness=4.5, origin = (-204.593, 182.745, -2793.316), gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444), n_lens=1.50145)

# L1 (diff coord frame; inverted z)
L1 = rt.DoubleAsphericLens(c_1 = 1/1035.55, k_1 = -30.001, a2_1 = 1.45e-4, a4_1 = 2.849e-9, a6_1 = -5.022e-14, a8_1 = 0, c_2 = -1/1672.58, k_2 = 29.998, a2_2 = 6.861e-4, a4_2 = 4.447e-10, a6_2 = -2.622e-14, a8_2 = 0, sign_1 = 1, sign_2 = 1, diam = 224*2, thickness = 43.5, origin=(-214.460, 136.950, -2268.575), gx_local=(0.97743859, -0.2103797, 0.01881997), gy_local=(-0.21122056, -0.97355259, 0.08707023), gz_local=(0, -0.0890804, -0.99602444), n_lens=3.36)

# Stop
stop = rt.Aperture(origin = (-214.716, 135.875, -2257.007), diam = 210*2, gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444))

# filters filters filters
# PTC2_AF
PTC2_AF = rt.FlatLens(diam=275*2, thickness=3, origin = (-215.229, 133.381, -2227.717), gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444), n_lens=3.14)

# HWP
HWP = rt.FlatLens(diam=250*2, thickness=10.86, origin = (-215.485, 132.197, -2214.171), gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444), n_lens=3.05)

# PTC1_AF
PTC1_AF = rt.FlatLens(diam=275*2, thickness=3, origin = (-217.221, 124.161, -2122.238), gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444), n_lens=3.14)

# skip IR filters (don't really affect the ray trace)

# window (double asphere)
window = rt.DoubleAsphericLens(c_1 = -1/1050, k_1 = 0, a2_1 = 0, a4_1 = 0, a6_1 = 0, a8_1 = 0, c_2 = -1/1050, k_2 = 0, a2_2 = 0, a4_2 = 0, a6_2 = 0, a8_2 = 0, sign_1 = -1, sign_2 = -1, diam = 350*2, thickness = 12, origin=(-218.720, 117.213, -2042.743), gx_local=(-0.97743859, -0.2103797, -0.01881997), gy_local=(0.21122056, -0.97355259, -0.08707023), gz_local=(0, -0.0890804, 0.99602444), n_lens=1.537)

SAT = [fp, LPE_3, L3, L2, LPE_2, LPE_1, L1, stop, PTC2_AF, HWP, PTC1_AF, window]

'''Coupling optics'''
# M1
M1 = rt.AsphericMirror(c = -1/1972.214, k = -1.17, a_2 = -8.581e-8, a_4 = 1.088e-14, diam = 1950*2, origin=(-590.487, -1603.001, 123.493), gx_local = (0.97742953, -0.2111199, 0.00774), gy_local = (-0.21126069, -0.97677318, 0.03582012), gz_local = (0, -0.03664993, -0.99932817), sign=-1)

# M2
M2 = rt.AsphericMirror(c = -1/1706.346, k = -6.046, a_2 = 1.623e-5, a_4 = -1.406e-12, diam = 1140*2, origin=(-558.230, -1453.723, -324.235), gx_local = (0.97743804, -0.20741958, -0.03989992), gy_local = (-0.2112297, -0.95983862, -0.18463974), gz_local = (0, 0.18889932, -0.98199646), sign=-1)

# beamsplitter
bs = rt.FlatMirror(diam=230*2, origin=(-447.394, -940.683, 1.487), gx_local=(-0.97743733, 0.12081967, 0.17325953), gy_local=(0.21122133, 0.55908352, 0.80175505), gz_local=(0, 0.8202604, -0.57199028))

coupling_optics = [M1, M2, bs]

'''FTS'''
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

# dm = rt.DihedralMirror(m=1, b=0.5, diam=124, origin= (0,0,0), gx_local = (1,0,0), gy_local = (0,0,-1), gz_local = (0,1,0))
central_mirror = rt.FlatMirror(diam=124, origin=(0,0,0), gx_local=(1,0,0), gy_local=(0,0,-1), gz_local=(0,1,0))

# considered modelling the FTS as a non-sequential system; but this will most likely be slower
# FTS_system = rt.NSSystem([wg_1, ell_1, ell_2])

'''End of FTS optics'''

'''Iris/focal plane'''
# origin in local coords: -238.577, 0, 368.267
iris = rt.FocalPlane(origin=(310.981, 309.566, 0), diam=76.2, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0))

'''Evaluation plane for including diffraction'''
eval_plane = rt.FocalPlane(origin = (251.836, 35.884, 0.0), diam = 100, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0))

iris_for_testing = rt.Aperture(origin=(310.981, 309.566, 0), diam=76.2, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0))
source_for_testing = rt.FocalPlane(origin=(359.266, 533.008, 0), diam=127, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0))

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
            "element": SAT + coupling_optics,
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

# add in source detector object
source = diff.Detector(origin=(359.266, 533.008, 0), diam=127, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0), resolution = 1)

# iris as a detector object
iris_detector = diff.Detector(origin=(310.981, 309.566, 0), diam=76.2, gx_local=(0.97743625, 0, -0.21123063), gy_local=(-0.21123063, 0, -0.97743625), gz_local=(0, 1, 0), resolution=1)