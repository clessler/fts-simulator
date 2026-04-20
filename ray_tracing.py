from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial import distance
import plotly.graph_objects as go
import plotly.io as pio
import random
import yaml
from dataclasses import dataclass, field
from collections import deque
from typing import Literal, Optional, Sequence, cast
import numpy.typing as npt

# pio.renderers.default = 'browser'

# input rays are directly from FTS: format:
# [linear pol angle, intensity, launch position, unit vector, total distance travelled]

# note: need to include something to handle polarized intensity from beamsplitter

# stuff to add: non-sequential system class; note the skeleton is here but it still doesn't work

'''Functions to generate rays'''

def generate_rays_from_beam_data(
    Object,
    start_disp,
    beam_data,
    theta_bound,
    method = 'random',
    num_rays = None,
    n_theta = None,
    n_phi = None,
    pol_angle=0.123,
    direction_sign='auto',
    direction_reference=None,
):
    """
    Generate rays from a source-plane Object.

    Ray format:
      [linear pol angle, intensity, launch position (global), unit vector (global), total distance traveled]

    Inputs:
      - Object: ray_tracing Object representing the source plane
      - start_disp: local (x, y) or (x, y, z) launch point in source coordinates
      - method: 'random' or 'regular'
      - num_rays: number of random rays (required for method='random')
      - n_theta: number of discrete theta rings (required for method='regular')
      - n_phi: number of rays per theta ring (required for method='regular')
      - theta_bound: cone half-angle in degrees
      - beam_data: 2D array of beam intensities as a function of (theta, phi) in degrees
      - direction_sign: +1, -1, or 'auto'
      - direction_reference: optical element (or 3-vector point) used when direction_sign='auto'

    Notes:
      - Rays are sampled on an n_theta x n_phi angular grid.
      - One extra on-axis ray (theta=0) is always appended to guarantee a central ray.
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    source = Object
    beam_data = np.asarray(beam_data, dtype=float)

    method = str(method).strip().lower()

    if beam_data.ndim != 2:
        raise ValueError("beam_data must be a 2D array [theta, phi].")

    if method == 'regular':
        if n_theta is None or n_phi is None:
            raise ValueError("method='regular' requires n_theta and n_phi.")
        if n_theta < 1 or n_phi < 1:
            raise ValueError("n_theta and n_phi must both be >= 1.")
    elif method == 'random':
        if num_rays is None:
            raise ValueError("method='random' requires num_rays.")
        if num_rays < 1:
            raise ValueError("num_rays must be >= 1 for method='random'.")
    else:
        raise ValueError("method must be 'random' or 'regular'.")

    start_disp = np.asarray(start_disp, dtype=float).reshape(-1)
    if start_disp.size == 2:
        launch_local = np.array([start_disp[0], start_disp[1], 0.0], dtype=float)
    elif start_disp.size == 3:
        launch_local = start_disp.astype(float)
    else:
        raise ValueError("start_disp must have 2 elements (x, y) or 3 elements (x, y, z).")

    if np.hypot(launch_local[0], launch_local[1]) > (source.diam / 2):
        raise ValueError("start_disp is outside the source aperture (diam/2).")

    if direction_sign == 'auto':
        if direction_reference is not None:
            if hasattr(direction_reference, 'origin'):
                ref_global = np.asarray(direction_reference.origin, dtype=float)
            else:
                ref_global = np.asarray(direction_reference, dtype=float)
                if ref_global.shape != (3,):
                    raise ValueError("direction_reference must be an optical element with .origin or a 3-vector point.")
            ref_local = source._to_local_point(ref_global)
            dz = float(ref_local[2] - launch_local[2])
            direction_sign = +1 if dz >= 0.0 else -1
        else:
            # Fallback if no reference optic is supplied.
            direction_sign = +1
    elif direction_sign not in (+1, -1):
        raise ValueError("direction_sign must be +1, -1, or 'auto'.")

    theta_bound_rad = np.deg2rad(float(theta_bound))
    if not (0.0 <= theta_bound_rad <= np.pi):
        raise ValueError("theta_bound must be in [0, 180] degrees.")

    mu_min = np.cos(theta_bound_rad)
    if method == 'regular':
        # Equal-solid-angle ring centers inside the cone.
        mu_edges = np.linspace(1.0, mu_min, n_theta + 1)
        mu_vals = 0.5 * (mu_edges[:-1] + mu_edges[1:])
        theta_vals = np.arccos(mu_vals)
        phi_vals = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

        th_grid, ph_grid = np.meshgrid(theta_vals, phi_vals, indexing='ij')
        theta_flat = th_grid.ravel()
        phi_flat = ph_grid.ravel()
    else:
        # Uniform in solid angle over the cone for random directions.
        mu_rand = np.random.uniform(mu_min, 1.0, int(num_rays))
        theta_flat = np.arccos(mu_rand)
        phi_flat = np.random.uniform(0.0, 2.0 * np.pi, int(num_rays))

    theta_grid_deg = np.linspace(0.0, 180.0, beam_data.shape[0])
    phi_grid_deg = np.linspace(0.0, 360.0, beam_data.shape[1])
    beam_interp = RegularGridInterpolator(
        (theta_grid_deg, phi_grid_deg),
        beam_data,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    sample_points = np.column_stack([
        np.degrees(theta_flat),
        np.mod(np.degrees(phi_flat), 360.0),
    ])
    intensities = beam_interp(sample_points)

    # Always include one explicit on-axis ray.
    theta_flat = np.concatenate([theta_flat, np.array([0.0])])
    phi_flat = np.concatenate([phi_flat, np.array([0.0])])
    center_intensity = float(beam_interp(np.array([[0.0, 0.0]], dtype=float))[0])
    intensities = np.concatenate([intensities, np.array([center_intensity])])

    launch_global = source._to_global_point(launch_local).tolist()
    rays = []

    for th, ph, intensity in zip(theta_flat, phi_flat, intensities):
        local_dir = np.array([
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            direction_sign * np.cos(th),
        ], dtype=float)

        global_dir = source._to_global_dir(local_dir).tolist()
        rays.append([float(pol_angle), float(intensity), launch_global.copy(), global_dir, 0.0])

    return rays

'''Surface Definitions'''

def _asphere_sag(r, c, k, a2, a4, a6=0.0, a8=0.0):
    r2 = r**2
    return (c * r2) / (1 + np.sqrt(1 - (1 + k) * c**2 * r2)) + a2 * r2 + a4 * r2**2 + a6 * r2**3 + a8 * r2**4


def _asphere_multiplier(r, c, k, a2, a4, a6=0.0, a8=0.0):
    fac = np.sqrt(1 - (1 + k) * c**2 * r**2)
    return (
        2 * c / (1 + fac)
        + c**3 * (1 + k) * r**2 / (fac * (1 + fac) ** 2)
        + 2 * a2
        + 4 * a4 * r**2
        + 6 * a6 * r**4
        + 8 * a8 * r**6
    )

def _dihedral_sag(y, m, b, sign):
    '''Returns the sag of one side of a dihedral mirror; side is determined by sign'''
    return sign * (m * np.abs(y) + b)

'''Root-finding Functions'''

def newton_root(f, fprime, x0, iter, *args):
    x = x0
    for i in range(iter):
        fx = f(x, *args)
        fpx = fprime(x, *args)
        if fpx == 0:  # Avoid division by zero
            print("Derivative is zero. No solution found.")
            return None
        x = x - fx / fpx
    return x

# currently only supports root-finding using Newton's and Brent's methods; default is Brent but Newton is faster
def _normalize_root_finder(root_finder, default='brent'):
    if root_finder is None:
        return default
    method = str(root_finder).strip().lower()
    if method not in ('brent', 'newton'):
        raise ValueError(f"Unsupported root_finder '{root_finder}'. Use 'brent' or 'newton'.")
    return method

'''Helper classes for all optical elements'''
class _PoseMixin:
    '''Provides methods for converting between local and global coordinate systems.
    Parent classes must define their own origin, gx_local, gy_local, and gz_local attributes.'''
    def _rotation_matrices(self):
        if not hasattr(self, '_R_l2g_cache') or not hasattr(self, '_R_g2l_cache'):
            R_local_to_global = np.vstack([self.gx_local, self.gy_local, self.gz_local]) # pyright: ignore[reportAttributeAccessIssue]
            assert np.allclose(np.linalg.norm(R_local_to_global, axis=1), 1), "Basis vectors must be unit length"
            assert np.allclose(R_local_to_global @ R_local_to_global.T, np.eye(3), atol=1e-4), "Basis vectors must be orthogonal"
            self._R_l2g_cache = R_local_to_global
            self._R_g2l_cache = R_local_to_global.T
        return self._R_l2g_cache, self._R_g2l_cache

    def _rotation_matrix_local_to_global(self):
        R_l2g, _ = self._rotation_matrices()
        return R_l2g

    def _to_local(self, point_global, direction_global):
        _, R_g2l = self._rotation_matrices()
        origin = np.asarray(self.origin, dtype=float) # pyright: ignore[reportAttributeAccessIssue]
        p_local = R_g2l @ (np.asarray(point_global, dtype=float) - origin)
        d_local = R_g2l @ np.asarray(direction_global, dtype=float)
        return p_local, d_local

    def _to_local_point(self, point_global):
        _, R_g2l = self._rotation_matrices()
        origin = np.asarray(self.origin, dtype=float) # pyright: ignore[reportAttributeAccessIssue]
        return R_g2l @ (np.asarray(point_global, dtype=float) - origin)

    def _to_global_point(self, point_local):
        R_l2g, _ = self._rotation_matrices()
        origin = np.asarray(self.origin, dtype=float) # pyright: ignore[reportAttributeAccessIssue]
        return origin + R_l2g @ np.asarray(point_local, dtype=float)

    def _to_global_dir(self, dir_local):
        R_l2g, _ = self._rotation_matrices()
        d_global = R_l2g @ np.asarray(dir_local, dtype=float)
        return d_global / np.linalg.norm(d_global)


class _RayOpsMixin:
    def _reflect_ray(self, ray_dir, normal_vec):
        reflected_dir = ray_dir - 2 * np.dot(ray_dir, normal_vec) * normal_vec
        reflected_dir /= np.linalg.norm(reflected_dir)
        return reflected_dir

    def _refract_ray(self, ray_dir, normal_vec, n1, n2):
        I = np.asarray(ray_dir, dtype=float)
        I /= np.linalg.norm(I)
        N = np.asarray(normal_vec, dtype=float)
        N /= np.linalg.norm(N)

        cosi = np.clip(np.dot(I, N), -1.0, 1.0)
        if cosi > 0:
            N = -N
            cosi = np.dot(I, N)

        eta = n1 / n2
        k_term = 1.0 - eta**2 * (1.0 - cosi**2)
        if k_term < 0:
            return None

        refracted_dir = eta * I - (eta * cosi + np.sqrt(k_term)) * N
        refracted_dir /= np.linalg.norm(refracted_dir)
        return refracted_dir


''' Optical Element Classes	'''

@dataclass
class Object(_PoseMixin, _RayOpsMixin):
    origin: tuple
    diam: float
    gx_local: tuple
    gy_local: tuple
    gz_local: tuple

    def trace(self, starting_rays, n_scan=500, debug=False, root_finder=None):
        return starting_rays

    def plot(self, num_points=100, fig=None, opacity=0.5):
        x = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        y = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.where(R <= self.diam / 2, 0.0, np.nan)

        R_l2g = self._rotation_matrix_local_to_global()
        pts = np.stack((X, Y, Z), axis=-1) @ R_l2g.T + self.origin
        Xg, Yg, Zg = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]

        if fig is None:
            fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=Xg,
                y=Yg,
                z=Zg,
                colorscale="Gray",
                showscale=False,
                opacity=opacity,
                name="Object",
                showlegend=False,
            )
        )
        return fig, (Xg, Yg, Zg)

@dataclass
class Aperture(_PoseMixin, _RayOpsMixin):
    origin: tuple
    diam: float
    gx_local: tuple
    gy_local: tuple
    gz_local: tuple

    # for now Aperture objects must be planar

    def _find_intersection(self, ray, t_min=0.01):
        ray_origin = np.asarray(ray[2], dtype=float)
        ray_dir = np.asarray(ray[3], dtype=float)
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            return None
        ray_dir /= norm

        ray_origin_local, ray_dir_local = self._to_local(ray_origin, ray_dir)
        denom = ray_dir_local[2]
        if np.isclose(denom, 0.0, atol=1e-12):
            return None

        t_hit = -ray_origin_local[2] / denom
        if t_hit < t_min:
            return None

        p_local = ray_origin_local + t_hit * ray_dir_local

        # this removes rays outside of the aperture diameter
        if np.hypot(p_local[0], p_local[1]) > (self.diam / 2):
            return None

        return ray_origin + t_hit * ray_dir

    def trace(self, starting_rays, n_scan=500, debug=False, root_finder=None):
        new_rays = []
        for ray in starting_rays:
            p_hit = self._find_intersection(ray)
            if p_hit is not None:
                new_rays.append(ray)
        return new_rays
    
    def plot(self, num_points=100, fig=None, opacity=0.5):
        x = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        y = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.where(R <= self.diam / 2, 0.0, np.nan)

        R_l2g = self._rotation_matrix_local_to_global()
        pts = np.stack((X, Y, Z), axis=-1) @ R_l2g.T + self.origin
        Xg, Yg, Zg = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]

        if fig is None:
            fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=Xg,
                y=Yg,
                z=Zg,
                colorscale="Gray",
                showscale=False,
                opacity=opacity,
                name="Aperture",
                showlegend=False,
            )
        )
        return fig, (Xg, Yg, Zg)

@dataclass
class AsphericMirror(_PoseMixin, _RayOpsMixin):
    c: float
    k: float
    a_2: float
    a_4: float
    diam: float
    origin: tuple
    gx_local: tuple
    gy_local: tuple
    gz_local: tuple
    sign: int

    def _sag(self, r):
        return _asphere_sag(r, self.c, self.k, self.a_2, self.a_4)

    def _local_normal_vec(self, x, y):
        r = np.sqrt(x**2 + y**2)
        m = _asphere_multiplier(r, self.c, self.k, self.a_2, self.a_4)
        n_local = np.array([-x * m, -y * m, 1 * self.sign])
        n_local /= np.linalg.norm(n_local)
        return self.sign * n_local

    def normal_vec(self, x, y, z):
        p_local, _ = self._to_local((x, y, z), (0, 0, 1))
        n_local = self._local_normal_vec(p_local[0], p_local[1])
        return self._to_global_dir(n_local)

    '''Methods for Newton root-finding algorithm'''
    def _max_valid_radius_local(self):
        """Largest local radius where conic sag is numerically valid."""
        aperture_radius = self.diam / 2.0
        conic_term = (1.0 + self.k) * (self.c ** 2)
        if conic_term <= 0.0:
            return aperture_radius
        conic_radius = np.sqrt(1.0 / conic_term)
        # Keep tiny margin from singular boundary in sag/multiplier.
        return min(aperture_radius, conic_radius * (1.0 - 1e-9))

    def _sag_and_radial_slope_extended(self, r):
        """
        Safe real-valued continuation of sag outside conic validity.
        Inside valid radius: exact sag/slope.
        Outside valid radius: linear continuation from r_max.
        """
        r_max = self._max_valid_radius_local()
        if r <= r_max:
            sag = self._sag(r)
            m = _asphere_multiplier(r, self.c, self.k, self.a_2, self.a_4)
            dsag_dr = m * r
            return sag, dsag_dr

        sag_max = self._sag(r_max)
        m_max = _asphere_multiplier(r_max, self.c, self.k, self.a_2, self.a_4)
        dsag_dr_max = m_max * r_max
        sag_ext = sag_max + dsag_dr_max * (r - r_max)
        return sag_ext, dsag_dr_max

    '''End of Newton-specific methods'''

    def _find_intersection_newton(self, ray, guess=None, iter=10, t_vals=None, debug=False, t_min=0.01):
        # root-finder built using Newton's method
        if t_vals is None: 
            t_vals = np.linspace(0.01, 2000, 500)
        if guess is None:
            guess = np.linalg.norm(np.asarray(self.origin) - np.asarray(ray[2]))

        ray_origin = np.asarray(ray[2], dtype=float)
        ray_dir = np.asarray(ray[3], dtype=float)
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            if debug:
                print("No intersection found: ray direction has zero magnitude")
            return None
        ray_dir /= norm
        ray_origin_local, ray_dir_local = self._to_local(ray_origin, ray_dir)

        def distance_from_surface(t):
            x_local, y_local, z_local = ray_origin_local + t * ray_dir_local
            r2 = x_local**2 + y_local**2
            r = np.sqrt(r2)
            sag_eval, _ = self._sag_and_radial_slope_extended(r)
            if debug and (r > self._max_valid_radius_local()):
                print(f"distance_from_surface_test: using linear sag continuation at r={r}, t={t}")
            return z_local - self.sign * sag_eval
        
        def distance_from_surface_derivative(t):
            x_local, y_local, z_local = ray_origin_local + t * ray_dir_local
            r2 = x_local**2 + y_local**2
            r = np.sqrt(r2)
            _, dsag_dr = self._sag_and_radial_slope_extended(r)
            if r > 0:
                d_sag_dx = dsag_dr * (x_local / r)
                d_sag_dy = dsag_dr * (y_local / r)
            else:
                d_sag_dx = 0.0
                d_sag_dy = 0.0
            if debug and (r > self._max_valid_radius_local()):
                print(f"distance_from_surface_derivative: using linear-slope continuation at r={r}, t={t}")

            d_distance_dt = ray_dir_local[2] - self.sign * (d_sag_dx * ray_dir_local[0] + d_sag_dy * ray_dir_local[1])
            return d_distance_dt
        
        root = newton_root(distance_from_surface, distance_from_surface_derivative, guess, iter)
        if root is None or not np.isfinite(root) or root < t_min:
            return None

        p_local = ray_origin_local + root * ray_dir_local
        r2_local = p_local[0]**2 + p_local[1]**2
        if np.hypot(p_local[0], p_local[1]) > (self.diam / 2):
            return None
        if 1 - (1 + self.k) * self.c**2 * r2_local < 0:
            return None

        return ray_origin + root * ray_dir
    
    def _find_intersection_brentq(self, ray, t_min=0.01, t_max=2000.0, n_scan=500, debug=False):
        ray_origin = np.asarray(ray[2], dtype=float)
        ray_dir = np.asarray(ray[3], dtype=float)
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            if debug:
                print("No intersection found: ray direction has zero magnitude")
            return None
        ray_dir /= norm

        ray_origin_local, ray_dir_local = self._to_local(ray_origin, ray_dir)

        def distance_from_surface(t):
            x_local, y_local, z_local = ray_origin_local + t * ray_dir_local
            r2 = x_local**2 + y_local**2
            if 1 - (1 + self.k) * self.c**2 * r2 < 0:
                return np.nan
            return z_local - self.sign * self._sag(np.sqrt(r2))

        t_values = np.linspace(t_min, t_max, n_scan)
        f_values = np.array([distance_from_surface(t) for t in t_values])
        finite = np.isfinite(f_values)

        for i in range(len(t_values) - 1):
            if not (finite[i] and finite[i + 1]):
                continue
            f0, f1 = f_values[i], f_values[i + 1]
            if f0 == 0:
                t_hit = t_values[i]
            elif f1 == 0:
                t_hit = t_values[i + 1]
            elif np.sign(f0) != np.sign(f1):
                try:
                    t_hit = optimize.brentq(distance_from_surface, t_values[i], t_values[i + 1])
                except ValueError:
                    continue
            else:
                continue

            p_local = ray_origin_local + t_hit * ray_dir_local
            if np.hypot(p_local[0], p_local[1]) <= (self.diam / 2):
                return ray_origin + t_hit * ray_dir

        if debug:
            print("No intersection found for ray with launch position {} and direction {}".format(ray[2], ray[3]))
        return None

    def plot(self, num_points=100, fig=None, opacity=0.8):
        x = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        y = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.where(R <= self.diam / 2, self.sign * self._sag(R), np.nan)

        R_l2g = self._rotation_matrix_local_to_global()
        pts = np.stack((X, Y, Z), axis=-1) @ R_l2g.T + self.origin
        Xg, Yg, Zg = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]

        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Surface(x=Xg, y=Yg, z=Zg, colorscale="teal", showscale=False, opacity=opacity, name="AsphericMirror", showlegend=False))
        return fig, (Xg, Yg, Zg)

    def trace(self, starting_rays, n_scan=2000, debug=False, root_finder='brent'):
        method = _normalize_root_finder(root_finder, default='brent')
        new_rays = []
        origin_arr = np.asarray(self.origin, dtype=float)
        for ray in starting_rays:
            pol_angle, intensity = ray[0], ray[1]
            launch_position = np.asarray(ray[2], dtype=float).copy()
            direction = np.asarray(ray[3], dtype=float).copy()
            total_distance = float(ray[4])

            ray_copy = [pol_angle, intensity, launch_position, direction, total_distance]
            if method == 'newton':
                guess = np.linalg.norm(origin_arr - launch_position)
                p_hit = self._find_intersection_newton(ray_copy, guess=guess, debug=debug)
            else:
                p_hit = self._find_intersection_brentq(ray_copy, n_scan=n_scan, debug=debug)
            if p_hit is None:
                continue

            normal_vec = self.normal_vec(*p_hit)
            reflected_dir = self._reflect_ray(direction, normal_vec)
            segment = np.linalg.norm(p_hit - launch_position)
            new_rays.append([pol_angle, intensity, p_hit, reflected_dir, total_distance + segment])
        return new_rays


@dataclass
class FlatMirror(_PoseMixin, _RayOpsMixin):
    diam: float
    origin: tuple
    gx_local: tuple
    gy_local: tuple
    gz_local: tuple

    def normal_vec(self, x, y, z):
        return self._to_global_dir(np.array([0.0, 0.0, 1.0]))

    def _find_intersection(self, ray, t_min=0.01, debug=False):
        ray_origin = np.asarray(ray[2], dtype=float)
        ray_dir = np.asarray(ray[3], dtype=float)
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            if debug:
                print("No intersection found: ray direction has zero magnitude")
            return None
        ray_dir /= norm

        ray_origin_local, ray_dir_local = self._to_local(ray_origin, ray_dir)
        denom = ray_dir_local[2]
        if np.isclose(denom, 0.0, atol=1e-12):
            if debug:
                print("No intersection found: ray is parallel to flat mirror")
            return None

        t_hit = -ray_origin_local[2] / denom
        if t_hit < t_min:
            if debug:
                print("No intersection found: flat-mirror hit is behind ray origin")
            return None

        p_local = ray_origin_local + t_hit * ray_dir_local
        if np.hypot(p_local[0], p_local[1]) > (self.diam / 2):
            if debug:
                print("No intersection found: flat-mirror hit is outside aperture")
            return None
        return ray_origin + t_hit * ray_dir

    def plot(self, num_points=100, fig=None, opacity=0.8):
        x = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        y = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        R_l2g = self._rotation_matrix_local_to_global()
        pts = np.stack((X, Y, Z), axis=-1) @ R_l2g.T + self.origin
        Xg, Yg, Zg = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]

        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Surface(x=Xg, y=Yg, z=Zg, colorscale="teal", showscale=False, opacity=opacity, name="FlatMirror", showlegend=False))
        return fig, (Xg, Yg, Zg)

    def trace(self, starting_rays, n_scan=2000, debug=False, root_finder=None):
        new_rays = []
        for ray in starting_rays:
            pol_angle, intensity = ray[0], ray[1]
            launch_position = np.asarray(ray[2], dtype=float).copy()
            direction = np.asarray(ray[3], dtype=float).copy()
            total_distance = float(ray[4])

            ray_copy = [pol_angle, intensity, launch_position, direction, total_distance]
            p_hit = self._find_intersection(ray_copy, debug=debug)
            if p_hit is None:
                continue

            normal_vec = self.normal_vec(*p_hit)
            reflected_dir = self._reflect_ray(direction, normal_vec)
            segment = np.linalg.norm(p_hit - launch_position)
            new_rays.append([pol_angle, intensity, p_hit, reflected_dir, total_distance + segment])
        return new_rays


@dataclass
class DihedralMirror(_PoseMixin, _RayOpsMixin):
    """
    Dihedral mirror with two flat surfaces at an angle.
    Each ray interacts with only one side based on its direction relative to the local z-axis.
    - If ray has positive z component → interacts with negative side (sign=-1)
    - If ray has negative z component → interacts with positive side (sign=+1)
    
    The dihedral surface is defined by z = sign * (m * |y| + b)
    where m is the slope and b is the offset.
    
    Rays can bounce multiple times on the same surface before exiting.
    """
    m: float  # slope parameter for dihedral
    b: float  # offset parameter for dihedral
    diam: float  # diameter to limit xy extent
    origin: tuple
    gx_local: tuple
    gy_local: tuple
    gz_local: tuple
    max_bounces: int = field(default=10, kw_only=True)  # maximum number of bounces per ray
    
    def _surface_z(self, y, sign):
        """Return z coordinate for the dihedral surface at given y with sign."""
        return _dihedral_sag(y, self.m, self.b, sign)
    
    def _surface_normal(self, y, sign):
        """Return the outward-pointing normal vector at position y on the surface."""
        # For z = sign * (m * |y| + b), we have:
        # dz/dy = sign * m * sign(y) = sign * m * (y/|y|) for y != 0
        # The tangent vector along y is [0, 1, dz/dy]
        # The tangent vector along x is [1, 0, 0]
        # Normal = tangent_x cross tangent_y
        if np.abs(y) < 1e-12:
            # At y=0, the surface has a kink; use normal based on sign
            dz_dy = sign * self.m  # arbitrary choice for y > 0
        else:
            dz_dy = sign * self.m * np.sign(y)
        
        # tangent_y = [0, 1, dz_dy]
        # tangent_x = [1, 0, 0]
        # normal = tangent_x × tangent_y = [0, -dz_dy, 1]
        normal_local = np.array([0.0, -dz_dy, 1.0])
        normal_local /= np.linalg.norm(normal_local)
        return normal_local
    
    def _find_intersection_analytical(self, ray_origin_local, ray_dir_local, sign, t_min=0.01, debug=False):
        """
        Find intersection with dihedral surface analytically.
        Surface: z = sign * (m * |y| + b)
        Ray: p = p0 + t * d
        
        We need to solve: z0 + t*dz = sign * (m * |y0 + t*dy| + b)
        """
        x0, y0, z0 = ray_origin_local
        dx, dy, dz = ray_dir_local
        
        # Check if ray is essentially parallel to the surface
        if np.abs(dz) < 1e-12 and np.abs(dy) < 1e-12:
            if debug:
                print(f"No intersection: ray parallel to dihedral surface (sign={sign})")
            return None
        
        # We need to solve: z0 + t*dz = sign * (m * |y0 + t*dy| + b)
        # This splits into two cases based on the sign of (y0 + t*dy)
        
        solutions = []
        
        # Case 1: y0 + t*dy >= 0, so |y0 + t*dy| = y0 + t*dy
        # z0 + t*dz = sign * (m * (y0 + t*dy) + b)
        # z0 + t*dz = sign*m*y0 + sign*m*t*dy + sign*b
        # t*(dz - sign*m*dy) = sign*m*y0 + sign*b - z0
        denom1 = dz - sign * self.m * dy
        if np.abs(denom1) > 1e-12:
            t1 = (sign * self.m * y0 + sign * self.b - z0) / denom1
            y1 = y0 + t1 * dy
            if t1 >= t_min and y1 >= -1e-12:  # Check that y is non-negative
                solutions.append(t1)
        
        # Case 2: y0 + t*dy < 0, so |y0 + t*dy| = -(y0 + t*dy)
        # z0 + t*dz = sign * (m * (-(y0 + t*dy)) + b)
        # z0 + t*dz = -sign*m*y0 - sign*m*t*dy + sign*b
        # t*(dz + sign*m*dy) = -sign*m*y0 + sign*b - z0
        denom2 = dz + sign * self.m * dy
        if np.abs(denom2) > 1e-12:
            t2 = (-sign * self.m * y0 + sign * self.b - z0) / denom2
            y2 = y0 + t2 * dy
            if t2 >= t_min and y2 <= 1e-12:  # Check that y is non-positive
                solutions.append(t2)
        
        if not solutions:
            if debug:
                print(f"No valid intersection found (sign={sign})")
            return None
        
        # Take the smallest valid t
        t_hit = min(solutions)
        p_local = ray_origin_local + t_hit * ray_dir_local
        
        # Check if hit is within diameter
        r_hit = np.hypot(p_local[0], p_local[1])
        if r_hit > self.diam / 2:
            if debug:
                print(f"Intersection outside diameter: r={r_hit}, diam/2={self.diam/2}")
            return None
        
        return ray_origin_local + t_hit * ray_dir_local  # Return in local coordinates
    
    def normal_vec(self, x, y, z):
        """Return the global normal vector at a point (only y matters for dihedral)."""
        # Determine which surface based on z
        # For now, we'll determine based on z relative to centerline
        z_center = self._surface_z(0, 1)  # Both surfaces meet at y=0
        if z > z_center:
            sign = 1
        else:
            sign = -1
        normal_local = self._surface_normal(y, sign)
        return self._to_global_dir(normal_local)
    
    def _find_intersection(self, ray, t_min=0.01, debug=False):
        """Find intersection with the appropriate side of the dihedral mirror."""
        ray_origin = np.asarray(ray[2], dtype=float)
        ray_dir = np.asarray(ray[3], dtype=float)
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            if debug:
                print("No intersection: ray direction has zero magnitude")
            return None
        ray_dir /= norm
        
        ray_origin_local, ray_dir_local = self._to_local(ray_origin, ray_dir)
        
        # Determine which surface to interact with based on ray direction
        # Positive z component → interact with negative side (sign=-1)
        # Negative z component → interact with positive side (sign=+1)
        if ray_dir_local[2] > 0:
            sign = -1
        elif ray_dir_local[2] < 0:
            sign = 1
        else:
            # Ray is exactly perpendicular to z-axis; default to positive side
            sign = 1
        
        if debug:
            print(f"Ray z-component: {ray_dir_local[2]:.6f}, using surface with sign={sign}")
        
        p_local = self._find_intersection_analytical(ray_origin_local, ray_dir_local, sign, t_min=t_min, debug=debug)
        if p_local is None:
            return None
        
        return self._to_global_point(p_local)
    
    def plot(self, num_points=100, fig=None, opacity=0.8):
        """Plot both surfaces of the dihedral mirror."""
        y = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        x = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        Y, X = np.meshgrid(y, x)
        
        # Only plot points within the circular aperture
        R = np.sqrt(X**2 + Y**2)
        mask = R <= self.diam / 2
        
        if fig is None:
            fig = go.Figure()
        
        R_l2g = self._rotation_matrix_local_to_global()
        
        # Plot positive side (sign=+1)
        Z_pos = _dihedral_sag(Y, self.m, self.b, 1)
        Z_pos = np.where(mask, Z_pos, np.nan)
        pts_pos = np.stack((X, Y, Z_pos), axis=-1) @ R_l2g.T + self.origin
        Xg_pos, Yg_pos, Zg_pos = pts_pos[:, :, 0], pts_pos[:, :, 1], pts_pos[:, :, 2]
        fig.add_trace(go.Surface(
            x=Xg_pos, y=Yg_pos, z=Zg_pos,
            colorscale="Blues", showscale=False, opacity=opacity,
            name="DihedralMirror (+)",
            showlegend=False
        ))
        
        # Plot negative side (sign=-1)
        Z_neg = _dihedral_sag(Y, self.m, self.b, -1)
        Z_neg = np.where(mask, Z_neg, np.nan)
        pts_neg = np.stack((X, Y, Z_neg), axis=-1) @ R_l2g.T + self.origin
        Xg_neg, Yg_neg, Zg_neg = pts_neg[:, :, 0], pts_neg[:, :, 1], pts_neg[:, :, 2]
        fig.add_trace(go.Surface(
            x=Xg_neg, y=Yg_neg, z=Zg_neg,
            colorscale="Reds", showscale=False, opacity=opacity,
            name="DihedralMirror (-)",
            showlegend=False
        ))
        
        return fig, ((Xg_pos, Yg_pos, Zg_pos), (Xg_neg, Yg_neg, Zg_neg))
    
    def trace(self, starting_rays, n_scan=2000, debug=False, root_finder=None, return_ports=False):
        """
        Trace rays through the dihedral mirror with multiple bounces.
        Each ray enters one V-channel (determined by initial direction) and bounces
        multiple times within that same surface until it exits the aperture.
        Supports both sequential and branch modes via return_ports.
        """
        new_rays = []
        
        for ray in starting_rays:
            pol_angle, intensity = ray[0], ray[1]
            launch_position = np.asarray(ray[2], dtype=float).copy()
            direction = np.asarray(ray[3], dtype=float).copy()
            total_distance = float(ray[4])
            
            # First, check if ray hits the mirror at all
            ray_copy = [pol_angle, intensity, launch_position, direction, total_distance]
            p_hit_first = self._find_intersection(ray_copy, debug=debug)
            
            if p_hit_first is None:
                # Ray missed the mirror entirely
                if return_ports:
                    new_rays.append(("default", ray_copy))
                else:
                    new_rays.append(ray_copy)
                continue
            
            # Determine which V-channel (which sign) based on INITIAL direction only
            _, ray_dir_local_initial = self._to_local(launch_position, direction)
            if ray_dir_local_initial[2] > 0:
                surface_sign = -1  # Interact with negative V (opening downward)
            elif ray_dir_local_initial[2] < 0:
                surface_sign = 1   # Interact with positive V (opening upward)
            else:
                surface_sign = 1   # Default to positive
            
            if debug:
                print(f"\nRay initial z-component: {ray_dir_local_initial[2]:.6f}, V-channel sign={surface_sign}")
                print(f"Ray will bounce on sign={surface_sign} surface for all bounces")
            
            # Now trace multiple bounces within the same V-channel (same sign)
            current_position = launch_position.copy()
            current_direction = direction.copy()
            accumulated_distance = total_distance
            bounce_count = 0
            
            while bounce_count < self.max_bounces:
                # Find intersection with the V-surface (always using the same sign)
                p_local, d_local = self._to_local(current_position, current_direction)
                p_hit_local = self._find_intersection_analytical(
                    p_local, d_local,
                    surface_sign,  # ALWAYS use the same sign determined at the start
                    t_min=1e-6,  # Small value to avoid self-intersection
                    debug=debug
                )
                
                if p_hit_local is None:
                    # Ray exits the dihedral mirror (no more intersections within aperture)
                    if debug:
                        print(f"Bounce {bounce_count}: Ray exits (no more intersections in aperture)")
                    break
                
                bounce_count += 1
                p_hit_global = self._to_global_point(p_hit_local)
                
                # Calculate distance traveled in this segment
                segment_distance = np.linalg.norm(p_hit_global - current_position)
                accumulated_distance += segment_distance
                
                # Get surface normal at hit point
                normal_local = self._surface_normal(p_hit_local[1], surface_sign)
                normal_global = self._to_global_dir(normal_local)
                
                # Reflect the ray
                reflected_dir = self._reflect_ray(current_direction, normal_global)
                
                if debug:
                    print(f"Bounce {bounce_count}: hit at y_local={p_hit_local[1]:.4f}, z_local={p_hit_local[2]:.4f}, " + 
                          f"segment={segment_distance:.4f}, total_dist={accumulated_distance:.4f}")
                
                # Update for next iteration (continue bouncing in same V-channel)
                current_position = p_hit_global
                current_direction = reflected_dir
            
            if debug:
                if bounce_count >= self.max_bounces:
                    print(f"Reached max_bounces ({self.max_bounces})")
                print(f"Total bounces: {bounce_count}")
            
            # Create final output ray
            final_ray = [pol_angle, intensity, current_position, current_direction, accumulated_distance]
            
            if return_ports:
                # Use port names to indicate which V-channel was used
                port = "pos" if surface_sign > 0 else "neg"
                new_rays.append((port, final_ray))
            else:
                new_rays.append(final_ray)
        
        return new_rays


@dataclass
class _BaseTwoSurfaceLens(_PoseMixin, _RayOpsMixin):
    diam: float
    thickness: float
    origin: tuple
    gx_local: tuple
    gy_local: tuple
    gz_local: tuple
    n_lens: float
    n_environment: float = field(default=1.0, kw_only=True)

    # Subclasses must implement these hooks.
    def _surface_names(self) -> tuple[str, str]:
        return ("front", "back")

    def _surface_z(self, surface, x, y):
        raise NotImplementedError

    def _surface_normal_local(self, surface, x, y):
        raise NotImplementedError

    def _surface_domain_valid(self, surface, x, y):
        return True

    def _is_inside_lens_local(self, point_local):
        raise NotImplementedError

    def _surface_plot_name(self, surface):
        return f"{self.__class__.__name__}-{surface.capitalize()}"

    def _surface_z_and_grad_local_newton(self, surface, x, y, debug=False) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
        """Return (z_surface, dz/dx, dz/dy, used_extended_sag) for Newton evaluations."""
        if not self._surface_domain_valid(surface, x, y):
            return None, None, None, False

        z_surface = self._surface_z(surface, x, y)
        n_local = np.asarray(self._surface_normal_local(surface, x, y), dtype=float)
        norm_n = np.linalg.norm(n_local)
        if norm_n == 0:
            return None, None, None, False
        n_local /= norm_n

        nz = n_local[2]
        if np.isclose(nz, 0.0, atol=1e-12):
            return None, None, None, False

        dzdx = -n_local[0] / nz
        dzdy = -n_local[1] / nz
        return z_surface, dzdx, dzdy, False

    def _find_surface_intersection_local_newton(self, ray_origin_local, ray_dir_local, surface, t_min=0.01, t_max=2000.0, n_scan=500, guess=None, iter=10, t_vals=None, debug=False):
        # root-finder built using Newton's method
        if t_vals is None:
            t_vals = np.linspace(t_min, t_max, n_scan)
        if guess is None:
            surface_center = np.array([0.0, 0.0, float(self._surface_z(surface, 0.0, 0.0))], dtype=float)
            guess = np.linalg.norm(surface_center - np.asarray(ray_origin_local, dtype=float))

        def distance_from_surface(t):
            x_local, y_local, z_local = ray_origin_local + t * ray_dir_local
            z_surface, _, _, _ = self._surface_z_and_grad_local_newton(surface, x_local, y_local, debug=debug)
            if z_surface is None:
                return np.nan
            return z_local - z_surface

        def distance_from_surface_derivative(t):
            x_local, y_local, z_local = ray_origin_local + t * ray_dir_local
            _, dzdx, dzdy, _ = self._surface_z_and_grad_local_newton(surface, x_local, y_local, debug=debug)
            if dzdx is None:
                return 0.0
            return ray_dir_local[2] - (dzdx * ray_dir_local[0] + dzdy * ray_dir_local[1])

        root = newton_root(distance_from_surface, distance_from_surface_derivative, guess, iter)
        if root is None or not np.isfinite(root):
            if debug:
                print(f"No Newton intersection found for surface '{surface}'")
            return None, None

        p_local = ray_origin_local + root * ray_dir_local
        if np.hypot(p_local[0], p_local[1]) <= (self.diam / 2):
            return root, p_local
        if debug:
            print(f"Newton intersection for surface '{surface}' fell outside aperture")
        return None, None

    def _find_surface_intersection_local_brentq(self, ray_origin_local, ray_dir_local, surface, t_min=0.01, t_max=2000.0, n_scan=500, debug=False):
        def distance_from_surface(t):
            x_local, y_local, z_local = ray_origin_local + t * ray_dir_local
            if not self._surface_domain_valid(surface, x_local, y_local):
                return np.nan
            return z_local - self._surface_z(surface, x_local, y_local)

        t_values = np.linspace(t_min, t_max, n_scan)
        f_values = np.array([distance_from_surface(t) for t in t_values])
        finite = np.isfinite(f_values)

        for i in range(len(t_values) - 1):
            if not (finite[i] and finite[i + 1]):
                continue
            f0, f1 = f_values[i], f_values[i + 1]
            if f0 == 0:
                t_hit = t_values[i]
            elif f1 == 0:
                t_hit = t_values[i + 1]
            elif np.sign(f0) != np.sign(f1):
                try:
                    t_hit = optimize.brentq(distance_from_surface, t_values[i], t_values[i + 1])
                except ValueError:
                    continue
            else:
                continue

            p_local = ray_origin_local + t_hit * ray_dir_local
            if np.hypot(p_local[0], p_local[1]) <= (self.diam / 2):
                return t_hit, p_local

        if debug:
            print(f"No Brent intersection found for surface '{surface}'")
        return None, None

    def _find_surface_intersection_local(self, ray_origin_local, ray_dir_local, surface, t_min=0.01, t_max=2000.0, n_scan=500, root_finder='brent', debug=False):
        method = _normalize_root_finder(root_finder, default='brent')
        if method == 'newton':
            return self._find_surface_intersection_local_newton(ray_origin_local, ray_dir_local, surface, t_min=t_min, t_max=t_max, n_scan=n_scan, debug=debug)
        return self._find_surface_intersection_local_brentq(ray_origin_local, ray_dir_local, surface, t_min=t_min, t_max=t_max, n_scan=n_scan, debug=debug)

    def plot(self, num_points=100, fig=None, opacity=0.8):
        x = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        y = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        if fig is None:
            fig = go.Figure()
        R_l2g = self._rotation_matrix_local_to_global()

        grids = []
        colors = ("teal", "ice")
        for idx, surface in enumerate(self._surface_names()):
            Z = self._surface_z(surface, X, Y)
            Z = np.where(R <= self.diam / 2, Z, np.nan)
            pts = np.stack((X, Y, Z), axis=-1) @ R_l2g.T + self.origin
            Xg, Yg, Zg = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
            fig.add_trace(
                go.Surface(
                    x=Xg,
                    y=Yg,
                    z=Zg,
                    colorscale=colors[idx % len(colors)],
                    showscale=False,
                    opacity=opacity,
                    name=self._surface_plot_name(surface),
                    showlegend=False,
                )
            )
            grids.append((Xg, Yg, Zg))
        return fig, tuple(grids)

    def trace(self, starting_rays, n_scan=2000, debug=False, eps=1e-6, root_finder='brent'):
        surf_a, surf_b = self._surface_names()
        new_rays = []

        for ray in starting_rays:
            pol_angle, intensity = ray[0], ray[1]
            launch_position = np.asarray(ray[2], dtype=float).copy()
            direction = np.asarray(ray[3], dtype=float).copy()
            direction_norm = np.linalg.norm(direction)
            total_distance = float(ray[4])

            if direction_norm == 0:
                new_rays.append([pol_angle, intensity, launch_position, direction, total_distance])
                continue
            direction /= direction_norm

            p0_local, d0_local = self._to_local(launch_position, direction)
            current_n = self.n_lens if self._is_inside_lens_local(p0_local) else self.n_environment

            t_a, p_a = self._find_surface_intersection_local(p0_local, d0_local, surf_a, n_scan=n_scan, root_finder=root_finder, debug=debug)
            t_b, p_b = self._find_surface_intersection_local(p0_local, d0_local, surf_b, n_scan=n_scan, root_finder=root_finder, debug=debug)

            if t_a is None and t_b is None:
                new_rays.append([pol_angle, intensity, launch_position, direction, total_distance])
                continue

            if t_a is not None and (t_b is None or t_a <= t_b):
                first_surface, second_surface, p1_local = surf_a, surf_b, p_a
            else:
                first_surface, second_surface, p1_local = surf_b, surf_a, p_b

            p1_global = self._to_global_point(p1_local)
            n1_global = self._to_global_dir(self._surface_normal_local(first_surface, p1_local[0], p1_local[1])) # pyright: ignore[reportOptionalSubscript]

            next_n = self.n_environment if np.isclose(current_n, self.n_lens) else self.n_lens
            d1_global = self._refract_ray(direction, n1_global, current_n, next_n)

            segment_1 = np.linalg.norm(p1_global - launch_position)
            opl_1 = current_n * segment_1
            if d1_global is None:
                new_rays.append([pol_angle, intensity, p1_global, self._reflect_ray(direction, n1_global), total_distance + opl_1])
                continue

            p1_local_offset, d1_local = self._to_local(p1_global + eps * d1_global, d1_global)
            t2, p2_local = self._find_surface_intersection_local(p1_local_offset, d1_local, second_surface, n_scan=n_scan, root_finder=root_finder, debug=debug)
            if t2 is None:
                new_rays.append([pol_angle, intensity, p1_global, d1_global, total_distance + opl_1])
                continue

            p2_global = self._to_global_point(p2_local)
            n2_global = self._to_global_dir(self._surface_normal_local(second_surface, p2_local[0], p2_local[1])) # pyright: ignore[reportOptionalSubscript]
            d2_global = self._refract_ray(d1_global, n2_global, next_n, current_n)
            if d2_global is None:
                d2_global = self._reflect_ray(d1_global, n2_global)

            segment_2 = np.linalg.norm(p2_global - p1_global)
            opl_2 = next_n * segment_2
            new_rays.append([pol_angle, intensity, p2_global, d2_global, total_distance + opl_1 + opl_2])

        return new_rays


@dataclass
class AsphericLens(_BaseTwoSurfaceLens):
    c: float
    k: float
    a_2: float
    a_4: float
    a_6: float
    a_8: float
    sign: int = 1

    def _surface_names(self) -> tuple[str, str]:
        return ("asphere", "flat")

    '''Methods for Newton root-finding algorithm'''
    def _max_valid_radius_local(self):
        """Largest local radius where the asphere sag is numerically valid."""
        aperture_radius = self.diam / 2.0
        conic_term = (1.0 + self.k) * (self.c ** 2)
        if conic_term <= 0.0:
            return aperture_radius
        conic_radius = np.sqrt(1.0 / conic_term)
        return min(aperture_radius, conic_radius * (1.0 - 1e-9))

    def _sag_and_radial_slope_extended(self, r):
        """
        Safe real-valued continuation of asphere sag outside conic validity.
        Inside valid radius: exact sag/slope.
        Outside valid radius: linear continuation from r_max.
        """
        r_max = self._max_valid_radius_local()
        if r <= r_max:
            sag = self._asphere_sag(r)
            m = _asphere_multiplier(r, self.c, self.k, self.a_2, self.a_4, self.a_6, self.a_8)
            dsag_dr = m * r
            return sag, dsag_dr, False

        sag_max = self._asphere_sag(r_max)
        m_max = _asphere_multiplier(r_max, self.c, self.k, self.a_2, self.a_4, self.a_6, self.a_8)
        dsag_dr_max = m_max * r_max
        sag_ext = sag_max + dsag_dr_max * (r - r_max)
        return sag_ext, dsag_dr_max, True
    '''End of Newton-specific methods'''

    def _asphere_sag(self, r):
        return _asphere_sag(r, self.c, self.k, self.a_2, self.a_4, self.a_6, self.a_8)

    def _surface_z(self, surface, x, y):
        if surface == "asphere":
            return self.sign * self._asphere_sag(np.sqrt(x**2 + y**2))
        return np.full_like(x, self.thickness) if isinstance(x, np.ndarray) else self.thickness

    def _surface_normal_local(self, surface, x, y):
        if surface == "flat":
            return np.array([0.0, 0.0, 1.0])
        r = np.sqrt(x**2 + y**2)
        m = _asphere_multiplier(r, self.c, self.k, self.a_2, self.a_4, self.a_6, self.a_8)
        n_local = np.array([-x * m, -y * m, 1 * self.sign])
        n_local /= np.linalg.norm(n_local)
        return n_local

    def _surface_z_and_grad_local_newton(self, surface, x, y, debug=False) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
        if surface != "asphere":
            return super()._surface_z_and_grad_local_newton(surface, x, y, debug=debug)

        r = np.sqrt(x**2 + y**2)
        sag_eval, dsag_dr, used_ext = self._sag_and_radial_slope_extended(r)
        if debug and used_ext:
            print(f"AsphericLens: using linear sag continuation at r={r}")

        z_surface = self.sign * sag_eval
        if r > 0:
            dzdx = self.sign * dsag_dr * (x / r)
            dzdy = self.sign * dsag_dr * (y / r)
        else:
            dzdx = 0.0
            dzdy = 0.0
        return z_surface, dzdx, dzdy, used_ext

    def _surface_domain_valid(self, surface, x, y):
        if surface != "asphere":
            return True
        r2 = x**2 + y**2
        return (1 - (1 + self.k) * self.c**2 * r2) >= 0

    def _is_inside_lens_local(self, point_local):
        x, y, z = point_local
        if np.hypot(x, y) > (self.diam / 2):
            return False
        z_front = self.sign * self._asphere_sag(np.hypot(x, y))
        return z_front <= z <= self.thickness

    def _surface_plot_name(self, surface):
        return "AsphericLens-Asphere" if surface == "asphere" else "AsphericLens-Flat"


@dataclass
class FlatLens(_BaseTwoSurfaceLens):
    def _surface_z(self, surface, x, y):
        z = 0.0 if surface == "front" else self.thickness
        return np.full_like(x, z) if isinstance(x, np.ndarray) else z

    def _surface_normal_local(self, surface, x, y):
        return np.array([0.0, 0.0, 1.0])

    def _is_inside_lens_local(self, point_local):
        x, y, z = point_local
        if np.hypot(x, y) > (self.diam / 2):
            return False
        return 0.0 <= z <= self.thickness

    def _surface_plot_name(self, surface):
        return f"FlatLens-{surface.capitalize()}"


@dataclass
class DoubleAsphericLens(_BaseTwoSurfaceLens):
    c_1: float
    k_1: float
    a2_1: float
    a4_1: float
    a6_1: float
    a8_1: float
    c_2: float
    k_2: float
    a2_2: float
    a4_2: float
    a6_2: float
    a8_2: float
    sign_1: int = 1
    sign_2: int = -1

    '''Methods for Newton root-finding algorithm'''
    def _surface_params(self, surface):
        if surface == "front":
            return self.c_1, self.k_1, self.a2_1, self.a4_1, self.a6_1, self.a8_1, self.sign_1, 0.0
        return self.c_2, self.k_2, self.a2_2, self.a4_2, self.a6_2, self.a8_2, self.sign_2, self.thickness

    def _max_valid_radius_local(self, surface):
        """Largest local radius where the given asphere sag is numerically valid."""
        c, k, _, _, _, _, _, _ = self._surface_params(surface)
        aperture_radius = self.diam / 2.0
        conic_term = (1.0 + k) * (c ** 2)
        if conic_term <= 0.0:
            return aperture_radius
        conic_radius = np.sqrt(1.0 / conic_term)
        return min(aperture_radius, conic_radius * (1.0 - 1e-9))

    def _sag_and_radial_slope_extended(self, surface, r):
        """
        Safe real-valued continuation of sag outside conic validity for each asphere.
        Inside valid radius: exact sag/slope.
        Outside valid radius: linear continuation from r_max.
        """
        c, k, a2, a4, a6, a8, _, _ = self._surface_params(surface)
        r_max = self._max_valid_radius_local(surface)
        if r <= r_max:
            sag = _asphere_sag(r, c, k, a2, a4, a6, a8)
            m = _asphere_multiplier(r, c, k, a2, a4, a6, a8)
            dsag_dr = m * r
            return sag, dsag_dr, False

        sag_max = _asphere_sag(r_max, c, k, a2, a4, a6, a8)
        m_max = _asphere_multiplier(r_max, c, k, a2, a4, a6, a8)
        dsag_dr_max = m_max * r_max
        sag_ext = sag_max + dsag_dr_max * (r - r_max)
        return sag_ext, dsag_dr_max, True
    '''End of Newton-specific methods'''

    def _front_sag(self, r):
        return _asphere_sag(r, self.c_1, self.k_1, self.a2_1, self.a4_1, self.a6_1, self.a8_1)

    def _back_sag(self, r):
        return _asphere_sag(r, self.c_2, self.k_2, self.a2_2, self.a4_2, self.a6_2, self.a8_2)

    def _surface_z(self, surface, x, y):
        r = np.sqrt(x**2 + y**2)
        if surface == "front":
            return self.sign_1 * self._front_sag(r)
        return self.thickness + self.sign_2 * self._back_sag(r)

    def _surface_normal_local(self, surface, x, y):
        r = np.sqrt(x**2 + y**2)
        if surface == "front":
            m = _asphere_multiplier(r, self.c_1, self.k_1, self.a2_1, self.a4_1, self.a6_1, self.a8_1)
            sign = self.sign_1
        else:
            m = _asphere_multiplier(r, self.c_2, self.k_2, self.a2_2, self.a4_2, self.a6_2, self.a8_2)
            sign = self.sign_2
        n_local = np.array([-x * m, -y * m, 1 * sign])
        n_local /= np.linalg.norm(n_local)
        return n_local

    def _surface_z_and_grad_local_newton(self, surface, x, y, debug=False) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
        r = np.sqrt(x**2 + y**2)
        sag_eval, dsag_dr, used_ext = self._sag_and_radial_slope_extended(surface, r)
        _, _, _, _, _, _, sign, z_offset = self._surface_params(surface)
        if debug and used_ext:
            print(f"DoubleAsphericLens ({surface}): using linear sag continuation at r={r}")

        z_surface = z_offset + sign * sag_eval
        if r > 0:
            dzdx = sign * dsag_dr * (x / r)
            dzdy = sign * dsag_dr * (y / r)
        else:
            dzdx = 0.0
            dzdy = 0.0
        return z_surface, dzdx, dzdy, used_ext

    def _surface_domain_valid(self, surface, x, y):
        r2 = x**2 + y**2
        if surface == "front":
            return (1 - (1 + self.k_1) * self.c_1**2 * r2) >= 0
        return (1 - (1 + self.k_2) * self.c_2**2 * r2) >= 0

    def _is_inside_lens_local(self, point_local):
        x, y, z = point_local
        if np.hypot(x, y) > (self.diam / 2):
            return False
        z_front = self.sign_1 * self._front_sag(np.hypot(x, y))
        z_back = self.thickness + self.sign_2 * self._back_sag(np.hypot(x, y))
        return min(z_front, z_back) <= z <= max(z_front, z_back)

    def _surface_plot_name(self, surface):
        return f"DoubleAsphericLens-{surface.capitalize()}"


@dataclass
class FocalPlane(_PoseMixin):
    origin: tuple
    diam: float
    gx_local: tuple
    gy_local: tuple
    gz_local: tuple

    def _find_intersection(self, ray, t_min=0.01):
        ray_origin = np.asarray(ray[2], dtype=float)
        ray_dir = np.asarray(ray[3], dtype=float)
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            return None
        ray_dir /= norm

        ray_origin_local, ray_dir_local = self._to_local(ray_origin, ray_dir)
        denom = ray_dir_local[2]
        if np.isclose(denom, 0.0, atol=1e-12):
            return None

        t_hit = -ray_origin_local[2] / denom
        if t_hit < t_min:
            return None

        p_local = ray_origin_local + t_hit * ray_dir_local
        if np.hypot(p_local[0], p_local[1]) > (self.diam / 2):
            return None

        return ray_origin + t_hit * ray_dir

    def plot(self, num_points=100, fig=None, opacity=0.5, colorscale="Gray"):
        x = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        y = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.where(R <= self.diam / 2, 0.0, np.nan)

        R_l2g = self._rotation_matrix_local_to_global()
        pts = np.stack((X, Y, Z), axis=-1) @ R_l2g.T + self.origin
        Xg, Yg, Zg = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]

        if fig is None:
            fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=Xg,
                y=Yg,
                z=Zg,
                colorscale=colorscale,
                showscale=False,
                opacity=opacity,
                name="FocalPlane",
                showlegend=False,
            )
        )
        return fig, (Xg, Yg, Zg)

    def trace(self, starting_rays, n_scan=2000, debug=False, root_finder=None, t_min = 0.01):
        # Return only rays that hit the focal plane aperture.
        hit_rays = []
        for ray in starting_rays:
            pol_angle, intensity = ray[0], ray[1]
            launch_position = np.asarray(ray[2], dtype=float).copy()
            direction = np.asarray(ray[3], dtype=float).copy()
            total_distance = float(ray[4])

            p_hit = self._find_intersection([pol_angle, intensity, launch_position, direction, total_distance], t_min=t_min)
            if p_hit is None:
                continue

            segment = np.linalg.norm(p_hit - launch_position)
            hit_rays.append([pol_angle, intensity, p_hit, direction, total_distance + segment])

        return hit_rays


'''FTS-specific optical elements'''
# classes to create: wire grids (generate reflected & transmitted rays; change pol state & intensity), dihedral mirror

@dataclass
class WireGrid(_PoseMixin, _RayOpsMixin):
    pol_axis: float  # radians
    origin: tuple
    diam: float
    gx_local: tuple
    gy_local: tuple
    gz_local: tuple

    def _find_intersection(self, ray, t_min=0.01, debug=False):
        # Planar wire grid in local z=0 plane with circular aperture.
        ray_origin = np.asarray(ray[2], dtype=float)
        ray_dir = np.asarray(ray[3], dtype=float)
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            if debug:
                print("No intersection found: ray direction has zero magnitude")
            return None
        ray_dir /= norm

        ray_origin_local, ray_dir_local = self._to_local(ray_origin, ray_dir)
        denom = ray_dir_local[2]
        if np.isclose(denom, 0.0, atol=1e-12):
            if debug:
                print("No intersection found: ray is parallel to wire grid")
            return None

        t_hit = -ray_origin_local[2] / denom
        if t_hit < t_min:
            if debug:
                print("No intersection found: wire-grid hit is behind ray origin")
            return None

        p_local = ray_origin_local + t_hit * ray_dir_local
        if np.hypot(p_local[0], p_local[1]) > (self.diam / 2):
            if debug:
                print("No intersection found: wire-grid hit is outside aperture")
            return None

        return ray_origin + t_hit * ray_dir

    def plot(self, num_points=100, fig=None, opacity=0.5):
        x = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        y = np.linspace(-self.diam / 2, self.diam / 2, num_points)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.where(R <= self.diam / 2, 0.0, np.nan)

        R_l2g = self._rotation_matrix_local_to_global()
        pts = np.stack((X, Y, Z), axis=-1) @ R_l2g.T + self.origin
        Xg, Yg, Zg = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]

        if fig is None:
            fig = go.Figure()
        fig.add_trace(
            go.Surface(
                x=Xg,
                y=Yg,
                z=Zg,
                colorscale="Gray",
                showscale=False,
                opacity=opacity,
                name="WireGrid",
                showlegend=False,
            )
        )
        return fig, (Xg, Yg, Zg)

    def trace(self, starting_rays, n_scan=2000, debug=False, root_finder=None, return_ports=False):
        # For each hit ray, output two rays:
        # - reflected: pol_axis, I * cos^2(delta)
        # - transmitted: pol_axis + pi/2, I * sin^2(delta)
        # where delta = incident_pol - pol_axis.
        new_rays = []
        grid_pol = float(self.pol_axis)
        grid_pol_trans = np.mod(grid_pol + np.pi / 2.0, 2.0 * np.pi)

        for ray in starting_rays:
            inc_pol_angle = float(ray[0])
            inc_intensity = float(ray[1])
            launch_position = np.asarray(ray[2], dtype=float).copy()
            direction = np.asarray(ray[3], dtype=float).copy()
            direction_norm = np.linalg.norm(direction)
            total_distance = float(ray[4])

            if direction_norm == 0:
                # Preserve degenerate rays unchanged.
                degenerate_ray = [inc_pol_angle, inc_intensity, launch_position, direction, total_distance]
                if return_ports:
                    new_rays.append(("default", degenerate_ray))
                else:
                    new_rays.append(degenerate_ray)
                continue
            direction /= direction_norm

            ray_copy = [inc_pol_angle, inc_intensity, launch_position, direction, total_distance]
            p_hit = self._find_intersection(ray_copy, debug=debug)
            if p_hit is None:
                # If the ray misses the finite wire-grid aperture, drop it.
                continue

            segment = np.linalg.norm(p_hit - launch_position)
            new_total_distance = total_distance + segment

            delta = inc_pol_angle - grid_pol
            reflected_intensity = inc_intensity * (np.cos(delta) ** 2)
            transmitted_intensity = inc_intensity * (np.sin(delta) ** 2)

            normal_vec = self._to_global_dir(np.array([0.0, 0.0, 1.0]))
            reflected_dir = self._reflect_ray(direction, normal_vec)
            transmitted_dir = direction.copy()

            reflected_ray = [
                np.mod(grid_pol, 2.0 * np.pi),
                float(reflected_intensity),
                p_hit,
                reflected_dir,
                new_total_distance,
            ]
            transmitted_ray = [
                grid_pol_trans,
                float(transmitted_intensity),
                p_hit,
                transmitted_dir,
                new_total_distance,
            ]

            if return_ports:
                new_rays.append(("R", reflected_ray))
                new_rays.append(("T", transmitted_ray))
            else:
                new_rays.append(reflected_ray)
                new_rays.append(transmitted_ray)

        return new_rays

# not ready yet
@dataclass
class NSSystem:
    """
    Non-sequential optical subsystem.

    Each input ray is propagated through `components` by repeatedly:
    1) finding the nearest valid positive intersection among all components
    2) applying that component's interaction model
    3) continuing with any resulting output rays

    This allows mixed non-sequential systems (e.g., ellipsoids + wire grids)
    without hard-coding explicit path routing.
    """
    components: list
    t_min: float = 0.01
    max_interactions: int = 100

    def _ray_direction_unit(self, ray):
        direction = np.asarray(ray[3], dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        return direction / norm

    def _candidate_hit_point(self, element, ray, n_scan=500, debug=False, root_finder='newton'):
        # Ignore pass-through placeholders.
        if isinstance(element, Object):
            return None

        # Aspheric mirror has separate Newton/Brent intersection routines.
        if isinstance(element, AsphericMirror):
            method = _normalize_root_finder(root_finder, default='brent')
            if method == 'newton':
                guess = np.linalg.norm(np.asarray(element.origin, dtype=float) - np.asarray(ray[2], dtype=float))
                return element._find_intersection_newton(ray, guess=guess, debug=debug)
            return element._find_intersection_brentq(ray, n_scan=n_scan, debug=debug)

        # Two-surface lenses: nearest intersection is whichever surface is hit first.
        if isinstance(element, _BaseTwoSurfaceLens):
            launch_position = np.asarray(ray[2], dtype=float)
            direction = self._ray_direction_unit(ray)
            if direction is None:
                return None

            p0_local, d0_local = element._to_local(launch_position, direction)
            surf_a, surf_b = element._surface_names()
            t_a, p_a = element._find_surface_intersection_local(
                p0_local, d0_local, surf_a, t_min=self.t_min, n_scan=n_scan, root_finder=root_finder, debug=debug
            )
            t_b, p_b = element._find_surface_intersection_local(
                p0_local, d0_local, surf_b, t_min=self.t_min, n_scan=n_scan, root_finder=root_finder, debug=debug
            )
            candidates = []
            if t_a is not None:
                candidates.append((t_a, p_a))
            if t_b is not None:
                candidates.append((t_b, p_b))
            if len(candidates) == 0:
                return None
            t_hit, p_local = min(candidates, key=lambda item: item[0])
            if t_hit < self.t_min:
                return None
            return element._to_global_point(p_local)

        # Planar/other elements exposing _find_intersection.
        if hasattr(element, "_find_intersection"):
            try:
                return element._find_intersection(ray, t_min=self.t_min, debug=debug)
            except TypeError as err:
                if "unexpected keyword argument" not in str(err):
                    raise
            try:
                return element._find_intersection(ray, t_min=self.t_min)
            except TypeError as err:
                if "unexpected keyword argument" not in str(err):
                    raise
            return element._find_intersection(ray)

        return None

    def _nearest_hit(self, ray, n_scan=500, debug=False, root_finder='newton'):
        launch = np.asarray(ray[2], dtype=float)
        direction = self._ray_direction_unit(ray)
        if direction is None:
            return None

        best = None  # (t_hit, element, p_hit)
        for element in self.components:
            p_hit = self._candidate_hit_point(element, ray, n_scan=n_scan, debug=debug, root_finder=root_finder)
            if p_hit is None:
                continue
            p_hit = np.asarray(p_hit, dtype=float)
            segment = p_hit - launch
            t_hit = float(np.dot(segment, direction))
            if t_hit < self.t_min:
                continue
            if best is None or t_hit < best[0]:
                best = (t_hit, element, p_hit)
        return best

    def _interact_with_component(self, element, ray, p_hit, t_hit, n_scan=500, debug=False, root_finder='newton'):
        # Aperture in non-sequential mode should transmit to the hit point.
        if isinstance(element, Aperture):
            direction = self._ray_direction_unit(ray)
            if direction is None:
                return []
            total_distance = float(ray[4]) + float(np.linalg.norm(np.asarray(p_hit, dtype=float) - np.asarray(ray[2], dtype=float)))
            return [[float(ray[0]), float(ray[1]), np.asarray(p_hit, dtype=float), direction, total_distance]]

        # For all other elements, use native trace() on a single-ray bundle.
        return _call_element_trace(
            element,
            [ray],
            n_scan=n_scan,
            debug=debug,
            root_finder=root_finder,
            return_ports=False,
        )

    def _ray_made_progress(self, in_ray, out_ray, tol=1e-9):
        in_pos = np.asarray(in_ray[2], dtype=float)
        out_pos = np.asarray(out_ray[2], dtype=float)
        in_dist = float(in_ray[4])
        out_dist = float(out_ray[4])
        return (out_dist > in_dist + tol) or (np.linalg.norm(out_pos - in_pos) > tol)

    def trace(self, starting_rays, n_scan=500, debug=False, root_finder='newton'):
        active = deque((ray, 0) for ray in starting_rays)
        output_rays = []

        while active:
            ray, depth = active.popleft()
            if depth >= self.max_interactions:
                output_rays.append(ray)
                continue

            hit = self._nearest_hit(ray, n_scan=n_scan, debug=debug, root_finder=root_finder)
            if hit is None:
                output_rays.append(ray)
                continue

            t_hit, element, p_hit = hit
            next_rays = self._interact_with_component(
                element, ray, p_hit, t_hit, n_scan=n_scan, debug=debug, root_finder=root_finder
            )

            if next_rays is None or len(next_rays) == 0:
                # e.g., clipped by a finite element
                continue

            for next_ray in next_rays:
                if self._ray_made_progress(ray, next_ray):
                    active.append((next_ray, depth + 1))
                else:
                    # Stagnation guard to avoid infinite loops.
                    output_rays.append(next_ray)

        return output_rays

    def plot(self, num_points=100, fig=None, opacity=0.8):
        if fig is None:
            fig = go.Figure()
        grids = []
        seen = set()
        for element in self.components:
            obj_id = id(element)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            fig, grid = element.plot(num_points=num_points, fig=fig, opacity=opacity)
            grids.append(grid)
        return fig, tuple(grids)
    
'''Helper functions for optical branching mode'''

def _call_element_trace(element, rays, n_scan=500, debug=False, root_finder='newton', return_ports=False):
    """Call element.trace with graceful fallback across optional kwargs."""
    candidates = []
    if return_ports:
        candidates.extend([
            {"n_scan": n_scan, "debug": debug, "root_finder": root_finder, "return_ports": True},
            {"n_scan": n_scan, "debug": debug, "return_ports": True},
            {"return_ports": True},
        ])
    candidates.extend([
        {"n_scan": n_scan, "debug": debug, "root_finder": root_finder},
        {"n_scan": n_scan, "debug": debug},
        {},
    ])

    last_exc = None
    for kwargs in candidates:
        try:
            return element.trace(rays, **kwargs)
        except TypeError as err:
            last_exc = err
            if "unexpected keyword argument" in str(err):
                continue
            raise

    if last_exc is not None:
        raise last_exc
    return element.trace(rays)


def _normalize_ported_rays(trace_result):
    """Normalize trace output to a list of (port, ray)."""
    if trace_result is None:
        return []
    if len(trace_result) == 0:
        return []

    def _is_port_item(item):
        return isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)

    if all(_is_port_item(item) for item in trace_result):
        return [(item[0], item[1]) for item in trace_result]
    return [("default", ray) for ray in trace_result]


def _append_path_point(path_points: Optional[list[npt.NDArray]], ray) -> Optional[list[npt.NDArray]]:
    if path_points is None:
        return None
    point = np.asarray(ray[2], dtype=float)
    new_points = list(path_points)
    if len(new_points) == 0 or np.linalg.norm(point - new_points[-1]) > 1e-9:
        new_points.append(point.copy())
    return new_points


def _build_path_color_map(path_ids, base_colors=None, color_sequence=None, default_color='red'):
    """Build deterministic color assignments for path IDs."""
    default_sequence = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    palette = color_sequence if color_sequence is not None else default_sequence
    color_map = dict(base_colors) if isinstance(base_colors, dict) else {}

    ordered_ids = []
    seen = set()
    for pid in path_ids:
        if pid in seen or pid in (None, ""):
            continue
        seen.add(pid)
        ordered_ids.append(pid)

    palette_idx = 0
    for pid in ordered_ids:
        if pid in color_map:
            continue
        color_map[pid] = palette[palette_idx % len(palette)]
        palette_idx += 1

    # Fallback color for empty/unknown path labels.
    color_map.setdefault("", default_color)
    color_map.setdefault(None, default_color)
    return color_map


def _plot_paths(fig, ray_paths: Sequence[Optional[list[npt.NDArray]]], ray_color='red', ray_alpha=1.0, ray_width=2, path_ids=None, path_colors=None):
    seen_path_legend = set()
    for i, path_points in enumerate(ray_paths):
        if path_points is None or len(path_points) == 0:
            continue
        path_id = None if path_ids is None else path_ids[i]
        this_color = ray_color if path_colors is None else path_colors.get(path_id, path_colors.get("", ray_color))
        if path_ids is not None and path_id not in (None, ""):
            trace_name = f"Path {path_id}"
            # Only show legend for the first occurrence of each unique path ID
            show_legend = trace_name not in seen_path_legend
            if show_legend:
                seen_path_legend.add(trace_name)
        else:
            trace_name = f'Ray {i}' if i == 0 else None
            show_legend = (i == 0)

        pts = np.array(path_points)
        if pts.shape[0] == 1:
            fig.add_trace(
                go.Scatter3d(
                    x=[pts[0, 0]],
                    y=[pts[0, 1]],
                    z=[pts[0, 2]],
                    mode='markers',
                    marker=dict(size=3, color=this_color),
                    name=trace_name,
                    showlegend=show_legend,
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode='lines',
                    line=dict(color=this_color, width=ray_width),
                    opacity=ray_alpha,
                    name=trace_name,
                    showlegend=show_legend,
                )
            )


def _iter_graph_elements(optical_graph):
    nodes = optical_graph.get("nodes", {})
    for node_spec in nodes.values():
        element_spec = node_spec.get("element") if isinstance(node_spec, dict) else node_spec
        if element_spec is None:
            continue
        if isinstance(element_spec, (list, tuple)):
            for element in element_spec:
                yield element
        else:
            yield element_spec


def _normalize_branch_port_token(token):
    """Canonicalize branch token to 'node:port' (no extra spaces)."""
    text = str(token).strip()
    if ":" not in text:
        raise ValueError(
            f"Invalid branch token '{token}'. Expected format 'node_name:port'."
        )
    node_name, port = text.split(":", 1)
    node_name = node_name.strip()
    port = port.strip()
    if node_name == "" or port == "":
        raise ValueError(
            f"Invalid branch token '{token}'. Expected format 'node_name:port'."
        )
    return f"{node_name}:{port}"


def _build_branch_sequence_routing(allowed_branch_sequences):
    """
    Build branch-history routing lookup from allowed_branch_sequences.

    Expected sequence format (alternating branch token and destination node):
      ("wg_1:T", "ell_1", "wg_2:R", "ell_3")
    which defines:
      ("wg_1:T",) -> "ell_1"
      ("wg_1:T", "wg_2:R") -> "ell_3"
    """
    if allowed_branch_sequences is None:
        return None

    routing = {}
    for seq in allowed_branch_sequences:
        if isinstance(seq, str):
            parts = [part.strip() for part in seq.split(";") if part.strip()]
        elif isinstance(seq, (list, tuple)):
            parts = [str(part).strip() for part in seq if str(part).strip()]
        else:
            raise TypeError(
                "Each branch sequence must be a string or a list/tuple."
            )

        if len(parts) == 0:
            continue
        if len(parts) % 2 != 0:
            raise ValueError(
                "Each branch sequence must alternate ('node:port', 'next_node'). "
                f"Received odd-length sequence: {parts}"
            )

        branch_history = []
        for i in range(0, len(parts), 2):
            branch_token = _normalize_branch_port_token(parts[i])
            next_node = parts[i + 1].strip()
            if next_node == "":
                raise ValueError(
                    f"Invalid empty destination node in branch sequence: {parts}"
                )
            branch_history.append(branch_token)
            key = tuple(branch_history)
            if key in routing and routing[key] != next_node:
                raise ValueError(
                    f"Conflicting branch routing for {key}: "
                    f"'{routing[key]}' vs '{next_node}'."
                )
            routing[key] = next_node

    return routing


'''Key function to trace rays through an optical system and optionally plot the results.

Supports:
1) Sequential mode (existing behavior): optical_elements is a list/tuple of elements.
2) Graph mode: optical_elements is a dict with keys:
   - "entry": name of first node
   - "nodes": {node_name: {"element": elem_or_list, "next": {"default"/"R"/"T": next_node_or_None}}}
   - optional "allowed_branch_sequences": list of alternating ('node:port', 'next_node')
     pairs used to route branch ports with history-aware behavior
'''
def trace_optical_system(
    optical_elements,
    starting_rays,
    plot=True,
    fig=None,
    extend_past_last=0.0,
    ray_color='red',
    ray_alpha=1.0,
    ray_width=2,
    color_paths_by_id=False,
    path_colors=None,
    path_color_sequence=None,
    optics_alpha=0.8,
    n_scan=500,
    debug=False,
    return_all=False,
    show=True,
    root_finder='newton'
):
    # -----------------------------
    # Sequential mode (backward compatible)
    # -----------------------------
    if not (isinstance(optical_elements, dict) and "nodes" in optical_elements):
        rays = starting_rays
        track_ray_paths = plot or extend_past_last > 0
        ray_paths = [[np.asarray(rays[i][2], dtype=float).copy()] for i in range(len(rays))] if track_ray_paths else None
        output_rays = [rays] if return_all else None

        for element in optical_elements:
            if track_ray_paths:
                assert ray_paths is not None  # Type narrowing: guaranteed by track_ray_paths logic
                new_rays = []
                new_ray_paths = []
                for in_ray, in_path in zip(rays, ray_paths):
                    out_rays = _call_element_trace(
                        element, [in_ray], n_scan=n_scan, debug=debug, root_finder=root_finder, return_ports=False
                    )
                    for out_ray in out_rays:
                        new_rays.append(out_ray)
                        new_ray_paths.append(_append_path_point(in_path, out_ray))
                rays = new_rays
                ray_paths = new_ray_paths
            else:
                rays = _call_element_trace(
                    element, rays, n_scan=n_scan, debug=debug, root_finder=root_finder, return_ports=False
                )
            if return_all:
                assert output_rays is not None  # Type narrowing: guaranteed by return_all logic
                output_rays.append(rays)

        if track_ray_paths and extend_past_last > 0:
            assert ray_paths is not None  # Type narrowing: guaranteed by track_ray_paths logic
            for i, ray in enumerate(rays):
                if ray_paths[i] is not None:  # Only extend non-None paths
                    tail_point = np.asarray(ray[2], dtype=float) + extend_past_last * np.asarray(ray[3], dtype=float)
                    ray_paths[i].append(tail_point)

        if plot:
            assert ray_paths is not None  # Type narrowing: guaranteed by plot implies track_ray_paths
            if fig is None:
                fig = go.Figure()
            for element in optical_elements:
                fig, _ = element.plot(fig=fig, opacity=optics_alpha)
            _plot_paths(fig, ray_paths, ray_color=ray_color, ray_alpha=ray_alpha, ray_width=ray_width)
            fig.update_layout(scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'))
            if show:
                fig.show(renderer='notebook_connected')

        return output_rays if return_all else rays

    # -----------------------------
    # Graph mode
    # -----------------------------
    nodes = optical_elements["nodes"]
    entry = optical_elements.get("entry")
    if entry is None:
        entry = next(iter(nodes), None)
        return {"final_rays": [], "rays_by_path": {}} if return_all else []

    branch_sequence_routing = _build_branch_sequence_routing(
        optical_elements.get("allowed_branch_sequences")
    )

    track_ray_paths = plot or extend_past_last > 0
    active = deque()
    for ray in starting_rays:
        initial_points = [np.asarray(ray[2], dtype=float).copy()] if track_ray_paths else None
        active.append((entry, ray, initial_points, "", tuple()))

    terminal_rays = []
    terminal_paths = []
    terminal_path_ids = []
    path_color_lookup = None

    while active:
        node_name, in_ray, in_path_points, in_path_id, in_branch_tokens = active.popleft()
        if node_name not in nodes:
            raise KeyError(f"Node '{node_name}' not found in optical graph.")

        node_spec = nodes[node_name]
        if isinstance(node_spec, dict):
            element_spec = node_spec.get("element")
            next_map = node_spec.get("next", {})
        else:
            element_spec = node_spec
            next_map = {}

        if element_spec is None:
            terminal_rays.append(in_ray)
            terminal_paths.append(in_path_points)
            terminal_path_ids.append(in_path_id)
            continue

        # state tuples: (port, ray, path_points)
        states = [("default", in_ray, in_path_points)]
        elements = element_spec if isinstance(element_spec, (list, tuple)) else [element_spec]
        is_block = isinstance(element_spec, (list, tuple))

        for element in elements:
            next_states = []
            for _, state_ray, state_points in states:
                trace_result = _call_element_trace(
                    element,
                    [state_ray],
                    n_scan=n_scan,
                    debug=debug,
                    root_finder=root_finder,
                    return_ports=(not is_block),
                )
                ported = _normalize_ported_rays(trace_result)
                for port, out_ray in ported:
                    # Inside sequential sub-blocks, ignore branch ports and keep default flow.
                    if is_block:
                        port = "default"
                    
                    # OPTIMIZATION: Early filtering for branch ports when allowed_branch_sequences exists.
                    # Check if this branch would be allowed before doing further processing.
                    if not is_block and port != "default" and branch_sequence_routing is not None:
                        branch_token = _normalize_branch_port_token(f"{node_name}:{port}")
                        if (in_branch_tokens + (branch_token,)) not in branch_sequence_routing:
                            continue  # Skip disallowed branches immediately
                    
                    out_points = _append_path_point(state_points, out_ray)
                    next_states.append((port, out_ray, out_points))
            states = next_states
            if len(states) == 0:
                break

        for port, out_ray, out_points in states:
            out_path_id = in_path_id
            out_branch_tokens = in_branch_tokens
            next_node = None
            
            if port != "default":
                branch_token = _normalize_branch_port_token(f"{node_name}:{port}")
                out_branch_tokens = in_branch_tokens + (branch_token,)
                out_path_id = f"{in_path_id}{node_name}:{port} "
                
                if branch_sequence_routing is not None:
                    # With early filtering above, this lookup should always succeed for branching nodes.
                    # For element blocks (is_block=True), ports are set to "default", so we don't reach here.
                    next_node = branch_sequence_routing.get(out_branch_tokens, None)
                else:
                    # No allowed_branch_sequences specified, use node-local routing.
                    next_node = next_map.get(port, next_map.get("default", None))
            else:
                next_node = next_map.get("default", None)

            if next_node is None:
                # Ray reached end of system (e.g., iris with "default": None).
                # Note: Disallowed branches are filtered early (above), so we shouldn't see them here.
                if out_points is not None and extend_past_last > 0:
                    tail_point = np.asarray(out_ray[2], dtype=float) + extend_past_last * np.asarray(out_ray[3], dtype=float)
                    out_points = list(out_points)
                    out_points.append(tail_point)
                terminal_rays.append(out_ray)
                terminal_paths.append(out_points)
                terminal_path_ids.append(out_path_id)
            else:
                active.append((next_node, out_ray, out_points, out_path_id, out_branch_tokens))

    if plot:
        if fig is None:
            fig = go.Figure()
        seen = set()
        for element in _iter_graph_elements(optical_elements):
            obj_id = id(element)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            fig, _ = element.plot(fig=fig, opacity=optics_alpha)
        if color_paths_by_id:
            path_color_lookup = _build_path_color_map(
                terminal_path_ids,
                base_colors=path_colors,
                color_sequence=path_color_sequence,
                default_color=ray_color,
            )
        _plot_paths(
            fig,
            terminal_paths,
            ray_color=ray_color,
            ray_alpha=ray_alpha,
            ray_width=ray_width,
            path_ids=(terminal_path_ids if color_paths_by_id else None),
            path_colors=path_color_lookup,
        )
        fig.update_layout(scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'))
        if show:
            fig.show(renderer='notebook_connected')

    if return_all:
        rays_by_path = {}
        for path_id, ray in zip(terminal_path_ids, terminal_rays):
            rays_by_path.setdefault(path_id, []).append(ray)
        return {
            "final_rays": terminal_rays,
            "rays_by_path": rays_by_path,
            "path_colors": path_color_lookup,
        }
    return terminal_rays

# function to fit a focal plane to a bundle of rays
def fit_focal_plane(rays, diam, bounds):

    # first: choose a direction to move in
    rays = np.asarray(rays, dtype=object) # for easier slicing
    mean_dir = np.mean(rays[:,3], axis=0) # mean ray direction

    # ensure mean_dir is a unit vector
    norm = np.linalg.norm(mean_dir)
    mean_dir /= norm

    # construct orthonormal basis with mean_dir as z-axis
    helper = np.array([0, 1, 0]) if abs(mean_dir[1]) < 0.9 else np.array([1, 0, 0])
    gx = np.cross(mean_dir, helper)
    gx /= np.linalg.norm(gx)
    gy = np.cross(mean_dir, gx)
    gy /= np.linalg.norm(gy)
    gx_local, gy_local, gz_local = np.vstack((gx, gy, mean_dir)).T # to match the FocalPlane convention

    # create an initial evaluation plane at the mean ray origin, oriented perpendicular to mean_dir
    mean_origin = np.mean(rays[:,2], axis=0)
    eval_plane = FocalPlane(origin = mean_origin, diam=diam, gx_local=gx_local, gy_local=gy_local, gz_local=gz_local)

    # function to evaluate spot size for a given plane offset along mean_dir
    def spot_size(offset):
        new_origin = mean_origin + offset * mean_dir
        eval_plane.origin = new_origin # update evaluation plane origin
        traced = np.array(eval_plane.trace(rays, t_min = -500), dtype=object) # trace rays to plane
        local_hits = np.asarray([eval_plane._to_local_point(pt)[:2] for pt in traced[:,2]], dtype=float)
        w = traced[:,1].astype(float) # weight by intensities

        # compute weighted RMS spot size
        centroid = np.average(local_hits, axis=0, weights=w)
        rms = np.sqrt(np.average(np.sum((local_hits - centroid)**2, axis=1), weights=w))
        return rms
    
    # scan along mean_dir to find offset minimizing spot size (use optimize.minimize_scalar)
    res = optimize.minimize_scalar(spot_size, bounds=(bounds[0], bounds[-1]), method='bounded')
    offset = res.x

    # update eval_plane to best focus position
    eval_plane.origin = mean_origin + offset * mean_dir

    return eval_plane