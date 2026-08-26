import ray_tracing as rt
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from tqdm import tqdm

c = 2.998e11 # speed of light in mm/s

@dataclass
class Detector(rt._PoseMixin):
	origin: tuple
	diam: float
	gx_local: tuple
	gy_local: tuple
	gz_local: tuple
	resolution: float
	# aperture: rt.FocalPlane | rt.Aperture

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
				colorscale="Purples",
				showscale=False,
				opacity=opacity,
				name="Detector",
				showlegend=False,
			)
		)
		return fig, (Xg, Yg, Zg)
	
	def _generate_grid_points(self):
		# returns a grid of points in global coordinates, spaced by resolution
		x = np.arange(-self.diam / 2, self.diam / 2 + self.resolution, self.resolution)
		y = np.arange(-self.diam / 2, self.diam / 2 + self.resolution, self.resolution)
		X, Y = np.meshgrid(x, y)
		Z = np.zeros_like(X)
		R_l2g = self._rotation_matrix_local_to_global()
		pts = np.stack((X, Y, Z), axis=-1) @ R_l2g.T + self.origin
		Xg, Yg, Zg = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
		return Xg, Yg, Zg
	
	def intensity_map(self, input_rays, freqs, weights=None, plot=False, fig=None, method = 'huygens', ray_chunk_size=32, return_field=False):
		''' propagates rays at the aperture plane to a 2D intensity pattern at the detector plane, using the specified method. For propagating complex fields rather than rays to the detector instance, use intensity_map_from_field.'''

		# create array for intensity values at each grid point
		x_local = np.arange(-self.diam / 2, self.diam / 2 + self.resolution, self.resolution)
		y_local = np.arange(-self.diam / 2, self.diam / 2 + self.resolution, self.resolution)
		Xg, Yg, Zg = self._generate_grid_points()
		intensities = np.zeros_like(Xg)

		# Extract ray data once — independent of grid position
		input_rays_arr = np.asarray(input_rays, dtype=object)
		starting_pts = np.stack(input_rays_arr[:, 2].tolist()).astype(np.float64)  # (N_rays, 3)
		ray_dirs     = np.stack(input_rays_arr[:, 3].tolist()).astype(np.float64)  # (N_rays, 3)
		ray_dists    = input_rays_arr[:, 4].astype(np.float64)                     # (N_rays,)
		thetas       = input_rays_arr[:, 0].astype(np.float64)                     # (N_rays,)
		amps         = np.sqrt(input_rays_arr[:, 1].astype(np.float64))            # (N_rays,)
		ex0 = amps * np.cos(thetas)  # (N_rays,)
		ey0 = amps * np.sin(thetas)  # (N_rays,)

		# Accumulate field sums in chunks to cap peak memory at O(N_freqs * Ny * Nx * ray_chunk_size).
		# Distances are frequency-independent, so dist_c is computed once per chunk rather than
		# once per (chunk × freq), saving ~40% of runtime for broadband inputs.
		grid_pts = np.stack([Xg, Yg, Zg], axis=-1)  # (Ny, Nx, 3)
		n_rays = len(starting_pts)
		n_freqs = len(freqs)
		wavelengths = c / np.asarray(freqs)          # (N_freqs,)

		Ex_sums = np.zeros((n_freqs,) + Xg.shape, dtype=complex)  # (N_freqs, Ny, Nx)
		Ey_sums = np.zeros((n_freqs,) + Xg.shape, dtype=complex)

		for start in range(0, n_rays, ray_chunk_size):
			end = start + ray_chunk_size
			sp_c   = starting_pts[start:end]          # (C, 3)
			rd_c   = ray_dists[start:end]             # (C,)
			rdir_c = ray_dirs[start:end]              # (C, 3)
			ex_c   = ex0[start:end]                   # (C,)
			ey_c   = ey0[start:end]                   # (C,)
			diff_vecs = grid_pts[:, :, None, :] - sp_c[None, None, :, :]  # (Ny, Nx, C, 3)
			dist_c = np.linalg.norm(diff_vecs, axis=-1)                    # (Ny, Nx, C)
			cos_chi = np.sum(diff_vecs * rdir_c[None, None, :, :], axis=-1) / dist_c  # (Ny, Nx, C)
			obliquity = (1.0 + cos_chi) * 0.5

			# include Kirchoff obliquity factor & i/lambda scaling
			inv_dist_ex = ex_c[None, None, :] * obliquity / dist_c  # (Ny, Nx, C)
			inv_dist_ey = ey_c[None, None, :] * obliquity / dist_c
			opl_c = dist_c + rd_c[None, None, :]     # (Ny, Nx, C)
			for i_freq in range(n_freqs):
				k_factor = -1j / wavelengths[i_freq]
				phases_c = np.exp(1j * 2 * np.pi * opl_c / wavelengths[i_freq])
				Ex_sums[i_freq] += np.nansum(k_factor * inv_dist_ex * phases_c, axis=-1)
				Ey_sums[i_freq] += np.nansum(k_factor * inv_dist_ey * phases_c, axis=-1)

		w = np.asarray(weights) if weights is not None else np.ones(n_freqs)
		intensities = np.sum(w[:, None, None] * (np.abs(Ex_sums)**2 + np.abs(Ey_sums)**2), axis=0)

		outside_aperture = self.aperture_mask()
		# set intensities outside the source aperture to NaN
		intensities[outside_aperture] = np.nan
		if return_field:
			# zero-fill (not NaN) outside the aperture so the field is safe to feed into an FFT
			Ex_sums[:, outside_aperture] = 0.0
			Ey_sums[:, outside_aperture] = 0.0

		if (plot):
			if fig is None:
				fig = go.Figure()

			# Plot intensity in detector-local coordinates.
			fig.add_trace(
				go.Heatmap(
					x=x_local,
					y=y_local,
					z=intensities,
					colorscale="Viridis",
					colorbar=dict(title="Intensity"),
					name="Detector Intensity",
					showscale=True,
				)
			)
			fig.update_layout(
				title='Intensity Map at Detector (Local Frame)',
				autosize=True,
				xaxis_title='Detector local x (mm)',
				yaxis_title='Detector local y (mm)',
				yaxis=dict(scaleanchor='x', scaleratio=1),
			)
			fig.show(renderer='notebook_connected')

		if return_field:
			return Ex_sums, Ey_sums, x_local, y_local
		return intensities, x_local, y_local

	def grid_axes(self):
		# local (x, y) coordinate axes of this detector's sampling grid, matching the grid built in intensity_map/_generate_grid_points
		x_local = np.arange(-self.diam / 2, self.diam / 2 + self.resolution, self.resolution)
		y_local = np.arange(-self.diam / 2, self.diam / 2 + self.resolution, self.resolution)
		return x_local, y_local

	def aperture_mask(self):
		'''Boolean array, matching this detector's grid_axes(), True outside the circular aperture.'''
		x_local, y_local = self.grid_axes()
		Xl, Yl = np.meshgrid(x_local, y_local)
		return np.hypot(Xl, Yl) > self.diam / 2

	def _propagate_angular_spectrum(self, field, x_local, y_local, wavelength, z, pad_factor=2, out_shape=None):
		'''Propagate a complex scalar field a distance z via the angular-spectrum method. Requires a complex field as input.

			field: complex array, shape (..., Ny, Nx).
			x_local, y_local: 1D coordinate arrays (mm) of the field's sampling grid at the reference detector. Must be uniformly spaced and square (dx == dy).
			wavelength: scalar or array broadcastable against field's leading axes (mm).
			z: propagation distance (mm), positive in the same forward sense as the
				ray-traced field that produced `field`.
			pad_factor: zero-padding factor (>=2 recommended) applied before transforming,
				to suppress circular-convolution wraparound between opposite edges of the
				grid (the free-space kernel is unbounded; the field array is not).
			out_shape: (Ny_out, Nx_out) to crop the result to. Defaults to field's own
				(Ny, Nx). Lets the output land on a differently-sized grid (e.g. a wider
				source-plane aperture) than the input, since the convolution itself keeps
				dx fixed on both ends rather than tying output pixel size to input size
				the way a single-FFT Fresnel transform would.

		Returns a complex array of shape (..., Ny_out, Nx_out).'''

		field = np.asarray(field, dtype=complex)
		*lead, Ny, Nx = field.shape

		x_local = np.asarray(x_local, dtype=float)
		y_local = np.asarray(y_local, dtype=float)
		assert len(x_local) == Nx and len(y_local) == Ny, \
			"x_local/y_local length must match field's (Ny, Nx) grid shape"
		dx_steps = np.diff(x_local)
		dy_steps = np.diff(y_local)
		assert np.allclose(dx_steps, dx_steps[0]), "x_local must be uniformly spaced"
		assert np.allclose(dy_steps, dy_steps[0]), "y_local must be uniformly spaced"
		dx, dy = dx_steps[0], dy_steps[0]
		assert np.isclose(dx, dy), "_propagate_angular_spectrum requires square pixels (dx == dy)"

		Ny_out, Nx_out = out_shape if out_shape is not None else (Ny, Nx)

		Ny_grid = max(Ny, Ny_out)
		Nx_grid = max(Nx, Nx_out)
		Ny_pad = int(np.ceil(Ny_grid * pad_factor))
		Nx_pad = int(np.ceil(Nx_grid * pad_factor))

		padded = np.zeros((*lead, Ny_pad, Nx_pad), dtype=complex)
		y0 = (Ny_pad - Ny) // 2
		x0 = (Nx_pad - Nx) // 2
		padded[..., y0:y0 + Ny, x0:x0 + Nx] = field

		kx = 2 * np.pi * np.fft.fftfreq(Nx_pad, d=dx)
		ky = 2 * np.pi * np.fft.fftfreq(Ny_pad, d=dx)
		Kx, Ky = np.meshgrid(kx, ky)
	
		k = 2 * np.pi / np.asarray(wavelength, dtype=float)
		k = k[..., np.newaxis, np.newaxis]  # broadcast against (Ny_pad, Nx_pad)
	
		# Drop evanescent wave components (kz imaginary)
		kz_sq = k**2 - Kx**2 - Ky**2
		propagating = kz_sq >= 0
		kz = np.sqrt(np.where(propagating, kz_sq, 0.0))
		H = np.where(propagating, np.exp(1j * kz * z), 0.0)
	
		propagated = np.fft.ifft2(np.fft.fft2(padded, axes=(-2, -1)) * H, axes=(-2, -1))
	
		yc0 = (Ny_pad - Ny_out) // 2
		xc0 = (Nx_pad - Nx_out) // 2
		return propagated[..., yc0:yc0 + Ny_out, xc0:xc0 + Nx_out]

	def intensity_map_from_field(self, Ex, Ey, ref_detector, freqs, weights=None, method = 'angular_spectrum', pad_factor=2, return_field=False):
		'''Produces an intensity map at the instance by propagating the full complex field (Ex, Ey) from a reference detector using the specified method. The field must be defined on the local grid axes of the reference detector, with shape (N_freqs, Ny, Nx). Ref detector and this detector must be parallel planes.'''

		assert isinstance(ref_detector, Detector), "ref_detector must be a Detector instance"

		# retrieve x_local and y_local from ref_detector
		x_local, y_local = ref_detector.grid_axes()

		# check that ref detector and this detector are parallel planes
		assert np.allclose(self.gx_local, ref_detector.gx_local) and np.allclose(self.gy_local, ref_detector.gy_local)

		# choose output shape to match this detector's size
		x_local_self, y_local_self = self.grid_axes()
		assert np.isclose(self.resolution, ref_detector.resolution), \
			"angular-spectrum propagation keeps grid spacing fixed; this detector and ref_detector must share the same resolution"
		out_shape = (len(y_local_self), len(x_local_self))

		# find propagation distance from ref detector to this detector
		z = ref_detector._to_local_point(self.origin)[2]

		# propagate angular spectrum for Ex and Ey separately, then combine to get intensity
		wavelengths = c / np.asarray(freqs)

		if method == 'angular_spectrum':
			Ex_prop = self._propagate_angular_spectrum(Ex, x_local, y_local, wavelengths, z, pad_factor=pad_factor, out_shape=out_shape)
			Ey_prop = self._propagate_angular_spectrum(Ey, x_local, y_local, wavelengths, z, pad_factor=pad_factor, out_shape=out_shape)

		w = np.asarray(weights) if weights is not None else np.ones(len(freqs))
		intensities = np.sum(w[:, None, None] * (np.abs(Ex_prop)**2 + np.abs(Ey_prop)**2), axis=0)

		# return intensity map, and optionally the propagated Ex and Ey fields if return_field is True
		if return_field:
			return Ex_prop, Ey_prop, x_local_self, y_local_self
		return intensities, x_local_self, y_local_self