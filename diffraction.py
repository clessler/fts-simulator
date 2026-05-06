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
	
	def intensity_map(self, input_rays, freqs, weights=None, plot=False, fig=None, method = 'huygens', ray_chunk_size=32):
		# propagates rays at the aperture plane to a 2D intensity pattern at the detector plane, using the specified method

		# create array for intensity values at each grid point
		x_local = np.arange(-self.diam / 2, self.diam / 2 + self.resolution, self.resolution)
		y_local = np.arange(-self.diam / 2, self.diam / 2 + self.resolution, self.resolution)
		Xl, Yl = np.meshgrid(x_local, y_local)
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

			# include Kirchoff obliquity factor
			inv_dist_ex = ex_c[None, None, :] * obliquity / dist_c  # (Ny, Nx, C)
			inv_dist_ey = ey_c[None, None, :] * obliquity / dist_c
			opl_c = dist_c + rd_c[None, None, :]     # (Ny, Nx, C)
			for i_freq in range(n_freqs):
				phases_c = np.exp(1j * 2 * np.pi * opl_c / wavelengths[i_freq])
				Ex_sums[i_freq] += np.nansum(inv_dist_ex * phases_c, axis=-1)
				Ey_sums[i_freq] += np.nansum(inv_dist_ey * phases_c, axis=-1)

		w = np.asarray(weights) if weights is not None else np.ones(n_freqs)
		intensities = np.sum(w[:, None, None] * (np.abs(Ex_sums)**2 + np.abs(Ey_sums)**2), axis=0)
		intensities[np.hypot(Xl, Yl) > self.diam / 2] = np.nan
				
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

		return intensities, x_local, y_local