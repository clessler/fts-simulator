import ray_tracing as rt
import numpy as np
import so_coupling_optics_TR_geometry as geo
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
	
	def intensity_map(self, input_rays, freqs, plot=False, fig=None, method = 'huygens'):
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
		ray_dists    = input_rays_arr[:, 4].astype(np.float64)                     # (N_rays,)
		thetas       = input_rays_arr[:, 0].astype(np.float64)                     # (N_rays,)
		amps         = np.sqrt(input_rays_arr[:, 1].astype(np.float64))            # (N_rays,)
		ex0 = amps * np.cos(thetas)  # (N_rays,)
		ey0 = amps * np.sin(thetas)  # (N_rays,)

		# Distances from every ray to every grid point: (Ny, Nx, N_rays)
		grid_pts  = np.stack([Xg, Yg, Zg], axis=-1)  # (Ny, Nx, 3)
		distances = np.linalg.norm(
			grid_pts[:, :, None, :] - starting_pts[None, None, :, :], axis=-1
		)
		total_distances = distances + ray_dists[None, None, :]  # (Ny, Nx, N_rays)

		for freq in freqs:
			wavelength = c / freq
			phases = np.exp(1j * 2 * np.pi * total_distances / wavelength)  # (Ny, Nx, N_rays)
			ex = (ex0[None, None, :] / distances) * phases
			ey = (ey0[None, None, :] / distances) * phases
			intensities += (np.abs(np.nansum(ex, axis=-1))**2
			              + np.abs(np.nansum(ey, axis=-1))**2)
				
		if (plot):
			if fig is None:
				fig = go.Figure()

			# Plot intensity in detector-local coordinates.
			intensities_plot = np.array(intensities, copy=True)
			outside_aperture = np.hypot(Xl, Yl) > (self.diam / 2)
			intensities_plot[outside_aperture] = np.nan

			fig.add_trace(
				go.Heatmap(
					x=x_local,
					y=y_local,
					z=intensities_plot,
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