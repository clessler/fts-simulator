'''
2D interpolation code using Gaussian-windowed, locally-weighted linear regression.
For each point, weight the surrounding data by a normalized Gaussian centered on the point,
fit a polynomial to the weighted data, and store the value of that polynomial fit at the interpolation point.

Optional peak-tracking mode: before interpolating, align peaks in y across rows (x-slices)
using Gaussian fits, interpolate the aligned surface, then restore peak positions afterward.
This prevents smearing when peaks shift systematically with x.
'''
import numpy as np
from scipy.interpolate import RectBivariateSpline, CubicSpline


def _parabolic_peak(y, row, i_peak, half_width=2):
    '''
    Sub-grid peak position via local quadratic fit around i_peak.
    More robust than a global Gaussian fit for non-Gaussian peak shapes.
    Returns the y position of the fitted peak, or y[i_peak] if the fit is degenerate.
    '''
    i_lo = max(0, i_peak - half_width)
    i_hi = min(len(row) - 1, i_peak + half_width)
    y_local = y[i_lo:i_hi + 1]
    z_local = row[i_lo:i_hi + 1]
    coeffs = np.polyfit(y_local, z_local, 2)   # a*y^2 + b*y + c
    if coeffs[0] < 0:                           # concave — genuine peak
        y_fit = -coeffs[1] / (2 * coeffs[0])
        if y[0] <= y_fit <= y[-1]:
            return y_fit
    return y[i_peak]


def align_peaks(x, y, z, method='nonparametric'):
    '''
    Align peaks in z along the x axis using a local quadratic fit near the argmax
    of each row, shifting so that all peaks land at a common reference position (median).

    Parameters:
    - x: 1D array of x coordinates, length N_x
    - y: 1D array of y coordinates, length N_y
    - z: 2D array of shape (N_x, N_y)

    Returns:
    - z_aligned: 2D array of shape (N_x, N_y) with peaks shifted to y_ref
    - peak_shifts: 1D array of length N_x; shift[i] = y_peak[i] - y_ref
                   (positive means the original peak was to the right of the reference)
    - y_peaks: 1D array of length N_x — fitted peak positions before alignment
    - y_ref: scalar — the common reference position peaks are aligned to
    '''
    N_x = len(x)
    y_peaks = np.zeros(N_x)

    for i in range(N_x):
        row = z[i, :]
        i_peak = np.argmax(row)
        y_peaks[i] = _parabolic_peak(y, row, i_peak)

    y_ref = np.median(y_peaks)
    peak_shifts = y_peaks - y_ref  # delta_i: positive = peak was right of reference

    # Shift each row left by delta_i: sample z[i] at y + delta_i
    # CubicSpline reduces the peak-underestimation error vs linear np.interp
    z_aligned = np.zeros_like(z)

    if method == 'nonparametric':
        for i in range(N_x):
            cs = CubicSpline(y, z[i, :], extrapolate=False)
            vals = cs(y + peak_shifts[i])
            z_aligned[i, :] = np.where(np.isnan(vals), 0.0, vals)
        return z_aligned, peak_shifts, y_peaks, y_ref, None

    if method == 'linear':
        linear_coeffs = np.polyfit(x, peak_shifts, 1)
        for i in range(N_x):
            delta_linear = np.polyval(linear_coeffs, x[i]) # predict shift from x using linear fit
            # shift/interpolate to z_aligned using CubicSpline
            cs = CubicSpline(y, z[i, :], extrapolate=False)
            vals = cs(y + delta_linear)
            z_aligned[i, :] = np.where(np.isnan(vals), 0.0, vals)
        return z_aligned, peak_shifts, y_peaks, y_ref, linear_coeffs

def restore_peaks(x, peak_shifts, x_new, y_new, z_interp, input_coeffs=None):
    '''
    Restore the original peak positions by applying the inverse shifts to the aligned
    (& interpolated) data. Peak shifts are linearly interpolated from x to x_new.

    Parameters:
    - x: 1D array of original x coordinates (where peak_shifts were computed)
    - peak_shifts: 1D array of length len(x); shift[i] = y_peak[i] - y_ref
    - x_new: 1D array of interpolated x coordinates
    - y_new: 1D array of interpolated y coordinates
    - z_interp: 2D array of shape (len(x_new), len(y_new)) — aligned interpolated surface
    - input_coeffs: If peak_shifts were computed using a parametric fit (e.g. linear),
                    provide the fit coefficients to apply the same model for extrapolation.

    Returns:
    - z_restored: 2D array of shape (len(x_new), len(y_new)) with peak positions restored
    '''
    if input_coeffs is not None:
        shifts_new = np.polyval(input_coeffs, x_new)
    else:
        shifts_new = np.interp(x_new, x, peak_shifts)

    z_restored = np.zeros_like(z_interp)
    for i, delta in enumerate(shifts_new):
        # Undo the left-shift: sample aligned signal at y_new - delta
        cs = CubicSpline(y_new, z_interp[i, :], extrapolate=False)
        vals = cs(y_new - delta)
        z_restored[i, :] = np.where(np.isnan(vals), 0.0, vals)

    return z_restored

# test out different align_peaks method which assumes the peak moves linearly with input freq (ie the 2D image is diagonal)
# def align_peaks_linear(x, y, z):
    # fit a diagonal line in (x,y) to the peak positions

def interp_2d(x, y, z, x_new, y_new, method='spline', peak_track=False, peak_track_method='nonparametric', amplitude_model=None, sigmax=1.0, sigmay=1.0, degree=1, kx=3, ky=3):
    """
    Interpolate z values at new (x_new, y_new) points based on known (x, y, z) data.

    Parameters:
    - x, y: 1D arrays of known x and y coordinates (must be strictly increasing)
    - z: 2D array of known z values with shape (len(x), len(y))
    - x_new, y_new: 1D arrays of new x and y coordinates where interpolation is desired
    - method: 'spline' (default) or 'gloess'
        'spline' — bicubic spline via RectBivariateSpline; exact at data points,
                   fast, but can oscillate with sparse/noisy x data.
        'gloess' — Gaussian-windowed locally-weighted polynomial regression;
                   smoother, more robust to noise, but slower and biased at peaks.
    - peak_track: If True, align peaks in y before interpolating and restore them after.
                  Improves accuracy when peaks shift systematically with x.
    - peak_track_method: Method for peak tracking ('nonparametric' or 'linear').
    - amplitude_model: If 'power_law', decouple row amplitude (peak height) from row shape.
                  Rows are normalized to unit peak before interpolating, then each interpolated row is renormalized to unit
                  peak again and rescaled by a power law fit to the known row peaks
                  (log(amp) = p*log(x) + log(A)), evaluated at x_new. This removes ripple in
                  peak height that a sparse-in-x exact 2D spline introduces when the row shape
                  changes across x. Default None.
    - sigmax, sigmay: (gloess only) Gaussian standard deviations in x and y.
    - degree: (gloess only) Degree of the local polynomial fit.
    - kx, ky: (spline only) Spline degrees in x and y (default 3 = cubic).

    Returns:
    - z_new: 2D array of interpolated z values at (x_new, y_new), shape (len(x_new), len(y_new))
    """
    if amplitude_model is not None:
        raw_peak = z.max(axis=1)
        z = z / raw_peak[:, np.newaxis]

    if peak_track:
        z_work, peak_shifts, _, _, linear_coeffs = align_peaks(x, y, z, method=peak_track_method)
    else:
        z_work = z

    if method == 'spline':
        z_new = RectBivariateSpline(x, y, z_work, kx=kx, ky=ky)(x_new, y_new)

    elif method == 'gloess':
        # Flatten the known grid (supports differently-sized x and y)
        X_known, Y_known = np.meshgrid(x, y, indexing='ij')
        x_flat = X_known.ravel()
        y_flat = Y_known.ravel()
        z_flat = z_work.ravel()

        # All 2D monomial exponents (i, j) with i+j <= degree
        exponents = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]

        # Design matrix: columns are x^i * y^j for each (i, j) in exponents
        A = np.column_stack([x_flat**i * y_flat**j for i, j in exponents])

        z_new = np.zeros((len(x_new), len(y_new)))
        for row, xi in enumerate(x_new):
            for col, yj in enumerate(y_new):
                dx = x_flat - xi
                dy = y_flat - yj
                w = np.exp(-0.5 * ((dx / sigmax)**2 + (dy / sigmay)**2))

                # Solve sqrt-weighted system directly (more numerically stable than normal equations)
                w_sqrt = np.sqrt(w)
                Aw = A * w_sqrt[:, np.newaxis]
                zw = z_flat * w_sqrt

                beta, _, _, _ = np.linalg.lstsq(Aw, zw, rcond=None)

                z_new[row, col] = sum(b * xi**i * yj**j for b, (i, j) in zip(beta, exponents))

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'spline' or 'gloess'.")

    if peak_track:
        z_new = restore_peaks(x, peak_shifts, x_new, y_new, z_new, input_coeffs=linear_coeffs if peak_track_method=='linear' else None)

    if amplitude_model is not None:
        if amplitude_model == 'power_law':
            p, log_A = np.polyfit(np.log(x), np.log(raw_peak), 1)
            amp_new = np.exp(log_A) * x_new**p
        else:
            raise ValueError(f"Unknown amplitude_model '{amplitude_model}'. Choose 'power_law' or None.")
        row_max = z_new.max(axis=1)
        z_new = np.where(row_max[:, np.newaxis] > 0, z_new / row_max[:, np.newaxis], 0.0) * amp_new[:, np.newaxis]

    return z_new
