"""
tracing.py

This module contains functions used to locate peaks in 1D data and trace them
in the orthogonal direction in a 2D image.

Functions in this module:
estimate_peak_width: estimate the widths of peaks
find_peaks:          locate peaks using the scipy.signal routine
pinpoint_peaks:      produce more accuate positions for existing peaks by
                     centroiding
reject_bad_peaks:    remove suspicious-looking peaks by a variety of methods

trace_lines:         trace lines from a set of supplied starting positions
"""
import numpy as np
import warnings
from scipy import signal, interpolate, optimize
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip, sigma_clipped_stats

from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.library.nddops import NDStacker, sum1d
from gempy.utils import logutils

from . import astromodels
from .astrotools import divide0

from astrodata import NDAstroData
from matplotlib import pyplot as plt
from datetime import datetime

log = logutils.get_logger(__name__)


################################################################################
class Aperture(object):
    def __init__(self, model, width=None, aper_lower=None, aper_upper=None):
        self.model = model
        if width is None:
            self.aper_lower = aper_lower
            self.aper_upper = aper_upper
        else:
            self.width = width

    @property
    def width(self):
        """Aperture width in pixels. Since the model is pixel-based, it makes
        sense for this to be stored in pixels rather than arcseconds."""
        return self.aper_upper - self.aper_lower

    @width.setter
    def width(self, value):
        if value > 0:
            self.aper_upper = 0.5 * value
            self.aper_lower = -self.aper_upper
        else:
            raise ValueError("Width must be positive ()".format(value))

    def limits(self):
        """Return maximum and minimum values of the model across the domain"""
        pixels = np.arange(*self.model.domain)
        values = self.model(pixels)
        return np.min(values), np.max(values)

    def check_domain(self, npix):
        """Simple function to warn user if aperture model appears inconsistent
        with the array containing the data"""
        try:
            if self.model.domain != [0, npix - 1]:
                log.warning("Model's domain is inconsistent with image size. "
                            "Results may be incorrect.")
        except AttributeError:  # no "domain" attribute
            pass

    def aperture_mask(self, ext=None, width=None, aper_lower=None,
                      aper_upper=None, grow=None):
        """
        This method creates a boolean array indicating pixels that will be
        used to extract a spectrum from the parent Aperture object.

        Parameters
        ----------
        ext: single-slice AD object
            image containing aperture (needs .shape attribute and
            .dispersion_axis() descriptor)
        width: float/None
            total extraction width for symmetrical aperture (pixels)
        aper_lower: float/None
            lower extraction limit (pixels)
        aper_upper: float/None
            upper extraction limit (pixels)
        grow: float/None
            additional buffer zone around width

        Returns
        -------
        ndarray: boolean array, set to True if a pixel will be used in the
                 extraction
        """
        if width is not None:
            aper_upper = 0.5 * width
            aper_lower = -aper_upper
        if aper_lower is None:
            aper_lower = self.aper_lower
        if aper_upper is None:
            aper_upper = self.aper_upper
        if grow is not None:
            aper_lower -= grow
            aper_upper += grow

        dispaxis = 2 - ext.dispersion_axis()  # python sense
        npix = ext.shape[dispaxis]
        slitlength = ext.shape[1 - dispaxis]
        self.check_domain(npix)
        center_pixels = self.model(np.arange(npix))
        x1, x2 = center_pixels + aper_lower, center_pixels + aper_upper
        ix1 = np.where(x1 < -0.5, 0, (x1 + 0.5).astype(int))
        ix2 = np.where(x2 >= slitlength - 0.5, None, (x2 + 1.5).astype(int))
        apmask = np.zeros_like(ext.data if dispaxis == 0 else ext.data.T,
                               dtype=bool)
        for i, limits in enumerate(zip(ix1, ix2)):
            apmask[i, slice(*limits)] = True
        return apmask if dispaxis == 0 else apmask.T

    def standard_extraction(self, data, mask, var, aper_lower, aper_upper):
        """Uniform extraction across an aperture of width pixels"""
        slitlength = data.shape[0]
        all_x1 = self.center_pixels + aper_lower
        all_x2 = self.center_pixels + aper_upper

        ext = NDAstroData(data, mask=mask)
        ext.variance = var
        results = [sum1d(ext[:,i], x1, x2) for i, (x1, x2) in enumerate(zip(all_x1, all_x2))]
        self.data[:] = [result.data for result in results]
        if mask is not None:
            self.mask[:] = [result.mask for result in results]
        if var is not None:
            self.var[:] = [result.variance for result in results]

    def optimal_extraction(self, data, mask, var, aper_lower, aper_upper,
                           cr_rej=5, max_iters=None):
        """Optimal extraction following Horne (1986, PASP 98, 609)"""
        BAD_BITS = DQ.bad_pixel | DQ.cosmic_ray | DQ.no_data | DQ.unilluminated

        slitlength, npix = data.shape
        pixels = np.arange(npix)

        all_x1 = self.center_pixels + aper_lower
        all_x2 = self.center_pixels + aper_upper
        ix1 = max(int(min(all_x1) + 0.5), 0)
        ix2 = min(int(max(all_x2) + 1.5), slitlength)

        fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                   outlier_func=sigma_clip, sigma_upper=3, sigma_lower=None)

        # If we don't have a VAR plane, assume uniform variance based
        # on the pixel-to-pixel variations in the data
        if var is None:
            var_model = models.Const1D(sigma_clipped_stats(data, mask=mask)[2] ** 2)
            var = np.full_like(data[ix1:ix2], var_model.amplitude)
            var_mask = np.zeros_like(var, dtype=bool)
        else:
            mvar_init = models.Polynomial1D(degree=1)
            var_model, var_mask = fit_it(mvar_init, np.ma.masked_where(mask.ravel(), abs(data).ravel()), var.ravel())
            var_mask = var_mask.reshape(var.shape)[ix1:ix2]
            var = np.where(var_mask, var[ix1:ix2], var_model(data[ix1:ix2]))

        if mask is None:
            mask = np.zeros((ix2 - ix1, npix), dtype=DQ.datatype)
        else:
            mask = mask[ix1:ix2]

        data = data[ix1:ix2]
        # Step 4; first calculation of spectrum. We don't do any masking
        # here since we need all the flux
        spectrum = data.sum(axis=0)
        weights = np.where(var > 0, var, 0)
        unmask = np.ones_like(data, dtype=bool)

        iter = 0
        while True:
            # Step 5: construct spatial profile for each wavelength pixel
            profile = np.divide(data, spectrum,
                                out=np.zeros_like(data, dtype=np.float32), where=spectrum > 0)
            profile_models = []
            for row, wt_row in zip(profile, weights):
                m_init = models.Chebyshev1D(degree=3, domain=[0, npix - 1])
                m_final, _ = fit_it(m_init, pixels, row, weights=wt_row)
                profile_models.append(m_final(pixels))
            profile_model_spectrum = np.array([np.where(pm < 0, 0, pm) for pm in profile_models])
            sums = profile_model_spectrum.sum(axis=0)
            model_profile = divide0(profile_model_spectrum, sums, like_num=True)

            # Step 6: revise variance estimates
            var = np.where(var_mask | mask & BAD_BITS, var, var_model(abs(model_profile * spectrum)))
            weights = divide0(1.0, var)

            # Step 7: identify cosmic ray hits: we're (probably) OK
            # to flag more than 1 per wavelength
            sigma_deviations = divide0(data - model_profile * spectrum, np.sqrt(var)) * unmask
            mask[sigma_deviations > cr_rej] |= DQ.cosmic_ray
            # most_deviant = np.argmax(sigma_deviations, axis=0)
            # for i, most in enumerate(most_deviant):
            #    if sigma_deviations[most, i] > cr_rej:
            #        mask[most, i] |= DQ.cosmic_ray

            last_unmask = unmask
            unmask = (mask & BAD_BITS) == 0
            spec_numerator = np.sum(unmask * model_profile * data * weights, axis=0)
            spec_denominator = np.sum(unmask * model_profile ** 2 * weights, axis=0)
            self.data = divide0(spec_numerator, spec_denominator)
            self.var = divide0(np.sum(unmask * model_profile, axis=0), spec_denominator)
            self.mask = np.bitwise_and.reduce(mask, axis=0)
            spectrum = self.data

            iter += 1
            # An unchanged mask means XORing always produces False
            if ((np.sum(unmask ^ last_unmask) == 0) or
                    (max_iters is not None and iter > max_iters)):
                break

    def extract(self, ext, width=None, aper_lower=None, aper_upper=None,
                method='standard', dispaxis=None, viewer=None):
        """
        Extract a 1D spectrum by following the model trace and extracting in
        an aperture of a given number of pixels.

        Parameters
        ----------
        ext: single-slice AD object/ndarray
            spectral image from which spectrum is to be extracted
        width: float/None
            full width of extraction aperture (in pixels)
        aper_lower: float/None
            lower extraction limit (pixels)
        aper_upper: float/None
            upper extraction limit (pixels)
        method: str (standard|optimal)
            type of extraction
        dispaxis: int/None
            dispersion axis (python sense)
        viewer: Viewer/None
            Viewer object on which to display extraction apertures

        Returns
        -------
        NDAstroData: 1D spectrum
        """
        if width is not None:
            aper_upper = 0.5 * width
            aper_lower = -aper_upper
        if aper_lower is None:
            aper_lower = self.aper_lower
        if aper_upper is None:
            aper_upper = self.aper_upper

        if dispaxis is None:
            dispaxis = 2 - ext.dispersion_axis()  # python sense
        slitlength = ext.shape[1 - dispaxis]
        npix = ext.shape[dispaxis]
        direction = "row" if dispaxis == 0 else "column"

        self.check_domain(npix)

        # make data look like it's dispersed horizontally
        # (this is best for optimal extraction, but not standard)
        try:
            mask = ext.mask
        except AttributeError:  # ext is just an ndarray
            data = ext if dispaxis == 1 else ext.T
            mask = None
            var = None
        else:
            data = ext.data if dispaxis == 1 else ext.data.T
            if dispaxis == 0 and mask is not None:
                mask = mask.T
            var = ext.variance
            if dispaxis == 0 and var is not None:
                var = var.T

        # Avoid having to recalculate them
        self.center_pixels = self.model(np.arange(npix))
        all_x1 = self.center_pixels + aper_lower
        all_x2 = self.center_pixels + aper_upper
        if viewer is not None:
            # Display extraction edges on viewer, every 10 pixels (for speed)
            pixels = np.arange(npix)
            edge_coords = np.array([pixels, all_x1]).T
            viewer.polygon(edge_coords[::10], closed=False, xfirst=(dispaxis == 1), origin=0)
            edge_coords = np.array([pixels, all_x2]).T
            viewer.polygon(edge_coords[::10], closed=False, xfirst=(dispaxis == 1), origin=0)

        # Remember how pixel coordinates are defined!
        off_low = np.where(all_x1 < -0.5)[0]
        if len(off_low):
            log.warning("Aperture extends off {} of image between {}s {} and {}".
                        format(("left", "bottom")[dispaxis], direction,
                               min(off_low), max(off_low)))
        off_high = np.where(all_x2 >= slitlength - 0.5)[0]
        if len(off_high):
            log.warning("Aperture extends off {} of image between {}s {} and {}".
                        format(("right", "top")[dispaxis], direction,
                               min(off_high), max(off_high)))

        # Create the outputs here so the extraction function has access to them
        self.data = np.zeros((npix,), dtype=np.float32)
        self.mask = None if mask is None else np.zeros_like(self.data,
                                                            dtype=DQ.datatype)
        self.var = None if var is None else np.zeros_like(self.data)

        extraction_func = getattr(self, "{}_extraction".format(method))
        extraction_func(data, mask, var, aper_lower, aper_upper)

        del self.center_pixels
        ndd = NDAstroData(self.data, mask=self.mask)
        ndd.variance = self.var
        try:
            ndd.meta['header'] = ext.hdr.copy()
        except AttributeError:  # we only had an ndarray
            pass
        return ndd


################################################################################
# FUNCTIONS RELATED TO PEAK-FINDING

def estimate_peak_width(data, min=2, max=8):
    """
    Estimates the FWHM of the spectral features (arc lines) by fitting
    Gaussians to the brightest peaks.

    Parameters
    ----------
    data:  ndarray
        1D data array (will be modified)
    min: int
        minimum plausible peak width
    max: int
        maximum plausible peak width (not inclusive)

    Returns
    -------
    float: estimate of FWHM of features
    """
    all_widths = []
    for fwidth in range(min, max + 1):  # plausible range of widths
        data_copy = data.copy()  # We'll be editing the data
        widths = []
        for i in range(15):  # 15 brightest peaks, should ensure we get real ones
            index = 2 * fwidth + np.argmax(data_copy[2 * fwidth:-2 * fwidth - 1])
            data_to_fit = data_copy[index - 2 * fwidth:index + 2 * fwidth + 1]
            m_init = models.Gaussian1D(stddev=0.42466 * fwidth) + models.Const1D(np.min(data_to_fit))
            m_init.mean_0.bounds = [-1, 1]
            m_init.amplitude_1.fixed = True
            fit_it = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(),
                                                       sigma_clip, sigma=3)
            with warnings.catch_warnings():
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                m_final, _ = fit_it(m_init, np.arange(-2 * fwidth, 2 * fwidth + 1),
                                    data_to_fit)
            # Quick'n'dirty logic to remove "peaks" at edges of CCDs
            if m_final.amplitude_1 != 0 and m_final.stddev_0 < fwidth:
                widths.append(m_final.stddev_0 / 0.42466)

            # Set data to zero so no peak is found here
            data_copy[index - 2 * fwidth:index + 2 * fwidth + 1] = 0.
        all_widths.append(sigma_clip(widths).mean())
    return sigma_clip(all_widths).mean()


def find_peaks(data, widths, mask=None, variance=None, min_snr=1, min_frac=0.25,
               reject_bad=True):
    """
    Find peaks in a 1D array. This uses scipy.signal routines, but requires some
    duplication of that code since the _filter_ridge_lines() function doesn't
    expose the *window_size* parameter, which is important. This also does some
    rejection based on a pixel mask and/or "forensic accounting" of relative
    peak heights.

    Parameters
    ----------
    data: 1D array
        The pixel values of the 1D spectrum
    widths: array-like
        Sigma values of line-like features to look for
    mask: 1D array (optional)
        Mask (peaks with bad pixels are rejected) - optional
    variance: 1D array (optional)
        Variance (to estimate SNR of peaks) - optional
    min_snr: float (optional, default=1)
        Minimum SNR required of peak pixel
    min_frac: float (optional, default=0.25)
        minimum fraction of *widths* values at which a peak must be found
    reject_bad: bool (optional, default=True)
        clip lines using the reject_bad() function?

    Returns
    -------
    2D array: peak wavelengths and SNRs
    """
    mask = mask if mask is not None else np.zeros_like(data, dtype=np.uint16)

    if not np.issubdtype(mask.dtype, np.unsignedinteger):
        raise TypeError("Expected input parameter 'mask' to be an array with unsigned integer. "
                        "Found: {}".format(mask.dtype))

    max_width = max(widths)
    window_size = 4 * max_width + 1

    wavelet_transformed_data = signal.cwt(data, signal.ricker, widths)

    eps = np.finfo(np.float32).eps  # Minimum representative data
    wavelet_transformed_data[wavelet_transformed_data < eps] = eps

    ridge_lines = signal._peak_finding._identify_ridge_lines(
        wavelet_transformed_data, 0.03 * widths, 2)

    filtered = signal._peak_finding._filter_ridge_lines(
        wavelet_transformed_data, ridge_lines, window_size=window_size,
        min_length=int(min_frac * len(widths)), min_snr=min_snr)

    peaks = sorted([x[1][0] for x in filtered])

    # If no variance is supplied we take the "snr" to be the data.
    # We do this on the filtered data because the continuum level gets
    # subtracted by the Ricker filter
    if variance is not None:
        snr = np.divide(wavelet_transformed_data[0], np.sqrt(variance),
                        out=np.zeros_like(data, dtype=np.float32),
                        where=variance > 0)
    else:
        snr = wavelet_transformed_data[0]

    peaks = [x for x in peaks if snr[x] > min_snr]

    # remove adjacent points
    min_separation = min(widths)
    new_peaks = []
    i = 0
    while i < len(peaks) - 1:
        j = i + 1
        while j <= len(peaks) - 1 and (peaks[j] - peaks[j - 1] < min_separation):
            j += 1
        new_peaks.append(np.mean(peaks[i:j]))
        i = j
        if i == len(peaks) - 1:
            new_peaks.append(peaks[i])

    # Turn into array and remove those too close to the edges
    peaks = np.array(new_peaks)
    edge = 2.35482 * max_width
    peaks = peaks[np.logical_and(peaks > edge, peaks < len(data) - 1 - edge)]

    # Remove peaks very close to unilluminated/no-data pixels
    # (e.g., chip gaps in GMOS)
    peaks = [x for x in peaks if np.sum(mask[int(x-edge):int(x+edge+1)] & (DQ.no_data | DQ.unilluminated)) == 0]

    # Clip the really noisy parts of the data before getting more accurate
    # peak positions and clip SNR again with new positions
    peaks = pinpoint_peaks(np.where(snr < 0.5, 0, data), mask, peaks)
    final_peaks = [p for p in peaks if snr[int(p + 0.5)] > min_snr]
    peak_snrs = list(snr[int(p + 0.5)] for p in final_peaks)

    # Remove suspiciously bright peaks and return as array of
    # locations and SNRs, sorted by location
    if reject_bad:
        good_peaks = reject_bad_peaks(list(zip(final_peaks, peak_snrs)))
    else:
        good_peaks = list(zip(final_peaks, peak_snrs))

    return np.array(sorted(good_peaks)).T


def pinpoint_peaks(data, mask, peaks, halfwidth=4, threshold=0):
    """
    Improves positions of peaks with centroiding. It uses a deliberately
    small centroiding box to avoid contamination by nearby lines, which
    means the original positions must be good. It also removes any peaks
    affected by the mask

    The method used here is a direct recoding from IRAF's center1d function,
    as utilized by the identify and reidentify tasks. Extensive use has
    demonstrated the reliability of that method.

    Parameters
    ----------
    data: 1D array
        The pixel values of the 1D spectrum
    mask: 1D array
        Mask (peaks with bad pixels are rejected)
    peaks: sequence
        approximate (but good) location of peaks
    halfwidth: int
        number of pixels either side of initial peak to use in centroid
    threshold: float
        threshold to cut data

    Returns
    -------
    list: more accurate locations of the peaks that are unaffected by the mask
    """
    int_limits = np.array([-1, -0.5, 0.5, 1])
    npts = len(data)
    final_peaks = []

    if mask is None:
        masked_data = data - threshold
    else:
        masked_data = np.where(np.logical_or(mask > 0, data < threshold),
                               0, data - threshold)

    for peak in peaks:
        xc = int(peak + 0.5)
        xc = np.argmax(masked_data[max(xc - 1, 0):min(xc + 2, npts)]) + xc - 1
        if xc < halfwidth or xc > npts - halfwidth:
            continue
        x1 = int(xc - halfwidth)
        x2 = int(xc + halfwidth + 1)
        xvalues = np.arange(x1, x2)

        # We fit splines to y(x) and x * y(x)
        t, c, k = interpolate.splrep(xvalues, masked_data[x1:x2], k=3,
                                     s=0)
        spline1 = interpolate.BSpline.construct_fast(t, c, k, extrapolate=False)
        t, c, k = interpolate.splrep(xvalues, masked_data[x1:x2] * xvalues,
                                     k=3, s=0)
        spline2 = interpolate.BSpline.construct_fast(t, c, k, extrapolate=False)

        # Then there's some centroiding around the peak, with convergence
        # tolerances. This is still just an IRAF re-code.
        final_peak = None
        dxlast = x2 - x1
        dxcheck = 0
        for i in range(50):
            # Triangle centering function
            limits = xc + int_limits * halfwidth
            splint1 = [spline1.integrate(x1, x2) for x1, x2 in zip(limits[:-1], limits[1:])]
            splint2 = [spline2.integrate(x1, x2) for x1, x2 in zip(limits[:-1], limits[1:])]
            sum1 = (splint2[1] - splint2[0] - splint2[2] +
                    (xc - halfwidth) * splint1[0] - xc * splint1[1] + (xc + halfwidth) * splint1[2])
            sum2 = splint1[1] - splint1[0] - splint1[2]

            if sum2 == 0:
                break

            dx = sum1 / abs(sum2)
            xc += max(-1, min(1, dx))
            if xc < halfwidth or xc > npts - halfwidth:
                break
            if abs(dx) < 0.001:
                final_peak = xc
                break
            if abs(dx) > dxlast + 0.005:
                dxcheck += 1
                if dxcheck > 3:
                    break
            elif abs(dx) > dxlast - 0.005:
                xc -= 0.5 * max(-1, min(1, dx))
                dxcheck = 0
            else:
                dxcheck = 0
                dxlast = abs(dx)

        if final_peak is not None:
            final_peaks.append(final_peak)

    return final_peaks


def reject_bad_peaks(peaks):
    """
    Go through the list of peaks and remove any that look really suspicious.
    I refer to this colloquially as "forensic accounting".

    Parameters
    ----------
    peaks: sequence of 2-tuples:
        peak location and "strength" (e.g., SNR) of peaks

    Returns
    -------
    sequence of 2-tuples:
        accepted elements of input list
    """
    diff = 3  # Compare 1st brightest to 4th brightest
    peaks.sort(key=lambda x: x[1])  # Sort by SNR
    while len(peaks) > diff and (peaks[-1][1] / peaks[-(diff + 1)][1] > 3):
        del peaks[-1]
    return peaks


def get_limits(data, mask, variance=None, peaks=[], threshold=0, method=None):
    """
    Determines the region in a 1D array associated with each already-identified
    peak.

    It operates by fitting a spline to the data (with weighting set based on
    pixel-to-pixel scatter) and then differentiating this to find maxima and
    minima. For each peak, the region is between the two closest minima, one on
    each side.

    If the threshold_function is None, then the locations of these minima
    are returned. Otherwise, this must be a callable function that takes
    the spline and the location of the minimum and the peak and returns
    zero at the location one wants to be returned.

    Parameters
    ----------
    data: ndarray
        1D profile containing the peaks
    mask: ndarray (bool-like)
        mask indicating pixels in data to ignore
    variance: ndarray/None
        variance of each pixel (None => recalculate)
    peaks: sequence
        peaks for which limits are to be found
    threshold_function: None/callable
        takes 3 arguments: the spline, the peak location, and the
                           location of the minimum, and returns a value
                           that must be zero at the desired limit's location

    Returns
    -------
    list of 2-element lists:
        the lower and upper limits for each peak
    """

    # We abstract the limit-finding function. Any function can be added here,
    # providing it has a standard call signature, taking parameters:
    # spline representation, location of peak, location of limit on this side,
    # location of limit on other side, thresholding value
    functions = {'threshold': threshold_limit,
                 'integral': integral_limit}
    try:
        limit_finding_function = functions[method]
    except KeyError:
        method = None

    x = np.arange(len(data))
    y = np.ma.masked_array(data, mask=mask)

    # Try to better estimate the true noise from the pixel-to-pixel
    # variations (the difference between adjacent pixels will be
    # sqrt(2) times larger than the rms noise).
    if variance is None:
        w = np.full_like(y, np.sqrt(2) / sigma_clipped_stats(np.diff(y))[2])
    else:
        w = divide0(1.0, np.sqrt(variance))

    # We need to fit a quartic spline since we want to know its
    # minima (roots of its derivative), and can only find the
    # roots of a cubic spline
    # TODO: Quartic splines look bad with outlier removal
    #spline = astromodels.UnivariateSplineWithOutlierRemoval(x, y, w=w, k=4)
    spline = interpolate.UnivariateSpline(x, y, w=w, k=4)

    derivative = spline.derivative(n=1)
    extrema = derivative.roots()
    second_derivatives = spline.derivative(n=2)(extrema)
    minima = [ex for ex, second in zip(extrema, second_derivatives) if second > 0]

    all_limits = []
    for peak in peaks:
        tweaked_peak = extrema[np.argmin(abs(extrema - peak))]

        # Now find the nearest minima above and below the peak
        for upper in minima:
            if upper > peak:
                break
        else:
            upper = len(data) - 1
        for lower in reversed(minima):
            if lower < peak:
                break
        else:
            lower = 0

        if method is None:
            all_limits.append((lower, upper))
        else:
            limit1 = limit_finding_function(spline, tweaked_peak, lower, upper, threshold)
            limit2 = limit_finding_function(spline, tweaked_peak, upper, lower, threshold)
            all_limits.append((limit1, limit2))

    return all_limits


def threshold_limit(spline, peak, limit, other_limit, threshold):
    """
    Finds a threshold as a fraction of the way from the signal at the minimum to
    the signal at the peak.

    Parameters
    ----------
    spline : callable
        ???
    peak : ???
        ???
    limit : ???
        ???
    other_limit : ???
        ???
    threshold : ???
        ???

    Returns
    -------
    ???
        ???
       """
    # target is the signal level at the threshold
    target = spline(limit) + threshold * (spline(peak) - spline(limit))
    func = lambda x: spline(x) - target
    return optimize.bisect(func, limit, peak)


def integral_limit(spline, peak, limit, other_limit, threshold):
    """Finds a threshold as a fraction of the missing flux, defined as the
       area under the signal between the peak and this limit, removing a
       straight line between the two points"""
    integral = spline.antiderivative()
    slope = (spline(other_limit) - spline(limit)) / (other_limit - limit)
    definite_integral = lambda x: integral(x) - integral(limit) - (x - limit) * (
                (spline(limit) - slope * limit) + 0.5 * slope * (x + limit))
    flux_this_side = definite_integral(peak) - definite_integral(limit)

    # definite_integral is the flux from the limit towards the peak, so this
    # should be equal to the required fraction of the flux om that side of
    # the peak.
    func = lambda x: definite_integral(x) / flux_this_side - threshold
    return optimize.bisect(func, limit, peak)


################################################################################
# FUNCTIONS RELATED TO PEAK-TRACING

def trace_lines(ext, axis, start=None, initial=None, width=5, nsum=10,
                step=1, initial_tolerance=1.0, max_shift=0.05, max_missed=10,
                func=NDStacker.mean, viewer=None):
    """
    This function traces features along one axis of a two-dimensional image.
    Initial peak locations are provided and then these are matched to peaks
    found a small distance away along the direction of tracing. In terms of
    its use to map the distortion from a 2D spectral image of an arc lamp,
    these lists of coordinates can then be used to determine a distortion map
    that will remove any curvature of lines of constant wavelength.

    For a horizontally-dispersed spectrum like GMOS, the reference y-coords
    will match the input y-coords, while the reference x-coords will all be
    equal to the initial x-coords of the peaks.

    Parameters
    ----------
    ext : single-sliced AD object
        The extension within which to trace features.

    axis : int (0 or 1)
        Axis along which to trace (0=y-direction, 1=x-direction).

    start : int/None
        Row/column to start trace (None => middle).

    initial : sequence
        Coordinates of peaks.

    width : int
        Width of centroid box in pixels.

    nsum : int
        Number of rows/columns to combine at each step.

    step : int
        Step size along axis in pixels.

    initial_tolerance : float
        Maximum perpendicular shift (in pixels) between provided location and
        first calculation of peak.

    max_shift: float
        Maximum perpendicular shift (in pixels) from pixel to pixel.

    max_missed: int
        Maximum number of interactions without finding line before line is
        considered lost forever.

    func: callable
        function to use when collapsing to 1D. This takes the data, mask, and
        variance as arguments.

    viewer: imexam viewer or None
        Viewer to draw lines on.

    Returns
    -------
    refcoords, incoords: 2xN arrays (x-first) of coordinates
    """
    log = logutils.get_logger(__name__)

    # We really don't care about non-linear/saturated pixels
    bad_bits = 65535 ^ (DQ.non_linear | DQ.saturated)

    halfwidth = int(0.5 * width)

    # Make life easier for the poor coder by transposing data if needed,
    # so that we're always tracing along columns
    if axis == 0:
        ext_data = ext.data
        ext_mask = None if ext.mask is None else ext.mask & bad_bits
        direction = "row"
    else:
        ext_data = ext.data.T
        ext_mask = None if ext.mask is None else ext.mask.T & bad_bits
        direction = "column"

    if start is None:
        start = int(0.5 * ext_data.shape[0])
        log.stdinfo("Starting trace at {} {}".format(direction, start))

    if initial is None:
        y1 = int(start - 0.5 * nsum + 0.5)
        data, mask, var = NDStacker.mean(ext_data[y1:y1 + nsum],
                                         mask=None if ext_mask is None else ext_mask[y1:y1 + nsum],
                                         variance=None)
        fwidth = estimate_peak_width(data.copy(), 10)
        widths = 0.42466 * fwidth * np.arange(0.8, 1.21, 0.05)  # TODO!
        initial, _ = find_peaks(data, widths, mask=mask,
                                variance=var, min_snr=5)
        print("Feature width", fwidth, "nlines", len(initial))

    coord_lists = [[] for peak in initial]
    for direction in (-1, 1):
        ypos = start
        last_coords = [[ypos, peak] for peak in initial]

        while True:
            y1 = int(ypos - 0.5 * nsum + 0.5)
            data, mask, var = func(ext_data[y1:y1 + nsum],
                                   mask=None if ext_mask is None else ext_mask[y1:y1 + nsum],
                                   variance=None)
            # Variance could plausibly be zero
            var = np.where(var <= 0, np.inf, var)
            clipped_data = np.where(data / np.sqrt(var) > 0.5, data, 0)
            last_peaks = [c[1] for c in last_coords if not np.isnan(c[1])]
            peaks = pinpoint_peaks(clipped_data, mask, last_peaks)
            # if ypos == start:
            #    print("Found {} peaks".format(len(peaks)))
            #    print(peaks)

            for i, (last_row, old_peak) in enumerate(last_coords):
                if np.isnan(old_peak):
                    continue
                # If we found no peaks at all, then continue through
                # the loop but nothing will match
                if peaks:
                    j = np.argmin(abs(np.array(peaks) - old_peak))
                    new_peak = peaks[j]
                else:
                    new_peak = np.inf

                # Is this close enough to the existing peak?
                tolerance = (initial_tolerance if ypos == start
                             else max_shift * abs(ypos - last_row))
                if (abs(new_peak - old_peak) > tolerance):
                    # If it's gone for good, set the coord to NaN to avoid it
                    # picking up a different line if there's significant tilt
                    if abs(ypos - last_row) > max_missed * step:
                        last_coords[i][1] = np.nan
                    continue

                # Too close to the edge?
                if (new_peak < halfwidth or
                        new_peak > ext_data.shape[1] - 0.5 * halfwidth):
                    last_coords[i][1] = np.nan
                    continue

                new_coord = [ypos, new_peak]
                if viewer:
                    kwargs = dict(zip(('y1', 'x1'), last_coords[i] if axis == 0
                    else reversed(last_coords[i])))
                    kwargs.update(dict(zip(('y2', 'x2'), new_coord if axis == 0
                    else reversed(new_coord))))
                    viewer.line(origin=0, **kwargs)

                if not (ypos == start and direction > 1):
                    coord_lists[i].append(new_coord)
                last_coords[i] = new_coord.copy()

            ypos += direction * step
            # Reached the bottom or top?
            if ypos < 0.5 * nsum or ypos > ext_data.shape[0] - 0.5 * nsum:
                break

            # Lost all lines!
            if all(np.isnan(c[1]) for c in last_coords):
                break

    # List of traced peak positions
    in_coords = np.array([c for coo in coord_lists for c in coo]).T
    # List of "reference" positions (i.e., the coordinate perpendicular to
    # the line remains constant at its initial value
    ref_coords = np.array([(ypos, ref) for coo, ref in zip(coord_lists, initial) for (ypos, xpos) in coo]).T

    # Return the coordinate lists, in the form (x-coords, y-coords),
    # regardless of the dispersion axis
    return (ref_coords, in_coords) if axis == 1 else (ref_coords[::-1], in_coords[::-1])
