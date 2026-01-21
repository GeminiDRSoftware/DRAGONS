# Copyright(c) 2019-2025 Association of Universities for Research in Astronomy, Inc.

"""
peaks.py

This module contains functions used to find peaks in 1D profiles.

Functions in this module:
average_along_slit:  collapse a 2D spectral image in the wavelength direction
                     to produce a slit profile
cross_correlate_subpixels: determine subpixel cross-correlation between two
                           arrays by subsampling each
cwt_ricker:          Ricker filter
estimate_peak_width: estimate the widths of peaks
find_apertures:      find sources and define apertures in a 2D spectral image
find_wavelet_peaks:  locate peaks using the scipy.signal routine
get_extrema:         find all minima and maxima in a profile
get_limits:          calculate aperture limits for a particular source
pinpoint_peaks:      produce more accuate positions for existing peaks by
                     centroiding
reject_bad_peaks:    remove suspicious-looking peaks by a variety of methods
stack_slit:          quick stack a of 2D spectral image along the slit
"""
import warnings

import numpy as np
from astropy.modeling import models
from astropy.stats import sigma_clip, sigma_clipped_stats
from matplotlib import pyplot as plt
from scipy import interpolate, optimize, signal

from astrodata import NDAstroData
from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.library.nddops import NDStacker, combine1d
from gempy.utils import logutils

from . import astrotools as at, astromodels as am
from ..utils.decorators import unpack_nddata, insert_descriptor_values

log = logutils.get_logger(__name__)


@insert_descriptor_values("dispersion_axis")
def average_along_slit(ext, center=None, offset_from_center=None,
                       nsum=None, dispersion_axis=None, combiner="mean"):
    """
    Calculates the average along the slit and its pixel-by-pixel variance.

    Parameters
    ----------
    ext : `AstroData` slice
        2D spectral image from which trace is to be extracted.
    center : float or None
        Center of averaging region (None => center of axis).
        Python 0-indexed. Used with longslit spectra.
    offset_from_center : float or None
        Offset of center of extracted aperature from center of illuminated
        region. Used with cross-dispersed spectra.
    nsum : int
        Number of rows/columns to combine
    combiner : str
        Method to use for combining

    Returns
    -------
    data : array_like
        Averaged data of the extracted region.
    mask : array_like
        Mask of the extracted region.
    variance : array_like
        Variance of the extracted region based on pixel-to-pixel variation.
    extract_slice : slice
        Slice object for extraction region.
    -- OR --
    slit_polynomial : `Chebyshev1D` model
        Chebyshev polynomial representing the center of the extracted aperture.
    """
    constant_slit = 'LS' in ext.tags or "TRANSFRM" in ext.phu
    npix, mpix = ext.shape[1 - dispersion_axis], ext.shape[dispersion_axis]
    if nsum is None:
        nsum = npix
    if center is None:
        center = 0.5 * (npix - 1)
    if offset_from_center is None:
        offset_from_center = 0

    if constant_slit:
        extract_slice = slice(max(0, int(center + 1 - 0.5 * nsum)),
                              min(npix, int(center + 1 + 0.5 * nsum)))
    else:
        extract_slice = slice(None)

    # Transpose the data if needed. If the slit is constant, extracting it here
    # is all that is needed.
    data, mask, variance = at.transpose_if_needed(
        ext.data, ext.mask, ext.variance,
        transpose=(dispersion_axis == 0), section=extract_slice)

    if constant_slit:
        # Create 1D spectrum; pixel-to-pixel variation is a better indicator
        # of S/N than the VAR plane
        # FixMe: "variance=variance" breaks test_gmos_spect_ls_distortion_determine.
        #  Use "variance=None" to make them pass again.
        data, mask, variance = NDStacker.combine(data, mask=mask, variance=None,
                                                 combiner=combiner)

        return data, mask, variance, extract_slice

    else:
        edge1, edge2 = [am.table_to_model(row) for row in ext.SLITEDGE]
        # Create a polynomial tracing the slit between the two edges.
        # The coefficients for this polynomial are a combination of
        # those of the two edge polynomials times the fraction across the
        # extension where the center line is to be.
        center_coeffs = [(p1 + p2) * 0.5 for p1, p2
                         in zip(edge1.parameters, edge2.parameters)]
        coeffs_dict = {coeff: value for coeff, value in zip(edge1.param_names,
                                                            center_coeffs)}
        coeffs_dict['c0'] += offset_from_center
        slit_polynomial = models.Chebyshev1D(degree=edge1.degree,
                                             domain=edge1.domain,
                                             **coeffs_dict)

        centers = np.array([slit_polynomial(n) for n in range(0, mpix)])
        data_out = np.empty([mpix], dtype=np.float32)
        mask_out = np.empty_like(data_out, dtype=np.uint16)
        variance_out = np.empty_like(data_out)

        # Average `nsum` pixels around the center at each column.
        # FIXME: This should ideally respect the "combiner" argument, and handle
        # all the combine methods NDStacker does. (mean, median, wtmean, lmedian)
        for i, center_pix in enumerate(centers):
            n1 = center_pix + 1 - 0.5 * nsum
            n2 = center_pix + 1 + 0.5 * nsum
            nddata = NDAstroData(data[:, i], mask=mask[:, i],
                                 variance=variance[:, i])
            data_out[i], mask_out[i], variance_out[i] = combine1d(
                nddata, n1, n2, average=True)

        # Pass the polynomial for the center instead of the extract slice
        return data_out, mask_out, variance_out, slit_polynomial


def cross_correlate_subpixels(data, template, sampling=10):
    """
    Cross-correlate two arrays to find the shift between them after
    subsampling each of them.

    Parameters
    ----------
    data : numpy.ndarray or maskedarray or NDData
        The data array.
    template : numpy.ndarray (no mask)
        The template array.
    sampling : int
        The number of subpixels to use for the cross-correlation.

    Returns
    -------
    float
        The shift between the two arrays.
    """
    has_mask = hasattr(data, 'mask')
    if has_mask:
        goodpix = ~data.mask.astype(bool)
        data = data.data
    else:
        goodpix = np.ones_like(data, dtype=bool)

    subpixels = np.arange(0, data.size - 0.01, 1. / sampling)
    spline = interpolate.interp1d(np.arange(data.size)[goodpix], data[goodpix],
                                  kind='cubic', copy=False, bounds_error=False,
                                  fill_value=np.nan)
    subsampled_data = spline(subpixels)

    subsampled_template = interpolate.interp1d(
        np.arange(data.size), template, kind='cubic', copy=False,
        bounds_error=False, fill_value=np.nan)(subpixels)

    # If the ends of the data array are masked, these will become NaNs in
    # the subsampled version and we need to ignore them
    goodpix = ~np.isnan(subsampled_data)  # same for subsampled_template
    xcorr = np.correlate(subsampled_template[goodpix],
                         subsampled_data[goodpix], mode="full")
    xpeaks, _ = pinpoint_peaks(xcorr, [xcorr.argmax()], halfwidth=5)
    if xpeaks:
        return (xpeaks[0] - xcorr.size // 2) / sampling
    return None


def cwt_ricker(data, widths):
    """
    Continuous wavelet transform, using the Ricker filter.
    """
    output = np.zeros((len(widths), len(data)), dtype=np.float64)
    for ind, width in enumerate(widths):
        N = int(np.min([10 * width, len(data)])) // 2 * 2 - 1
        x = np.arange(N) - N // 2
        normsq = (x / width) ** 2
        ricker = 2 / (np.sqrt(3 * width * np.sqrt(np.pi))) * (1 - normsq) * np.exp(-0.5 * normsq)
        wavelet_data = np.conj(ricker[::-1])
        output[ind] = np.convolve(data, wavelet_data, mode='same')
    return output


def estimate_peak_width(data, mask=None, boxcar_size=None, nlines=None):
    """
    Estimates the FWHM of the spectral features (arc lines) by inspecting
    pixels around the brightest peaks.

    Parameters
    ----------
    data : ndarray
        1D data array
    mask : ndarray/None
        mask to apply to data
    boxcar_size : float/None
        subtract a median boxcar from the data first?

    Returns
    -------
    float : estimate of FWHM of features in pixels
    """
    if mask is None:
        goodpix = np.ones_like(data, dtype=bool)
    else:
        goodpix = ~mask.astype(bool)
    widths = []
    niters = 0
    if boxcar_size:
        data = data - at.boxcar(data, size=boxcar_size)
    num = 10
    if nlines is not None and nlines > num:
        num = nlines
    while len(widths) < num and niters < 100:
        index = np.argmax(data * goodpix)
        with warnings.catch_warnings():  # width=0 warnings
            warnings.simplefilter("ignore")
            width = signal.peak_widths(data, [index], 0.5)[0][0]
        # Block 2 * FWHM
        hi = min(int(index + width + 1.5), len(data))
        lo = max(int(index - width), 0)
        if all(goodpix[lo:hi]) and width > 0:
            widths.append(width)
        goodpix[lo:hi] = False
        niters += 1

    if len(widths) == 0:
        return None
    else:
        return sigma_clip(widths).mean()


def get_extrema(profile, prof_mask=None, min_snr=3, remove_edge_maxima=True):
    """
    Find all the significant maxima and minima in a 1D profile. Significance
    is calculated from the prominence of each peak divided by an estimate of
    the noise from the pixel-to-pixel variations in the profile, and whether
    this exceeds the minimum S/N ratio threshold.

    Maxima and minima are found by identifying all pixels that are either
    lower than both their neighbours, or higher than both their neighbours,
    or equal to one neighbour. The precise locations of these extrema are
    then calculated (with any that fail to produce a clear result simply
    being discarded) and any adjacent pairs of minima or maxima (when
    ordered by location) merged to leave only the most prominent. As a
    result, the returned list alternates perfectly between minima and
    maxima.

    Parameters
    ----------
    profile: 1D array
        profile from which extrema are to be found
    prof_mask: 1D array/mask
        mask to apply to this profile
    min_snr: float
        minimum S/N ratio for a maximum to be considered significant
    remove_edge_maxima: bool
        remove maxima that do not have a minimum between them and the edge?

    Returns
    -------
    extrema: list of 3-element lists (float, float, bool)
        [position, value, is_maximum?]
    """
    diffs = np.array([np.diff(profile[:-1]), -np.diff(profile[1:])])
    extrema = np.multiply.reduce(diffs, axis=0) >= 0
    extrema_types = np.add.reduce(diffs, axis=0)
    xpixels = np.arange(profile.size)[1:-1]
    max_locations = xpixels[np.logical_and(extrema, extrema_types > 0)]
    min_locations = xpixels[np.logical_and(extrema, extrema_types < 0)]
    maxima = pinpoint_peaks(profile, mask=prof_mask, peaks=max_locations,
                            halfwidth=3)
    minima = pinpoint_peaks(-profile, mask=prof_mask, peaks=min_locations,
                            halfwidth=3)
    extrema = sorted(zip(minima[0]+maxima[0], [-x for x in minima[1]]+maxima[1],
                         [False]*len(minima[0])+[True]*len(maxima[0])))

    # Now remove all adjacent minima with no maxima between them, and vice versa
    i = 0
    while i < len(extrema) - 1:
        if extrema[i][2] == extrema[i+1][2]:
            if (extrema[i][1] > extrema[i+1][1]) == extrema[i][2]:
                del extrema[i+1]
            else:
                del extrema[i]
        else:
            i += 1

    # These checks are required for the noiseless data in the tests
    if not extrema:
        return []

    if prof_mask is not None:
        # Find the first and last unmasked points in the profile
        l_ind = prof_mask.argmin()
        r_ind = prof_mask.size - prof_mask[::-1].argmin() - 1
    else:
        # If no prof_mask, just get the ends of the profile
        l_ind = 0
        r_ind = len(profile) - 1

    # If the first of last pixel of the profile has been marked as a maximum,
    # delete the first or last extremum (a maximum) as it doesn't represent a
    # peak we can do anything with, physically-speaking.
    if l_ind in max_locations:
        extrema = extrema[1:]
    if r_ind + 1 in max_locations:
        extrema = extrema[:-1]

    # Delete a maximum if there is no minimum between it and the edge,
    # unless it's the ONLY maximum
    if extrema[0][2]:
        if len(extrema) == 1:
            extrema = [(l_ind, profile[l_ind], False)] + extrema +\
                [(r_ind, profile[r_ind], False)]
        elif len(extrema) == 2 or not remove_edge_maxima:
            extrema = [(l_ind, profile[l_ind], False)] + extrema
        else:
            del extrema[0]
    if extrema and extrema[-1][2]:
        if len(extrema) == 2 or not remove_edge_maxima:
            extrema = extrema + [(r_ind, profile[r_ind], False)]
        else:
            del extrema[-1]

    if not extrema:
        return []

    # Now get rid of insignificant maxima
    stddev = at.std_from_pixel_variations(profile if prof_mask is None else
                                          profile[~prof_mask],
                                          subtract_linear_fits=True)

    def merge_with_left(index):
        del apertures[index]
        if extrema[index*2+1][1] > extrema[index*2-1][1]:
            del extrema[index*2-1:index*2+1]
        else:
            del extrema[index*2:index*2+2]

    def merge_with_right(index):
        del apertures[index]
        if extrema[index*2+1][1] > extrema[index*2+3][1]:
            del extrema[index*2+2:index*2+4]
        else:
            del extrema[index*2+1:index*2+3]

    def merge_with_neighbor(index):
        if extrema[index*2][1] < extrema[index*2+2][1] and index < len(apertures) - 1 or index == 0:
            merge_with_right(index)
            height = extrema[index * 2 + 1][1]
        else:
            merge_with_left(index)
            height = extrema[index*2-1][1]
        # Reset the search from the height of the new combined peak
        for i, _ in enumerate(apertures):
            if extrema[i*2+1][1] <= height:
                apertures[i] = 0

    # For all maxima below threshold sort by lowest value and merge as below,
    # then set value of all minima below threshold to threshold
    median = sigma_clipped_stats(profile, mask=prof_mask,
                                 sigma_lower=2, sigma_upper=1.5)[1]  # median

    # This next block was added later to the code, but has to shadow the variable
    # 'apertures' defined (further) below because of the way the three functions
    # above were written. This isn't looking for apertures, it's looking for maxima
    # less than the median value and merging them to get rid of them, before
    # setting any remaining minima below the median to the median. This removes
    # any negative apertures resulting from subtracting offset frames in NIR
    # observations, as the rest of the code isn't equipped to handle negative
    # values. DB 20240131
    apertures = [0] * (len(extrema) // 2)
    if not apertures:
        return []

    apnext = 1
    niter = 0
    while True:
        # Find highest unassigned peak lower than the threshold
        order = np.argsort([x[1] for x in extrema if x[2]])
        for lowest in order:
            if apertures[lowest] == 0:
                break
        l = 0 if lowest == 0 else apertures[lowest-1]
        r = 0 if lowest == len(apertures)-1 else apertures[lowest+1]
        if extrema[lowest*2+1][1] < median:
            if len(order) > 1:
                merge_with_neighbor(lowest)
            else:  # this is the only peak so we've not found anything
                return []
        else:
            apertures[lowest] = apnext
            apnext += 1
        if apertures.count(0) == 0:  # shouldn't happen here, but just in case
            break
        niter += 1
        if niter > 4 * profile.size:
            raise RuntimeError("Failed to converge in removing negative maxima")

    # Set value of minima < threshold to threshold
    extrema = [(e[0], median, e[2]) if (not e[2] and e[1] < median) else e
               for e in extrema]

    ### WE NEED TO GET RID OF THEM IN A SMARTER WAY. PERCOLATING?
    apertures = [0] * (len(extrema) // 2)

    apnext = 1
    niter = 0
    while True:
        # Find highest unassigned peak
        order = np.argsort([x[1] for x in extrema if x[2]])[::-1]
        for highest in order:
            if apertures[highest] == 0:
                break
        l = 0 if highest == 0 else apertures[highest-1]
        r = 0 if highest == len(apertures)-1 else apertures[highest+1]
        snr = _prominence(extrema, highest) / stddev
        if l == 0 and r == 0 or snr >= min_snr :  # new aperture
            apertures[highest] = apnext
            apnext += 1
        else:
            merge_with_neighbor(highest)
        if apertures.count(0) == 0:
            break
        niter += 1
        if niter > 4 * profile.size:
            raise ValueError("Failed to converge")

    return extrema


def _prominence(extrema, index):
    """
    Calculate the prominence of the index-th maximum. The prominence is
    defined as the height of the peak above the height of the highest adjacent
    minimum. This function is abstracted in case we want to change that.
    """
    return extrema[index*2+1][1] - max(extrema[index*2][1], extrema[index*2+2][1])


def find_apertures(ext, max_apertures, min_sky_region, percentile,
                   threshold, section, min_snr, use_snr,
                   max_separation=None, min_sep=3, profile=None):
    """
    Finds sources in 2D spectral images and compute aperture sizes. Used by
    findApertures as well as by the interactive code. Data MUST always be
    dispersed along the rows (i.e., GMOS orientation).

    An existing profile can be passed to avoid the need to recalculate it.

    Parameters
    ----------
    ext: NDAstroData-like
        the data, with the spatial direction vertical and the spectra
        dispersed horizontally
    max_apertures: int/None
        maximum number of apertures to return
    min_sky_region: int/None
        minimum separation between sky lines if the wavelength region is to
        be used to construct the profile
    percentile: float/None
        when constructing the profile, use this percentile for each row
        (if None, use the mean)
    threshold: float
        thresholding parameter between peak and continuum for determining the
        aperture size
    section: str
        pixel section in the wavelength direction for determining the profile
        (generally used if percentile=None to reproduce IRAF-like finding)
    min_snr: float
        minimum S/N ratio for acceptable apertures
    use_snr: bool
        if True, divide each pixel in the 2D image by its standard deviation
        in order to construct the profile (this helps to mitigate the effects
        of varying exposure times along the slit -- e.g., because of the GMOS
        longslit bridges -- when using percentile > 50)
    max_separation: float
        ignore sources found at a larger distance (in arcseconds) from the
        target than this
    min_sep: float
        merge sources closer than this separation (in pixels)
    profile: 1D ndarray/masked_array/None
        the 1D spatial profile can be provided; if not, it is calculated

    Returns
    -------
    locations: list
        pixel locations of all peaks, in order of decreasing S/N ratio
    all_limits: list of 2-tuples
        pixel locations of the upper and lower limits of the aperture
        of each peak
    profile: ndarray
        the spatial profile
    prof_mask: ndarray/None
        the mask
    """
    if profile is None:
        # Collapse image along spatial direction to find noisy regions
        # (caused by sky lines, regardless of whether image has been
        # sky-subtracted or not)
        profile, prof_mask = _construct_slit_profile(
            ext, min_sky_region, percentile, section, use_snr)
    elif hasattr(profile, 'mask'):
        prof_mask = profile.mask
        profile = profile.data
    else:
        prof_mask = None

    extrema = get_extrema(np.nan_to_num(profile), prof_mask, min_snr=min_snr,
                          remove_edge_maxima=False)

    # 10 is a good value to capture artifacts
    stddev = at.std_from_pixel_variations(profile if prof_mask is None
                                          else profile[~prof_mask], separation=10,
                                          subtract_linear_fits=True)
    peaks = [x[0] for x in extrema[1::2]]
    all_limits = get_limits(np.nan_to_num(profile), prof_mask, peaks=peaks,
                            threshold=threshold, extrema=extrema)
    fwhm_limits = get_limits(np.nan_to_num(profile), prof_mask, peaks=peaks,
                            threshold=0.5, extrema=extrema)

    snrs = [_prominence(extrema, i // 2) / stddev
            for i in range(1, len(extrema), 2)]

    # The code below here tries to remove things that aren't real sources,
    # either noise spikes or artifacts
    # Start by removing low-S/N apertures
    ok_apertures = {i: snr >= min_snr for i, snr in enumerate(snrs)}

    # Remove apertures too close to other apertures
    for i, (peak1, limit1) in enumerate(zip(peaks, all_limits)):
        for j, (peak2, limit2) in enumerate(list(zip(peaks, all_limits))[i+1:], start=i+1):
            if abs(peak1 - peak2) < min_sep or limit1[1] >= peak2 or limit2[0] <= peak1:
                if extrema[i*2+1] > extrema[j*2+1]:
                    ok_apertures[j] = False
                    all_limits[i] = (all_limits[i][0], all_limits[j][1])
                    fwhm_limits[i] = (fwhm_limits[i][0], fwhm_limits[j][1])
                    extrema[i*2+2] = extrema[j*2+2]
                else:
                    ok_apertures[i] = False
                    all_limits[j] = (all_limits[i][0], all_limits[j][1])
                    fwhm_limits[j] = (fwhm_limits[i][0], fwhm_limits[j][1])
                    extrema[j*2] = extrema[i*2]

    # Remove sources larger than a certain distance from the target coords
    if max_separation is not None:
        max_separation /= ext.pixel_scale()
        target_location = ext.wcs.invert(
            ext.central_wavelength(asNanometers=True), ext.target_ra(),
            ext.target_dec())[1]
        if not np.isnan(target_location):
            ok_apertures.update({i: False for i, x in enumerate(peaks)
                                 if abs(target_location - x) > max_separation})

    spline = at.fit_spline_to_data(profile, mask=prof_mask)
    for i, (peak, limits, flimits, snr) in enumerate(zip(peaks, all_limits, fwhm_limits, snrs)):
        # Remove apertures that are too wide
        for side in (0, 1):
            height = (extrema[i*2+1][1] - extrema[i*2+2*side][1]) / stddev
            width = abs(peak - limits[side])
            if width > height / min_snr * 20:
                ok_apertures[i] = False
            # Eliminate things with square edges that are likely artifacts
            elif (flimits[side] - peak) / (limits[side] - peak + 1e-6) > 0.85:
                # But keep them if the square edge butts up against another aperture
                if not ((i > 0 and limits[0] - all_limits[i-1][1] < 1) or
                        (i < len(peaks) - 1 and all_limits[i+1][0] - limits[1] < 1)):
                    ok_apertures[i] = False
        # Remove apertures that don't appear in a smoothed version of the
        # data (these are basically noise peaks)
        if spline(peak) - spline(limits).min() < stddev:
            ok_apertures[i] = False

    good_apertures = [info for i, info in enumerate(zip(peaks, all_limits, snrs))
                      if ok_apertures[i]]
    if not good_apertures:
        log.warning("Found no sources")
        locations, all_limits = [], []
    else:
        good_apertures = sorted(good_apertures, key=lambda ap: ap[2], reverse=True)
        locations = [ap[0] for ap in good_apertures[:max_apertures]]
        all_limits = [ap[1] for ap in good_apertures[:max_apertures]]
    return locations, all_limits, profile, prof_mask


@unpack_nddata
def find_wavelet_peaks(data, widths=None, mask=None, variance=None, min_snr=1, min_sep=3,
                       min_frac=0.20, reject_bad=True, pinpoint_index=-1):
    """
    Find peaks in a 1D array using a wavelet method. This uses scipy.signal
    routines, but requires some duplication of that code since the
    _filter_ridge_lines() function doesn't expose the *window_size* parameter,
    which is important. This also does some rejection based on a pixel mask
    and/or "forensic accounting" of relative peak heights.

    Parameters
    ----------
    data : 1D array
        The pixel values of the 1D spectrum
    widths : array-like
        Sigma values of line-like features to look for
    mask : 1D array (optional)
        Mask (peaks with bad pixels are rejected) - optional
    variance : 1D array (optional)
        Variance (to estimate SNR of peaks) - optional
    min_snr : float
        Minimum SNR required of peak pixel
    min_sep : float
        Minimum separation in pixels for lines in final list
    min_frac : float
        minimum fraction of *widths* values at which a peak must be found
    reject_bad : bool
        clip lines using the reject_bad() function?
    pinpoint_index : int / None
        which index (in the wavelet-transformed array, ordered by "widths")
        should be used for determining more the more accurate peak positions
        (None => use untransformed data)

    Returns
    -------
    2D array: peak pixels, peak values, and SNRs (sorted by pixel coordinate)
    """
    # Non-linear peaks are OK but saturated ones are not
    mask = ((mask & (DQ.max ^ DQ.saturated)).astype(bool) if mask is not None
            else np.zeros_like(data, dtype=bool))

    max_width = max(widths)
    window_size = 4 * max_width + 1

    # If no variance is supplied we estimate S/N from pixel-to-pixel variations
    # (do this before any smoothing)
    if variance is None:
        variance = at.std_from_pixel_variations(data[~mask],
                                                separation=int(max_width),
                                                subtract_linear_fits=True) ** 2

    # For really broad peaks we can do a median filter to remove spikes
    if min(widths) > 10:
        data = at.boxcar(data, size=2)
        mask = at.boxcar(mask, size=2, operation=np.logical_or)

    wavelet_transformed_data = cwt_ricker(data, widths)

    eps = np.finfo(np.float32).eps  # Minimum representative data
    wavelet_transformed_data[np.nan_to_num(wavelet_transformed_data) < eps] = eps

    ridge_lines = signal._peak_finding._identify_ridge_lines(
        wavelet_transformed_data, 0.03 * np.asarray(widths), 2)

    filtered = signal._peak_finding._filter_ridge_lines(
        wavelet_transformed_data, ridge_lines, window_size=window_size,
        min_length=int(min_frac * len(widths)), min_snr=min_snr)
    peaks = sorted([x[1][0] for x in filtered])

    # Estimate the SNR from the wavelet-transformed data to remove continuum
    snr = np.divide(wavelet_transformed_data[0], np.sqrt(variance),
                    out=np.zeros_like(data, dtype=np.float32),
                    where=variance > 0)

    peaks = [x for x in peaks if snr[x] > min_snr]

    # remove adjacent points
    new_peaks = []
    i = 0
    while i < len(peaks):
        j = i + 1
        while j < len(peaks) and (peaks[j] - peaks[j - 1] <= 1):
            j += 1
        new_peaks.append(np.mean(peaks[i:j]))
        i = j

    # Turn into array and remove those too close to the edges
    peaks = np.array(new_peaks)
    edge = 2.35482 * np.median(widths)
    peaks = peaks[np.logical_and(peaks > edge, peaks < len(data) - 1 - edge)]

    # Remove peaks very close to unilluminated/no-data pixels
    # (e.g., chip gaps in GMOS)
    peaks = [x for x in peaks if np.sum(mask[int(x-edge):int(x+edge+1)]) == 0]

    # We need to find accurate peak positions from convolved data so we're
    # not affected by noise or problems with broad, flat-topped features.
    # There appears to be a bias with narrow Ricker transforms, so we use the
    # broadest one for this purpose.
    pinpoint_data = (data if pinpoint_index is None else
                     wavelet_transformed_data[pinpoint_index])

    # Clip the really noisy parts of the data and get more accurate positions
    #pinpoint_data[snr < 0.5] = 0
    peaks, values = pinpoint_peaks(pinpoint_data, peaks=peaks, mask=mask,
                                   halfwidth=max(edge // 2, 2))

    # Clean up peaks that are too close together
    while True:
        diffs = np.diff(peaks)
        if all(diffs >= min_sep):
            break
        i = np.argmax(diffs < min_sep)
        # Replace with mean of re-pinpointed points
        new_peaks = pinpoint_peaks(pinpoint_data, peaks=peaks[i:i+2], mask=mask,
                                   halfwidth=edge//2+1)
        del peaks[i+1]
        del values[i+1]
        if new_peaks[0]:
            peaks[i] = np.mean(new_peaks[0])
            values[i] = np.mean(new_peaks[1])
        else:  # somehow both peaks vanished
            del peaks[i]
            del values[i]

    #final_peaks = [p for p in peaks if snr[int(p + 0.5)] > min_snr]
    final_peaks = peaks
    peak_snrs = list(snr[int(p + 0.5)] for p in final_peaks)

    # Remove suspiciously bright peaks and return as array of
    # locations and SNRs, sorted by location
    if reject_bad:
        good_peaks = reject_bad_peaks(list(zip(final_peaks, values, peak_snrs)))
    else:
        good_peaks = list(zip(final_peaks, values, peak_snrs))
    #print("KLDEBUG: T=", np.array(sorted(good_peaks)).T)

    # When no peaks are found the array is an empty list.  When called,
    # and peaks are found, two items (two lists) are returned and that is
    # what is expected.  Returning an empty crashes the calling code.
    # Return a list of two empty lists to prevent the crash and let the calling
    # routine decide what to do with it.

    T = np.array(sorted(good_peaks)).T
    if not T.size:
        T = np.array([[],[], []])
    return T


@unpack_nddata
def pinpoint_peaks(data, peaks=None, mask=None, halfwidth=None, threshold=None,
                   keep_bad=False, debug=False):
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
    mask: 1D array (bool)
        Mask (peaks with bad pixels are rejected)
    peaks: sequence
        approximate (but good) location of peaks
    halfwidth: int
        number of pixels either side of initial peak to use in centroid
    threshold: float
        threshold to cut data
    keep_bad: bool
        if True, keeps the output peak list the same size as the input list,
        with NaN values for the peaks that didn't converge

    Returns
    -------
    peaks: list of more accurate locations of the peaks that converged
          (may be shorter than the input list of peaks)
    values: list of the fitted peak values (same length as peaks)
    """
    if halfwidth is None:
        raise ValueError('halfwidth cannot be None')
    halfwidth = max(halfwidth, 2)  # Need at least 5 pixels to constrain spline
    int_limits = np.array([-1, -0.5, 0.5, 1])
    npts = len(data)
    final_peaks, peak_values = [], []

    if mask is None:
        mask = np.isnan(data)
    else:
        mask = mask.astype(bool) | np.isnan(data)
    if threshold is not None:
        mask |= data < threshold
    if data.size - mask.sum() < 4:
        return [], []

    xvalues = np.arange(data.size)

    for peak in peaks:
        xc = int(peak + 0.5)
        if xc < 0 or xc > data.size:
            if keep_bad:
                final_peaks.append(np.nan)
                peak_values.append(np.nan)
            continue
        xc = np.argmax(data[max(xc - 1, 0):min(xc + 2, npts)]) + xc - 1
        x1 = int(xc - halfwidth - 1)
        x2 = int(xc + halfwidth + 2)
        m = mask[x1:x2].astype(bool)
        if x1 < 0 or x2 > data.size - 1 or np.isnan(data[xc]) or np.sum(~m) < 4:
            if keep_bad:
                final_peaks.append(np.nan)
                peak_values.append(np.nan)
            continue
        data_min = data[x1:x2].min()
        data_snippet = data[x1:x2] - data_min
        # We fit splines to y(x) and x * y(x)
        t, c, k = interpolate.splrep(xvalues[x1:x2][~m], data_snippet[~m], k=3,
                                     s=0)
        spline1 = interpolate.BSpline.construct_fast(t, c, k, extrapolate=False)
        t, c, k = interpolate.splrep(xvalues[x1:x2][~m],
                                     (data_snippet * xvalues[x1:x2])[~m], k=3, s=0)
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
                peak_value = spline1(xc)
                break
            if abs(dx) > dxlast + 0.001:
                dxcheck += 1
                if dxcheck > 3:
                    break
            elif abs(dx) > dxlast - 0.001:
                xc -= 0.5 * max(-1, min(1, dx))
                dxcheck = 0
            else:
                dxcheck = 0
                dxlast = abs(dx)

        if final_peak is not None and not np.isnan(peak_value):
            final_peaks.append(final_peak)
            peak_values.append(peak_value + data_min)
        else:
            if keep_bad:
                final_peaks.append(np.nan)
                peak_values.append(np.nan)

        if debug:
            print("DEBUGGING", final_peaks, peak_values)
            fig, ax = plt.subplots()
            ax.plot(data, 'k-')
            ax.plot(np.arange(x1, x2), data[x1:x2])
            plt.show()

    return final_peaks, peak_values


def reject_bad_peaks(peaks):
    """
    Go through the list of peaks and remove any that look really suspicious.
    I refer to this colloquially as "forensic accounting".

    Parameters
    ----------
    peaks: sequence of 3-tuples:
        peak location, value, and "strength" (e.g., SNR) of peaks

    Returns
    -------
    sequence of 3-tuples:
        accepted elements of input list
    """
    diff = 3  # Compare 1st brightest to 4th brightest
    peaks.sort(key=lambda x: x[2])  # Sort by SNR
    while len(peaks) > diff and (peaks[-1][2] / peaks[-(diff + 1)][2] > 3):
        del peaks[-1]
    return peaks


@unpack_nddata
def get_limits(data, mask=None, variance=None, peaks=[], threshold=0, min_snr=3,
               extrema=None):
    """
    Determines the region in a 1D array associated with each already-identified
    peak.

    This function operates in one of two ways:
        1. A list of already-determined extrema (maxima and minima) can be
           provided, in which case each peak is shifted to the nearest
           maximum and the limits determined from the adjacent minima. This
           is the mode of operation during the findApertures() first pass.
        2. If not list is provided, extrema are calculated, each peak is
           shifted, and then extrema are recaculated based on the S/N of
           this peak, and the limits determined. This is the mode of operation
           if the user marks a new aperture in the findApertures() UI, which
           may be less significance than the nominal minimum S/N ratio.

    Limits are determined between the peak and the adjacent minimum, which is
    assumed to represent the continuum level (since we do not require that
    data have a continuum level of zero).

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
    threshold: float
        parameter that determines where to place the aperture edge
        between the peak and the continuum level
    min_snr: float
        minimum S/N ratio for detection (used to find minima if
        extrema is None)
    extrema: sequence/None
        output from get_extrema(); if not provided, extrema will be calculated
        again; if provided, it is assumed that they satisfy min_snr conditions

    Returns
    -------
    list of 2-element lists:
        the lower and upper limits for each peak
    """
    if extrema is None:
        extrema = get_extrema(data, mask, min_snr=0)
        niter = 1  # need to iterate to refind extrema given source's S/N
    else:
        niter = 0
    if variance is None:
        stddev = np.full_like(data, at.std_from_pixel_variations(
            data if mask is None else data[~mask], separation=10,
            subtract_linear_fits=True))
    else:
        stddev = np.sqrt(variance)

    all_limits = []
    for peak in peaks:
        for iter in range(niter+1):
            maxima = np.array([x[:2] for x in extrema if x[2]]).T
            i = np.argmin(abs(maxima[0] - peak)) * 2 + 1
            true_peak = extrema[i][0]
            if abs(true_peak - peak) > 2:
                log.warning(f'Difficulty finding peak near {peak:.2f} - continuing')
                true_peak = None
                break
            if iter == niter:
                break
            # Estimate S/N of this peak and find extrema with that
            # value if it's lower than min_snr
            this_snr = _prominence(extrema, i // 2) / stddev[int(true_peak+0.5)]
            extrema = get_extrema(data, mask, min_snr=min(this_snr, min_snr))

        if true_peak is None:
            continue

        lower, upper = extrema[i-1][0], extrema[i+1][0]
        targets = [threshold * extrema[i][1] +
                   (1 - threshold) * extrema[j][1] for j in (i-1, i+1)]
        # 0.999 is needed to avoid bringing in an extra pixel if the upper
        # limit is the last pixel (or unmasked pixel) in the data
        i1, i2, p = int(lower), int(upper+0.999), int(true_peak+0.5)

        limits = []
        for target, _slice in zip(targets, (slice(i1, p+2), slice(p-1, i2+1))):
            npts = _slice.stop - _slice.start
            # reduce variance to ensure spline goes through points
            with warnings.catch_warnings():
                # can arise from poor spline fit
                warnings.simplefilter("ignore", UserWarning)
                spline = at.fit_spline_to_data(
                    data[_slice], mask=None if mask is None else mask[_slice],
                    variance=0.01 * stddev[_slice]**2, k=1)

            limit = peak_limit(spline, true_peak-_slice.start,
                               0 if _slice.start==i1 else npts-1,
                               target) + _slice.start
            limits.append(limit)

        limit1, limit2 = min(limits[0], peak), max(limits[1], peak)
        all_limits.append((limit1, limit2))

    return all_limits


def peak_limit(spline, peak, limit, target):
    """
    Finds a threshold as a fraction of the way from the signal at the minimum to
    the signal at the peak.

    Parameters
    ----------
    spline : callable
        the function within which aperture-extraction limits are desired
    peak : float
        location of peak
    limit : float
        location of the minimum -- an aperture edge is required between
        this location and the peak
    target : float
        signal level where limit should be placed

    Returns
    -------
    float : the pixel location of the aperture edge
    """
    func = lambda x: spline(x) - target
    # We need to deal with the possibility that the spline may cross the
    # target level more than once, and we want the one closest to the peak
    #new_limit = peak
    #while func(new_limit) > 0:
    #    new_limit += 1
    try:
        return optimize.bisect(func, limit, peak)
    except ValueError:
        return limit


@insert_descriptor_values("dispersion_axis")
@unpack_nddata
def stack_slit(data, mask=None, percentile=50, section=slice(None), dispersion_axis=None):
    _slice = tuple([section if axis == dispersion_axis else slice(None)
                   for axis in range(data.ndim)])
    if mask is None:
        return np.percentile(data[_slice], percentile, axis=dispersion_axis)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice')
        profile = np.nanpercentile(np.where(mask[_slice], np.nan, data[_slice]),
                                   percentile, axis=dispersion_axis)
    return np.nan_to_num(profile, copy=False, nan=np.nanmedian(profile))


def _construct_slit_profile(ext, min_sky_region=50, percentile=80,
                            section=None, use_snr=True):
    """

    Parameters
    ----------
    ext: NDData-like object
        2D spectral image, with data dispersed along the rows
    min_sky_region: int/None
        minimum separation between sky lines for a region to be used in
        profile construction
    percentile: float
        percentile to take at each spatial element
    section: str
        specific sections in the wavelength direction for determining profile
    use_snr: bool
        divide data by sqrt(variance) first?

    Returns
    -------
    profile, prof_mask:
        1D profile and mask to be applied
    """
    _, _, var1d = NDStacker.mean(ext.data, mask=ext.mask,
                                 variance=ext.variance)

    # Mask sky-line regions and find clumps of unmasked pixels
    # Very light sigma-clipping to remove bright sky lines
    var_excess = var1d - at.boxcar(
        var1d, np.median, size=min_sky_region // 2 if min_sky_region else 25)

    # We need to construct a spatial profile along the slit. First, remove
    # columns where too few pixels are good
    if ext.mask is not None:
        mask1d = (np.sum(ext.mask == DQ.good, axis=0) < 0.25 * ext.shape[0])
    else:
        mask1d = np.zeros_like(var1d, dtype=bool)

    _, _, std = sigma_clipped_stats(var_excess, mask=mask1d, sigma=5.0,
                                    maxiters=3)
    mask1d |= (var_excess > 5 * std)
    slices = np.ma.clump_unmasked(np.ma.masked_array(var1d, mask1d))

    sky_mask = np.ones_like(mask1d, dtype=bool)
    if min_sky_region:
        for reg in slices:
            if (reg.stop - reg.start) >= min_sky_region:
                sky_mask[reg] = False
    # If nothing satisfies the min_sky_region requirement, ignore it
    if sky_mask.all():
        log.warning(f"No regions in {ext.filename} between sky lines exceed "
                    f"{min_sky_region} pixels. Ignoring requirement.")
        sky_mask[:] = True
        for reg in slices:
            sky_mask[reg] = False

    if section:
        sec_mask = np.ones_like(mask1d, dtype=bool)
        for x1, x2 in (s.split(':') for s in section.split(',')):
            reg = slice(None if x1 == '' else int(x1) - 1,
                        None if x2 == '' else int(x2))
            sec_mask[reg] = False
    else:
        sec_mask = False

    # Ensure we have some valid pixels left
    if (sky_mask | sec_mask).all():
        log.warning(f"No valid regions remain in {ext.filename} after "
                    "applying sections. Ignoring sky mask.")
        sky_mask[:] = False

    full_mask = (ext.mask > 0 if ext.mask is not None else
                 np.zeros(ext.shape, dtype=bool))
    full_mask |= sky_mask | sec_mask

    signal = (ext.data if (ext.variance is None or not use_snr) else
              np.divide(ext.data, np.sqrt(ext.variance),
                        out=np.zeros_like(ext.data), where=ext.variance > 0))
    if ext.variance is not None:
        full_mask |= ext.variance == 0
    masked_data = np.where(full_mask, np.nan, signal)
    prof_mask = np.bitwise_and.reduce(full_mask, axis=1)

    # Need to catch warnings for rows full of NaNs
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice')
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        if percentile:
            profile = np.nanpercentile(masked_data, percentile, axis=1)
        else:
            profile = np.nanmean(masked_data, axis=1)
    return profile, prof_mask
