"""
tracing.py

This module contains function used to locate peaks in 1D data and trace them
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
from scipy import signal, interpolate
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.library.nddops import NDStacker
from gempy.utils import logutils

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
        maximum plausible peak width

    Returns
    -------
    float: estimate of FWHM of features
    """
    all_widths = []
    for fwidth in range(min, max+1):  # plausible range of widths
        data_copy = data.copy()  # We'll be editing the data
        widths = []
        for i in range(15):  # 15 brightest peaks, should ensure we get real ones
            index = 2*fwidth + np.argmax(data_copy[2*fwidth:-2*fwidth-1])
            data_to_fit = data_copy[index - 2 * fwidth:index + 2 * fwidth + 1]
            m_init = models.Gaussian1D(stddev=0.42466*fwidth) + models.Const1D(np.min(data_to_fit))
            m_init.mean_0.bounds = [-1, 1]
            m_init.amplitude_1.fixed = True
            fit_it = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(),
                                                       sigma_clip, sigma=3)
            with warnings.catch_warnings():
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                m_final, _ = fit_it(m_init, np.arange(-2*fwidth, 2*fwidth+1),
                                    data_to_fit)
            # Quick'n'dirty logic to remove "peaks" at edges of CCDs
            if m_final.amplitude_1 != 0:
                widths.append(m_final.stddev_0/0.42466)

            # Set data to zero so no peak is found here
            data_copy[index-2*fwidth:index+2*fwidth+1] = 0.
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
    mask: 1D array
        Mask (peaks with bad pixels are rejected)
    variance: 1D array
        Variance (to estimate SNR of peaks)
    min_snr: float
        Minimum SNR required of peak pixel
    min_frac: float
        minimum number of *widths* values at which a peak must be found
    rank_clip: bool
        clip brightest lines based on relative SNR?

    Returns
    -------
    2D array: peak wavelengths and SNRs
    """
    maxwidth = max(widths)
    window_size = 4*maxwidth+1
    cwt_dat = signal.cwt(data, signal.ricker, widths)
    eps = np.finfo(np.float32).eps
    cwt_dat[cwt_dat<eps] = eps
    ridge_lines = signal._peak_finding._identify_ridge_lines(cwt_dat, 0.25*widths, 2)
    filtered = signal._peak_finding._filter_ridge_lines(cwt_dat, ridge_lines,
                                                        window_size=window_size,
                                                        min_length=int(min_frac*len(widths)),
                                                        min_snr=1.)
    peaks = sorted([x[1][0] for x in filtered])

    # If no variance is supplied we take the "snr" to be the data
    if variance is not None:
        snr = np.divide(cwt_dat[0], np.sqrt(variance), np.zeros_like(data),
                        where=variance>0)
    else:
        snr = cwt_dat[0]
    peaks = [x for x in peaks if snr[x] > min_snr]

    # remove adjacent points
    while True:
        new_peaks = peaks
        for i in range(len(peaks)-1):
            if peaks[i+1]-peaks[i] == 1:
                new_peaks[i] += 0.5
                new_peaks[i+1] = -1
        new_peaks = [x for x in new_peaks if x>-1]
        if len(new_peaks) == len(peaks):
            break
        peaks = new_peaks

    # Turn into array and remove those too close to the edges
    peaks = np.array(peaks)
    edge = 2.35482 * maxwidth
    peaks = peaks[np.logical_and(peaks>edge, peaks<len(data)-1-edge)]

    # Clip the really noisy parts of the data before getting more accurate
    # peak positions
    final_peaks = pinpoint_peaks(np.where(snr<0.5, 0, data), mask, peaks)
    peak_snrs = list(snr[int(p+0.5)] for p in final_peaks)

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
    masked_data = np.where(np.logical_or(mask > 0, data < threshold), 0,
                           data - threshold)
    for peak in peaks:
        xc = int(peak + 0.5)
        xc = np.argmax(masked_data[max(xc-1,0):min(xc+2,npts)]) + xc - 1
        if xc < halfwidth or xc > npts - halfwidth:
            continue
        x1 = int(xc - halfwidth)
        x2 = int(xc + halfwidth + 1)
        xvalues = np.arange(x1, x2)
        t, c, k = interpolate.splrep(xvalues, masked_data[x1:x2], k=3,
                                     s=0)
        spline1 = interpolate.BSpline.construct_fast(t, c, k, extrapolate=False)
        t, c, k = interpolate.splrep(xvalues, masked_data[x1:x2] * xvalues,
                                     k=3, s=0)
        spline2 = interpolate.BSpline.construct_fast(t, c, k, extrapolate=False)

        final_peak = None
        dxlast = x2 - x1
        dxcheck = 0
        for i in range(50):
            # Triangle centering function
            limits = xc + int_limits * halfwidth
            splint1 = [spline1.integrate(x1, x2) for x1, x2 in zip(limits[:-1], limits[1:])]
            splint2 = [spline2.integrate(x1, x2) for x1, x2 in zip(limits[:-1], limits[1:])]
            sum1 = (splint2[1] - splint2[0] - splint2[2] +
                    (xc-halfwidth) * splint1[0] - xc*splint1[1] + (xc+halfwidth)*splint1[2])
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
    while len(peaks) > diff and (peaks[-1][1] / peaks[-(diff+1)][1] > 3):
        del peaks[-1]
    return peaks

################################################################################
# FUNCTIONS RELATED TO PEAK-TRACING

def trace_lines(ext, axis, start=None, initial=None, width=5, nsum=10,
                step=1, max_shift=0.05, max_missed=10, viewer=None):
    """

    Parameters
    ----------
    ext: single-sliced AD object
        the extension within which to trace features
    axis: int (0 or 1)
        axis along which to trace (0=y-direction, 1=x-direction)
    start: int/None
        row/column to start trace (None => middle)
    initial: sequence
        coordinates of peaks
    width: int
        width of centroiding box in pixels
    nsum: int
        number of rows/columns to combine at each step
    step: int
        step size along axis in pixels
    max_shift: float
        maximum perpendicular shift from pixel to pixel
    max_missed: int
        maximum number of interations without finding line before
        line is considered lost forever
    viewer: imexam viewer/None
        viewer to draw lines on
    """
    log = logutils.get_logger(__name__)

    # We really don't care about non-linear/saturated pixels
    bad_bits = 65535 ^ (DQ.non_linear | DQ.saturated)

    halfwidth = int(0.5 * width)

    # Make life easier by always tracing along columns
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
        y1 = int(start - 0.5*nsum + 0.5)
        data, mask, var = NDStacker.mean(ext_data[y1:y1 + nsum],
                                         mask=ext_mask[y1:y1 + nsum],
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
            y1 = int(ypos - 0.5*nsum + 0.5)
            data, mask, var = NDStacker.mean(ext_data[y1:y1+nsum],
                                             mask=ext_mask[y1:y1+nsum],
                                             variance=None)
            clipped_data = np.where(data/np.sqrt(var) > 0.5, data, 0)
            last_peaks = [c[1] for c in last_coords if not np.isnan(c[1])]
            peaks = pinpoint_peaks(clipped_data, mask, last_peaks)
            #if ypos == start:
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
                tolerance = 1.0 if ypos == start else max_shift * abs(ypos - last_row)
                if (abs(new_peak - old_peak) > tolerance):
                    # If it's gone for good, set the coord to NaN to avoid it
                    # picking up a different line if there's significant tilt
                    if abs(ypos - last_row) > max_missed * step:
                        last_coords[i][1] = np.nan
                    continue

                # Too close to the edge?
                if (new_peak < halfwidth or
                    new_peak > ext_data.shape[1] - 0.5*halfwidth):
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
            if ypos < 0.5 * nsum or ypos > ext_data.shape[0] - 0.5*nsum:
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