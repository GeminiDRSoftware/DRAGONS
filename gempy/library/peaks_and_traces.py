"""
peaks_and_traces.py

This module contains function used to locate peaks in 1D data and trace them
in the orthogonal direction in a 2D image.

Functions in this module:
find_peaks:       locate peaks using the scipy.signal routine
pinpoint_peaks:   produce more accuate positions for existing peaks by
                  centroiding
reject_bad_peaks: remove suspicious-looking peaks by a variety of methods
"""
import numpy as np
from scipy import signal

from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.library.nddops import NDStacker
from gempy.utils import logutils

################################################################################
# FUNCTIONS RELATED TO PEAK-FINDING

def find_peaks(data, widths, mask=None, variance=None, min_snr=1, min_frac=0.25,
               rank_clip=True):
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
    good_peaks = reject_bad_peaks(list(zip(final_peaks, peak_snrs)))
    return np.array(sorted(good_peaks)).T

def pinpoint_peaks(data, mask, peaks, halfwidth=2):
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

    Returns
    -------
    list: more accurate locations of the peaks that are unaffected by the mask
    """
    final_peaks = []
    for peak in peaks:
        x1 = int(peak - halfwidth)
        x2 = int(peak + halfwidth+1)
        if np.sum(mask[x1:x2]) == 0:
            final_peaks.append(np.sum(data[x1:x2] * np.arange(x1,x2)) / np.sum(data[x1:x2]))
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

def trace_lines(ext, axis, start=None, initial=None, width=3, nsum=10,
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
        start = int(0.5 * ext.shape[0])
        log.stdinfo("Starting trace at {} {}".format(direction, start))

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

            for i, (last_row, old_peak) in enumerate(last_coords):
                if np.isnan(old_peak):
                    continue
                j = np.argmin(abs(np.array(peaks) - old_peak))
                new_peak = peaks[j]

                # Is this close enough to the existing peak?
                if abs(new_peak - old_peak) > max_shift * abs(ypos - last_row):
                    # If it's gone for good, set the coord to NaN to avoid it
                    # picking up a different line if there's significant tilt
                    if abs(ypos - last_row) > max_missed * step:
                        last_coords[i][1] = np.nan
                    continue

                # Too close to the edge?
                if new_peak < halfwidth or new_peak > ext_data.shape[1] - 0.5*halfwidth:
                    last_coords[i][1] = np.nan
                    continue

                new_coord = [ypos, new_peak]
                if viewer:
                    kwargs = dict(zip(('y1', 'x1'), last_coords[i]))
                    kwargs.update(dict(zip(('y2', 'x2'), new_coord)))
                    viewer.line(origin=0, **kwargs)

                coord_lists[i].append(new_coord)
                last_coords[i] = new_coord

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

    # Return the coordinate lists, but revert to the original axis order
    return (ref_coords, in_coords) if axis == 0 else (ref_coords[::-1], in_coords[::-1])