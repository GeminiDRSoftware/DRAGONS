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


