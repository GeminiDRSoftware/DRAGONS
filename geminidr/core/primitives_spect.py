#
#                                                                  gemini_python
#
#                                                             primtives_spect.py
# ------------------------------------------------------------------------------
from geminidr import PrimitivesBASE
from . import parameters_spect

import numpy as np
from scipy import signal
from astropy.modeling import models

from matplotlib import pyplot as plt

from gempy.gemini import gemini_tools as gt
from gempy.library.nddops import NDStacker
from gempy.library import matching
from geminidr.gemini.lookups import DQ_definitions as DQ

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Spect(PrimitivesBASE):
    """
    This is the class containing all of the preprocessing primitives
    for the Spect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "SPECT"])

    def __init__(self, adinputs, **kwargs):
        super(Spect, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_spect)

    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        This primitive determines the wavelength solution

        Parameters
        ----------
        center: int/None
            central row/column for 1D extraction (None => use middle)
        nsum: int
            number of rows/columns to average
        order: int
            order of Chebyshev fitting function
        min_snr: float
            minimum S/N ratio in line peak to be used in fitting
        width: float
            expected width of arc lines in pixels
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        center = params["center"]
        nsum = params["nsum"]
        order = params["order"]
        min_snr = params["min_snr"]
        fwidth = params["fwidth"]
        arc_file = params["linelist"]

        # Get list of arc lines (probably from a text file dependent on the
        # input spectrum, so a private method of the primitivesClass)
        old_linelist = None
        if arc_file is not None:
            try:
                arc_lines = np.loadtxt(arc_file, usecols=[0]) * 0.1
            except (IOError, TypeError):
                log.warning("Cannot read file {} - using default linelist".format(arc_file))
                arc_file = None

        for ad in adinputs:
            for ext in ad:
                log.info("Determining wavelength solution for {}:{}".
                         format(ad.filename, ext.hdr['EXTVER']))
                # Determine direction of extraction for 2D spectrum
                if ext.data.ndim > 1:
                    slitaxis = ext.dispersion_axis() - 1
                    middle = 0.5 * ext.data.shape[slitaxis]
                    extract = slice(max(0, int((center or middle) - 0.5*nsum)),
                                  min(ext.data.shape[slitaxis], int((center or middle) + 0.5*nsum)))
                    if slitaxis == 1:
                        data = ext.data.T[extract]
                        mask = None if ext.mask is None else ext.mask.T[extract]
                        variance = None if ext.variance is None else ext.variance.T[extract]
                        direction = "column"
                    else:
                        data = ext.data[extract]
                        mask = None if ext.mask is None else ext.mask[extract]
                        variance = None if ext.variance is None else ext.variance[extract]
                        direction = "row"
                    log.info("Extracting 1D spectrum from {}s {} to {}".
                             format(direction, extract.start+1, extract.stop))
                else:
                    data = ext.data
                    mask = ext.mask
                    variance = ext.variance

                # Create 1D spectrum; pixel-to-pixel variation is a better indicator
                # of S/N than the VAR plane
                data, mask, variance = NDStacker.mean(data, mask=mask, variance=None)

                # Mask bad columns but not saturated/non-linear data points
                data[(mask & (65535 ^ (DQ.saturated | DQ.non_linear)))>0] = 0.

                cenwave = params["central_wavelength"] or ext.central_wavelength(asNanometers=True)
                dw = params["dispersion"] or ext.dispersion(asNanometers=True)
                max_allowed_rms = 3. * abs(dw)

                # Don't read linelist if it's the one we already have
                # (For user-supplied, we read it at the start, so don't do this at all)
                if arc_file is None:
                    linelist = self._get_linelist_filename(ext, cenwave, dw)
                    if linelist != old_linelist:
                        arc_lines = np.loadtxt(linelist, usecols=[0]) * 0.1
                    old_linelist = linelist

                # Find peaks; convert width FWHM to sigma
                widths = 0.42466*fwidth * np.arange(0.8,1.21,0.05)  # TODO!
                peaks = _find_peaks(data, variance, widths, min_snr=min_snr)

                log.fullinfo('{}:{} {} peaks and {} arc lines'.
                             format(ad.filename, ext.hdr['EXTVER'], len(peaks), len(arc_lines)))

                # Loop to attempt to reject bad fits
                for kditer in np.arange(22):
                    # Choice of kdsigma can have a big effect. This oscillates
                    # around the initial choice, with increasing amplitude.
                    # Final iteration reverts to original choice (because we
                    # haven't found a decent fit)
                    kdsigma = 10.*dw * (((1.0+0.1*((kditer+1)//2)))**((-1)**kditer)
                                        if kditer<21 else 1)

                    # Do initial fit: start with linear, then increase order
                    for ord in range(1, order+1):
                        m_init = models.Chebyshev1D(degree=ord, domain=[0, len(data)-1])
                        if ord == 1:
                            m_init.c0 = cenwave
                            m_init.c0.bounds = (m_init.c0-30*abs(dw), m_init.c0+30*abs(dw))
                            m_init.c1 = 0.5*dw*len(data)
                            c1_bounds = (0.475*dw*len(data), 0.525*dw*len(data))
                        else:
                            m_init.c0 = m_final.c0
                            m_init.c0.bounds = (m_init.c0-10*abs(dw), m_init.c0+10*abs(dw))
                            m_init.c1 = m_final.c1
                            c1_bounds = (0.98*m_final.c1, 1.02*m_final.c1)
                        # Needed to cope with dw<0
                        m_init.c1.bounds = (min(c1_bounds), max(c1_bounds))
                        for i in range(2, ord+1):
                            setattr(m_init, 'c{}'.format(i), getattr(m_final, 'c{}'.format(i), 1))
                            getattr(m_init, 'c{}'.format(i)).bounds = (-2, 2)

                        fit_it = matching.KDTreeFitter()
                        m_final = fit_it(m_init, peaks, arc_lines, method='Nelder-Mead',
                                         sigma=kdsigma,
                                         options={'xtol': 1.0e-6, 'ftol': 1.0e-8})

                        log.debug('{} {}'.format(repr(m_final), fit_it.statistic))

                    # Need to remove bounds from model as LevMarLSQFitter can't deal
                    for p in m_final.param_names:
                        getattr(m_final, p).bounds = (None, None)

                    # Match peaks to arc lines and iterative 3-sigma clip
                    match_radius = 2*fwidth*abs(m_final.c1) / len(data)  # fwidth pixels
                    num_matches = 0
                    while True:
                        matched = matching.match_sources(m_final(peaks), arc_lines,
                                                         radius=match_radius)
                        for i, m in enumerate(matched):
                            if m > -1:
                                log.debug("{:3d} {:8.2f} {:9.4f} {:9.4f}".
                                          format(i, peaks[i], m_final(peaks[i]), arc_lines[m]))
                        incoords, outcoords = zip(*[(peaks[i], arc_lines[m])
                                                    for i, m in enumerate(matched) if m > -1])
                        m = matching.Chebyshev1DMatchBox(incoords, outcoords,
                                                         forward_model=m_final)
                        m.fit_forward()
                        rms = m.rms_output
                        m_final = m.forward
                        match_radius = 3.0*rms
                        if len(incoords) == num_matches:
                            break
                        num_matches = len(incoords)

                    if rms < max_allowed_rms:
                        break
                        log.debug("Matched {} lines with rms = {:.3f} nm. "
                                  "Re-trying.".format(len(incoords), rms))
                else:
                    log.warning("{}:{} Failed to find a fit with an acceptable "
                                "rms.".format(ad.filename, ext.hdr['EXTVER']))

                log.stdinfo(m_final)
                log.stdinfo("Matched {} lines with rms = {:.3f} nm.".format(len(incoords), rms))

                # Some diagnostic plotting
                fig, ax = plt.subplots()
                ax.plot(m_final(np.arange(len(data))), 20 * data / np.max(data))
                # ax.plot(m_final(np.arange(len(data))), np.sqrt(variance), 'r-')
                cwt_dat = signal.cwt(data, signal.ricker, widths)[0]
                cwt_dat[cwt_dat<0] = 0.
                ax.plot(m_final(np.arange(len(data))), 20*cwt_dat / np.max(data), 'g-')
                ax.plot(arc_lines, [0.01] * len(arc_lines), 'ro')
                ax.plot(m_final(peaks), [0]*len(peaks), 'bo')
                ax.plot(m_final(incoords), [0]*len(incoords), 'go')
                # ax.plot(matched_line_pixels, [-0.05]*len(matched_line_pixels), 'go')
                limits = (m_final(0), m_final(len(data)-1))
                ax.set_xlim(min(limits), max(limits))
                ax.set_ylim(-0.1, 1.1)

                m.display_fit()

                plt.show()

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs


def _find_peaks(data, variance, widths, min_snr=5., min_frac=0.25):
    # Need to recode signal.find_peaks.cwt since some important parameters
    # aren't exposed, notably window_size
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
    snr = np.divide(cwt_dat[0], np.sqrt(variance), np.zeros_like(data), where=variance>0)
    peaks = [x for x in peaks if snr[x]>min_snr]

    #peaks = signal.find_peaks_cwt(data, widths, min_snr=2., min_length=int(min_frac*len(widths)))
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

    # Improve positions of peaks with centroiding. Use a deliberately small
    # centroiding box to avoid contamination by nearby lines
    #data[snr<2] = 0
    data = np.where(snr<2, 0, data)  # To keep data intact
    for i, peak in enumerate(peaks):
        x1 = int(peak - maxwidth)
        x2 = int(peak + maxwidth+1)
        peaks[i] = np.sum(data[x1:x2] * np.arange(x1, x2)) / np.sum(data[x1:x2])

    return np.array(peaks)
