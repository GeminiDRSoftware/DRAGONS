#
#                                                                  gemini_python
#
#                                                             primtives_spect.py
# ------------------------------------------------------------------------------
from geminidr import PrimitivesBASE
from . import parameters_spect

import numpy as np
from scipy import signal
from astropy.modeling import models, fitting

from matplotlib import pyplot as plt

from gempy.library.nddops import NDStacker
from gempy.library.matching import MatchBox, fit_brute_then_simplex, match_sources

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
        """
        log = self.log
        center = params["center"]
        nsum = params["nsum"]
        order = params["order"]

        no_grid_search = False
        arc_lines = self._get_arc_lines()

        for ad in adinputs:
            for ext in ad:
                if ext.data.ndim > 1:
                    slitaxis = ext.dispersion_axis() - 1
                    middle = 0.5 * ext.data.shape[slitaxis]
                    extract = slice(max(0, int((middle if center is None else center) - 0.5*nsum)),
                                  min(ext.data.shape[slitaxis], int((middle if center is None else center) + 0.5*nsum)))
                    if slitaxis == 1:
                        data = ext.data.T[extract]
                        mask = None if ext.mask is None else ext.mask.T[extract]
                        direction = "row"
                    else:
                        data = ext.data[extract]
                        mask = None if ext.mask is None else ext.mask[extract]
                        direction = "column"
                    log.info("Extracting 1D spectrum from {}s {} to {}".
                             format(direction, extract.start+1, extract.stop))
                else:
                    data = ext.data
                    mask = ext.mask

                # Create 1D spectrum
                data, mask, variance = NDStacker.mean(data, mask=mask, variance=None)
                data /= np.max(data)  # For ease of plotting
                # Bad columns
                data[1916] = 0.
                data[2822:2825] = 0.

                # Find peaks
                widths = 2.5 * np.arange(0.8,1.21,0.05)  # TODO!
                peaks = _find_peaks(data, widths)
                print peaks

                print len(peaks), len(arc_lines)

                # Do initial fit
                cenwave = params["central_wavelength"] or ext.central_wavelength(asAngstroms=True)
                dw = params["dispersion"] or ext.dispersion(asAngstroms=True)

                # Define a domain 10% larger than expected in case initial
                # estimates are poor
                domain_size = 1.1 * dw * (len(data)-1)
                m_init = models.Chebyshev1D(degree=order,
                                            domain=[cenwave-0.5*domain_size,
                                                    cenwave+0.5*domain_size])
                m_init.c0 = 0.5*len(data)
                m_init.c0.bounds = (0.4*len(data), 0.6*len(data))
                m_init.c1 = 0.55*len(data)
                m_init.c1.bounds = (0.5*len(data), 0.6*len(data))
                if no_grid_search:
                    m_init.c0.fixed = True
                    m_init.c1.fixed = True
                for i in range(2, order+1):
                    getattr(m_init, 'c{}'.format(i)).fixed = True
                    getattr(m_init, 'c{}'.format(i)).bounds = (-20, 20)

                m_final = fit_brute_then_simplex(m_init, arc_lines, peaks,
                                     sigma=5.0, unfix=True, unbound=False)

                matches = match_sources((m_final(arc_lines),), (peaks,), radius=3.0)
                matched_line_wavelengths = arc_lines[matches>-1]
                matched_line_pixels = [peaks[i] for i in matches if i>-1]
                #print matched_line_pixels
                #print matched_line_wavelengths
                for w, x in zip(arc_lines, m_final(arc_lines)):
                    print w, x

                w = np.arange(4000.,9500.)
                x = m_final(w)
                fig, ax = plt.subplots()
                ax.plot(x[:-1], 1./np.diff(x))
                plt.show()

                # Some diagnostic plotting
                fig, ax = plt.subplots()
                ax.plot(np.arange(len(data)), 20*data/np.max(data))
                ax.plot(m_final(arc_lines), [0]*len(arc_lines), 'ro')
                ax.plot(peaks, [-0.05]*len(peaks), 'bo')
                ax.plot(matched_line_pixels, [-0.05]*len(matched_line_pixels), 'go')
                ax.set_xlim(0, len(data))
                ax.set_ylim(-0.1,1.1)
                plt.show()

        return adinputs

def _find_peaks(data, widths, min_snr=2, min_frac=0.5):
    peaks = list(signal.find_peaks_cwt(data, widths, min_snr=min_snr,
                                       min_length=int(min_frac * len(widths))))
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
    edge = 2.35482 * max(widths)

    # Turn into array and remove those too close to the edges
    peaks = np.array(peaks)
    peaks = peaks[np.logical_and(peaks>edge, peaks<len(data)-edge)]
    return peaks
