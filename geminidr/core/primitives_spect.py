#
#                                                                  gemini_python
#
#                                                             primtives_spect.py
# ------------------------------------------------------------------------------
from geminidr import PrimitivesBASE
from . import parameters_spect

import numpy as np
from scipy import signal, spatial
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.table import Table

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
        suffix: str
            suffix to be added to output files
        center: int/None
            central row/column for 1D extraction (None => use middle)
        nsum: int
            number of rows/columns to average
        order: int
            order of Chebyshev fitting function
        min_snr: float
            minimum S/N ratio in line peak to be used in fitting
        fwidth: float
            expected width of arc lines in pixels
        linelist: str/None
            name of file containing arc lines
        weighting: str (none/natural/relative)
            how to weight the detected peaks
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
        weighting = params["weighting"]

        # TODO: This decision would prevent MOS data being reduced so need
        # to think a bite more about what we're going to do. Maybe make
        # central_wavelength() return a one-per-ext list? Or have the GMOS
        # determineWavelengthSolution() recipe check the input has been
        # mosaicked before calling super()?
        #
        # Top-level decision for this to only work on single-extension ADs
        #if not all(len(ad)==1 for ad in adinputs):
        #    raise ValueError("Not all inputs are single-extension AD objects")

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
                log.info("Determining wavelength solution for {}".format(ad.filename))
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
                    log.stdinfo("Extracting 1D spectrum from {}s {} to {}".
                                format(direction, extract.start+1, extract.stop))
                else:
                    data = ext.data
                    mask = ext.mask
                    variance = ext.variance

                # Create 1D spectrum; pixel-to-pixel variation is a better indicator
                # of S/N than the VAR plane
                data, mask, variance = NDStacker.mean(data, mask=mask, variance=None)

                # Mask bad columns but not saturated/non-linear data points
                mask &= 65535 ^ (DQ.saturated | DQ.non_linear)
                data[mask>0] = 0.

                cenwave = params["central_wavelength"] or ext.central_wavelength(asNanometers=True)
                dw = params["dispersion"] or ext.dispersion(asNanometers=True)
                log.stdinfo("Using central wavelength {:.1f} nm and dispersion "
                            "{:.3f} nm/pixel".format(cenwave, dw))

                fwidth = _estimate_peak_width(data.copy(), fwidth)
                log.stdinfo("Estimated feature width: {:.2f} pixels".format(fwidth))

                # Don't read linelist if it's the one we already have
                # (For user-supplied, we read it at the start, so don't do this at all)
                if arc_file is None:
                    linelist = self._get_linelist_filename(ext, cenwave, dw)
                    # TODO: Convert linelist to nm
                    if linelist != old_linelist:
                        arc_lines = np.loadtxt(linelist, usecols=[0]) * 0.1
                    old_linelist = linelist

                # Find peaks; convert width FWHM to sigma
                widths = 0.42466*fwidth * np.arange(0.8,1.21,0.05)  # TODO!
                peaks, peak_snrs = _find_peaks(data, widths, mask=mask,
                                               variance=variance, min_snr=min_snr)
                log.stdinfo('{}: {} peaks and {} arc lines'.
                             format(ad.filename, len(peaks), len(arc_lines)))

                if weighting == "none":
                    in_weights = np.ones((len(peaks),))
                elif weighting == "natural":
                    in_weights = peak_snrs
                elif weighting == "relative":
                    tree = spatial.cKDTree(np.array([peaks]).T)
                    # Find lines within 10% of the array size
                    indices = tree.query(np.array([peaks]).T, k=10,
                                         distance_upper_bound=abs(0.1*len(data)*dw))[1]
                    snrs = np.array(list(peak_snrs) + [np.nan])[indices]
                    # Normalize weights by the maximum of these lines
                    in_weights = peak_snrs / np.nanmax(snrs, axis=1)

                # Construct a new array of data to plot
                #x = np.arange(len(data))
                #y = data.copy()
                #x[mask>0] = -1
                #for peak in peaks:
                #    x[int(peak-fwidth):int(peak+fwidth+1)] = -1
                #for i in (0,1):
                #    y = y[x>-1]
                #   x = x[x>-1]
                #    continuum = interpolate.LSQUnivariateSpline(x, y, np.linspace(0, len(data), 10)[1:-1])
                #    x[y-continuum(x)>0.1*max(data)] = -1
                #plot_data = data - continuum(np.arange(len(data)))
                plot_data = data

                # Some diagnostic plotting
                yplot = 0
                fig, ax = plt.subplots()

                init_order = 1
                ord = init_order
                m_init = models.Chebyshev1D(degree=ord, c0=cenwave,
                                            c1=0.5*dw*len(data), domain=[0, len(data)-1])
                kdsigma = 3*fwidth*abs(dw)
                #min_snr = 1
                while (ord <= order) or (kdsigma > fwidth*abs(dw)):
                    ord = min(ord, order)
                    peaks_to_fit = peaks[peak_snrs>min_snr]
                    m_init = _set_model(m_init, order=int(ord), initial=(ord==1), kdfit=True)
                    if ord == init_order:
                        plot_arc_fit(data, peaks, arc_lines, m_init, "Initial model")
                        log.stdinfo('Initial model: {}'.format(repr(m_init)))
                    fit_it = matching.KDTreeFitter()
                    m_final = fit_it(m_init, peaks_to_fit, arc_lines, maxsig=10, k=1,
                                     in_weights=in_weights[peak_snrs>min_snr], ref_weights=None,
                                     sigma=kdsigma, method='Nelder-Mead',
                                     options={'xtol': 1.0e-7, 'ftol': 1.0e-8})
                    log.stdinfo('{} {}'.format(repr(m_final), fit_it.statistic))
                    plot_arc_fit(plot_data, peaks, arc_lines, m_final, "KDFit model order {} KDsigma = {}".format(ord, kdsigma))

                    kdsigma = fwidth * abs(dw)
                    if ord < order:
                        ord += 1
                        yplot += 1
                        m_init = m_final
                        continue

                    # Remove bounds from the model
                    m_final._constraints['bounds'] = {p: (None, None)
                                                      for p in m_final.param_names}
                    match_radius = 2*fwidth*abs(m_final.c1) / len(data)  # fwidth pixels
                    #match_radius = kdsigma
                    m = matching.Chebyshev1DMatchBox.create_from_kdfit(peaks, arc_lines,
                                    model=m_final, match_radius=match_radius, sigma_clip=5)
                    for incoord, outcoord in zip(m.forward(m.input_coords), m.output_coords):
                       ax.text(incoord, yplot, '{:.4f}'.format(outcoord), rotation=90,
                               ha='center', va='top')

                    log.stdinfo('{} {} {}'.format(repr(m.forward), len(m.input_coords), m.rms_output))
                    plot_arc_fit(plot_data, peaks, arc_lines, m.forward, "MatchBox model order {}".format(ord))

                    # Choice of kdsigma can have a big effect. This oscillates
                    # around the initial choice, with increasing amplitude.
                    #kdsigma = 10.*abs(dw) * (((1.0+0.1*((kditer+1)//2)))**((-1)**kditer)
                    #                    if kditer<21 else 1)

                    kdsigma *= 0.5
                    ord += 1
                    m_init = m.forward

                m_final = m.forward
                rms = m.rms_output
                nmatched = len(m.input_coords)
                log.stdinfo(m_final)
                log.stdinfo("Matched {} lines with rms = {:.3f} nm.".format(nmatched, rms))

                plot_arc_fit(plot_data, peaks, arc_lines, m_final, "Final fit")

                m.display_fit()
                plt.show()

                m.sort()
                incoords = m.input_coords
                outcoords = m.output_coords
                coeff_column = (list(m_final.domain) + [order] +
                                list(getattr(m_final, 'c{}'.format(i)).value for i in range(order+1)))
                # Ensure all columns have the same length
                if len(coeff_column) <= nmatched:
                    coeff_column += [0] * (nmatched - len(coeff_column))
                else:  # Really shouldn't be the case
                    incoords += [0] * (len(coeff_column) - nmatched)
                    outcoords += [0] * (len(coeff_column) - nmatched)

                fit_table = Table([coeff_column, incoords, outcoords],
                                  names=("coefficients", "peaks", "wavelengths"))
                ext.FITTABLE = fit_table

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs


def _estimate_peak_width(data, fwidth):
    """
    Estimates the FWHM of the spectral features (arc lines) by fitting
    Gaussians to the brightest peaks.

    Parameters
    ----------
    data:  ndarray
        1D data array (will be modified)
    fwidth: float
        Estimated FWHM of features

    Returns
    -------
    float: Better estimate of FWHM of features
    """
    fwidth = int(fwidth+0.5)
    widths = []
    for i in range(15):
        index = 2*fwidth + np.argmax(data[2*fwidth:-2*fwidth-1])
        data_to_fit = data[index - 2 * fwidth:index + 2 * fwidth + 1]
        m_init = models.Gaussian1D(stddev=0.42466*fwidth) + models.Const1D(np.min(data_to_fit))
        m_init.mean_0.fixed = True
        m_init.amplitude_1.fixed = True
        fit_it = fitting.LevMarLSQFitter()
        m_final = fit_it(m_init, np.arange(-2*fwidth, 2*fwidth+1),
                         data_to_fit)
        #print (index, m_final)
        # Quick'n'dirty logic to remove "peaks" at edges of CCDs
        if m_final.amplitude_1 != 0:
            widths.append(m_final.stddev_0/0.42466)
        data[index-2*fwidth:index+2*fwidth+1] = 0.
    return sigma_clip(widths).mean()

def _find_peaks(data, widths, mask=None, variance=None, min_snr=1, min_frac=0.25,
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
    snr = np.divide(cwt_dat[0], np.sqrt(variance), np.zeros_like(data), where=variance>0)
    peaks = [x for x in peaks if snr[x]>min_snr]

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
    # Remove any peaks affected by the mask
    clipped_data = np.where(snr<0.5, 0, data)
    final_peaks = []
    for peak in peaks:
        x1 = int(peak - maxwidth)
        x2 = int(peak + maxwidth+1)
        if np.sum(mask[x1:x2]) == 0:
            if max(snr[x1:x2] >= min_snr):
                final_peaks.append(np.sum(clipped_data[x1:x2] * np.arange(x1,x2)) / np.sum(clipped_data[x1:x2]))
    peak_snrs = list(snr[int(p+0.5)] for p in final_peaks)

    # Remove suspiciously bright peaks
    if rank_clip:
        diff = 3  # Compare 1st brightest to 4th brightest
        rank_order = list(np.argsort(peak_snrs))
        while len(rank_order) > diff and (peak_snrs[rank_order[-1]] /
                                          peak_snrs[rank_order[-(diff+1)]]) > 3:
            del rank_order[-1]
            peak = final_peaks[rank_order[-1]]
            mask[int(peak-maxwidth):int(peak+maxwidth+1)] |= DQ.bad_pixel
        final_peaks = [final_peaks[i] for i in rank_order]
        peak_snrs = [peak_snrs[i] for i in rank_order]

    pixels, snrs = zip(*sorted(zip(final_peaks, peak_snrs)))
    return np.array([pixels, snrs])

def _set_model(model, order=None, initial=False, kdfit=False):
    """
    Initialize a new Model object based on an existing one, e.g., because
    the order has increased.

    Parameters
    ----------
    model: Model (Chebyshev1D)
        existing model transformation
    order: int
        degree of new Chebyshev1D model
    initial: bool
        is this the first fit attempt? (if so, bounds are larger)
    kdfit: bool
        will this model be fit with KDTreeFitter (if not, bounds aren't allowed)

    Returns
    -------
    Model: a new Chebyshev1D model instance
    """
    old_order = model.degree
    assert old_order > 0
    if order is None:
        order = old_order

    # Creating an entirely new model means we don't have to worry about
    # deleting the bounds from the existing model
    new_model = models.Chebyshev1D(degree=order, domain=model.domain)
    for i in range(order+1):
        param = 'c{}'.format(i)
        setattr(new_model, param, getattr(model, param, 0))

    if kdfit:
        dw = abs(2 * new_model.c1 / np.diff(new_model.domain)[0])
        c0_unc = 50*dw if initial else 50*dw
        new_model.c0.bounds = (new_model.c0-c0_unc, new_model.c0+c0_unc)
        c1_unc = (0.1 if initial else 0.02)*abs(new_model.c1)
        new_model.c1.bounds = tuple(sorted([new_model.c1-c1_unc, new_model.c1+c1_unc]))
        for i in range(2, order+1):
            getattr(new_model, 'c{}'.format(i)).bounds = (-5, 5)
    return new_model

def plot_arc_fit(data, peaks, arc_lines, model, title):
    fig, ax = plt.subplots()
    ax.plot(model(np.arange(len(data))), data/np.max(data))
    for line in arc_lines:
        ax.plot([line,line], [0,1], 'k')
    for peak in model(peaks):
        ax.plot([peak,peak], [0,1], 'r:')
    limits = model([0, len(data)])
    ax.set_xlim(min(limits), max(limits))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(title)