#
#                                                                  gemini_python
#
#                                                             primtives_spect.py
# ------------------------------------------------------------------------------
from geminidr import PrimitivesBASE
from . import parameters_spect

import numpy as np
from scipy import spatial
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.table import Table

from matplotlib import pyplot as plt

from datetime import datetime

from gempy.gemini import gemini_tools as gt
from gempy.library import matching, peaks_and_traces as peak
from gempy.library.nddops import NDStacker
from gempy.library.transform import Transform, DataGroup, AstroDataGroup
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

    def determineDistortion(self, adinputs=None, **params):
        """
        This primitives maps the distortion on a detector by tracing lines
        perpendicular to the dispersion direction, and then fitting a
        2D Chebyshev polynomial to the distortion.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        spatial_order: int
            order of fit in spatial direction
        spectral_order: int
            order of fit in spectral direction
        nsum: int
            number of rows/columns to sum at each step
        step: int
            size of step in pixels when tracing
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        spatial_order = params["spatial_order"]
        spectral_order = params["spectral_order"]
        nsum = params["nsum"]
        step = params["step"]
        max_shift = params["max_shift"]

        orders = (spectral_order, spatial_order)

        for ad in adinputs:
            for ext in ad:
                self.viewer.display_image(ext)
                self.viewer.width = 2
                dispaxis = 2 - ext.dispersion_axis()  # python sense
                middle = 0.5 * ext.shape[1-dispaxis]

                # The coordinates are always returned as (x-coords, y-coords)
                initial = ext.WAVECAL['peaks']-1
                #initial = [1361.560327644395]
                ref_coords, in_coords = peak.trace_lines(ext, axis=1-dispaxis,
                        start=middle, initial=initial,
                        width=5, step=step, nsum=nsum, max_missed=5,
                        max_shift=max_shift, viewer=self.viewer)

                m_init = models.Chebyshev2D(x_degree=orders[1-dispaxis],
                                            y_degree=orders[dispaxis],
                                            x_domain=[0, ext.shape[1]-1],
                                            y_domain=[0, ext.shape[0]-1])
                # Find model to transform actual (x,y) locations to the
                # value of the reference pixel along the dispersion axis
                fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                           sigma_clip, sigma=3)
                m_final, _ = fit_it(m_init, *in_coords, ref_coords[1-dispaxis])
                m_inverse, masked = fit_it(m_init, *ref_coords, in_coords[1-dispaxis])

                diff = m_inverse(*ref_coords) - in_coords[1-dispaxis]
                print(np.min(diff), np.max(diff), np.std(diff))

                if dispaxis == 1:
                    model = models.Mapping((0, 1, 1)) | (m_final & models.Identity(1))
                    model.inverse = models.Mapping((0, 1, 1)) | (m_inverse & models.Identity(1))
                else:
                    model = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_final)
                    model.inverse = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_inverse)

                self.viewer.color = "blue"
                yref = np.arange(0, ext.shape[1-dispaxis], step)
                for xref in initial:
                    mapped_coords = np.array(model.inverse([xref] * len(yref), yref)).T
                    self.viewer.polygon(mapped_coords, closed=False, xfirst=True, origin=0)


                columns = []
                for m in (m_final, m_inverse):
                    columns.append(['ndim', 'x_degree', 'y_degree', 'x_domain_start',
                                    'x_domain_end', 'y_domain_start', 'y_domain_end'
                                    ] + list(m.param_names))
                    columns.append([2, m.x_degree, m.y_degree] +
                                    list(m.x_domain) + list(m.y_domain) +
                                    [getattr(m, p).value for p in m.param_names])
                ext.FITCOORD = Table(columns,
                                     names=("name", "coefficients", "inv_name", "inv_coefficients"))
                ext.COORDS = Table([*in_coords] + [*ref_coords], names=('xin','yin','xref','yref'))

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def distortionCorrect(self, adinputs=None, **params):
        """

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        sfx = params["suffix"]
        order = params["order"]

        # The forward Transform is *only* used to determine the shape
        # of the output image, which we want to be the same as the input
        m_ident = models.Identity(2)

        adoutputs = []
        for ad in adinputs:
            transforms = []
            for ext in ad:
                kwargs = dict(zip(ext.FITCOORD['inv_name'],
                                  ext.FITCOORD['inv_coefficients']))
                kwargs.pop('ndim')
                kwargs['x_degree'] = int(kwargs['x_degree'])
                kwargs['y_degree'] = int(kwargs['y_degree'])
                kwargs['x_domain'] = [kwargs.pop('x_domain_start'),
                                      kwargs.pop('x_domain_end')]
                kwargs['y_domain'] = [kwargs.pop('y_domain_start'),
                                      kwargs.pop('y_domain_end')]
                m_inverse = models.Chebyshev2D(**kwargs)

                dispaxis = ext.dispersion_axis() - 1
                if dispaxis == 0:
                    m_ident.inverse = models.Mapping((0, 1, 1)) | (m_inverse & models.Identity(1))
                else:
                    m_ident.inverse = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_inverse)
                transforms.append(Transform(m_ident.copy()))

            adg = AstroDataGroup(ad, transforms)
            ad_out = adg.transform(order=order, parallel=False)


            # Timestamp and update the filename
            #gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        This primitive determines the wavelength solution for an ARC and
        stores it as an attached WAVECAL table.

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

        plot = params["plot"]
        plt.ioff()

        # TODO: This decision would prevent MOS data being reduced so need
        # to think a bit more about what we're going to do. Maybe make
        # central_wavelength() return a one-per-ext list? Or have the GMOS
        # determineWavelengthSolution() recipe check the input has been
        # mosaicked before calling super()?
        #
        # Top-level decision for this to only work on single-extension ADs
        #if not all(len(ad)==1 for ad in adinputs):
        #    raise ValueError("Not all inputs are single-extension AD objects")

        # Get list of arc lines (probably from a text file dependent on the
        # input spectrum, so a private method of the primitivesClass)
        linelists = {}
        if arc_file is not None:
            try:
                arc_lines = np.loadtxt(arc_file, usecols=[0])
            except (IOError, TypeError):
                log.warning("Cannot read file {} - using default linelist".format(arc_file))
                arc_file = None
            else:
                linelists[arc_file] = arc_lines

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
                data[mask > 0] = 0.

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
                    try:
                        arc_lines = linelists[linelist]
                    except KeyError:
                        arc_lines = np.loadtxt(linelist, usecols=[0])
                        linelists[linelist] = arc_lines

                if min(arc_lines) > cenwave+0.5*len(data)*abs(dw):
                    log.warning("Line list appears to be in Angstroms; converting to nm")
                    arc_lines *= 0.1

                # Find peaks; convert width FWHM to sigma
                widths = 0.42466*fwidth * np.arange(0.8,1.21,0.05)  # TODO!
                peaks, peak_snrs = peak.find_peaks(data, widths, mask=mask,
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
                if plot:
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
                        if plot:
                            plot_arc_fit(data, peaks, arc_lines, m_init, "Initial model")
                        log.stdinfo('Initial model: {}'.format(repr(m_init)))
                    fit_it = matching.KDTreeFitter()
                    m_final = fit_it(m_init, peaks_to_fit, arc_lines, maxsig=10, k=1,
                                     in_weights=in_weights[peak_snrs>min_snr], ref_weights=None,
                                     sigma=kdsigma, method='Nelder-Mead',
                                     options={'xtol': 1.0e-7, 'ftol': 1.0e-8})
                    log.stdinfo('{} {}'.format(repr(m_final), fit_it.statistic))
                    if plot:
                        plot_arc_fit(plot_data, peaks, arc_lines, m_final,
                                     "KDFit model order {} KDsigma = {}".format(ord, kdsigma))

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
                    if plot:
                        for incoord, outcoord in zip(m.forward(m.input_coords), m.output_coords):
                           ax.text(incoord, yplot, '{:.4f}'.format(outcoord), rotation=90,
                                   ha='center', va='top')

                    log.stdinfo('{} {} {}'.format(repr(m.forward), len(m.input_coords), m.rms_output))
                    if plot:
                        plot_arc_fit(plot_data, peaks, arc_lines, m.forward,
                                     "MatchBox model order {}".format(ord))

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

                if plot:
                    plot_arc_fit(plot_data, peaks, arc_lines, m_final, "Final fit")

                    m.display_fit()
                    plt.show()

                m.sort()
                # Add 1 to pixel coordinates so they're 1-indexed
                incoords = np.float32(m.input_coords) + 1
                outcoords = np.float32(m.output_coords)
                name_column = ["ndim", "degree", "domain_start", "domain_end"] + list(m_final.param_names)
                coeff_column = [1, order] + list(m_final.domain) + [getattr(m_final, p).value for p in m_final.param_names]
                # Ensure all columns have the same length
                if len(coeff_column) <= nmatched:
                    name_column += [''] * (nmatched - len(coeff_column))
                    coeff_column += [0] * (nmatched - len(coeff_column))
                else:  # Really shouldn't be the case
                    incoords += [0] * (len(coeff_column) - nmatched)
                    outcoords += [0] * (len(coeff_column) - nmatched)

                fit_table = Table([name_column, coeff_column, incoords, outcoords],
                                  names=("name", "coefficients", "peaks", "wavelengths"))
                fit_table.meta['comments'] = ['coefficients are based on 0-indexing',
                                              'peaks column is 1-indexed']
                ext.WAVECAL = fit_table

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def extract1DSpectra(self, adinputs=None, **params):
        """
        This primitive extracts a 1D spectrum. No tracing is done yet, it's
        basically a placeholder.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        center: int/None
            central row/column for 1D extraction (None => use middle)
        nsum: int
            number of rows/columns to average
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        center = params["center"]
        nsum = params["nsum"]

        # This is just cut-and-paste code from determineWavelengthSolution()
        for ad in adinputs:
            for ext in ad:
                if len(ext.shape) == 1:
                    log.warning("{}:{} is already one-dimensional".
                                format(ad.filename, ext.hdr['EXTVER']))
                    continue

                # Determine direction of extraction for 2D spectrum
                slitaxis = ext.dispersion_axis() - 1
                middle = 0.5 * ext.shape[slitaxis]
                extract = slice(max(0, int((center or middle) - 0.5*nsum)),
                              min(ext.shape[slitaxis], int((center or middle) + 0.5*nsum)))
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

                data, mask, variance = NDStacker.mean(data, mask=mask, variance=variance)
                ext.reset(data, mask=mask, variance=variance)

                # Update some header keywords
                for kw in ("CTYPE", "CRPIX", "CRVAL", "CUNIT", "CD1_", "CD2_"):
                    for ax in (1, 2):
                        try:
                            del ext.hdr["{}{}".format(kw, ax)]
                        except KeyError:
                            pass

                # TODO: Properly. Simply put the linear approximation here for now
                ext.hdr["CTYPE1"] = "Wavelength"
                try:
                    coeffs = ext.WAVECAL["coefficients"]
                except AttributeError:
                    crpix = 0.5 * (len(data) + 1)
                    crval = ext.central_wavelength(asNanometers=True)
                    cdelt = ext.dispersion(asNanometers=True)
                else:
                    crpix = np.mean(coeffs[:2]) + 1
                    crval = coeffs[3]
                    cdelt = 2 * coeffs[4] / float(np.diff(coeffs[:2]))

                ext.hdr["CRPIX1"] = crpix
                ext.hdr["CRVAL1"] = crval
                ext.hdr["CDELT1"] = cdelt
                ext.hdr["CUNIT1"] = "nanometers"

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def linearizeSpectra(self, adinputs=None, **params):
        """
        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        w1: float
            Wavelength of first pixel (nm)
        w2: float
            Wavelength of last pixel (nm)
        dw: float
            Dispersion (nm/pixel)
        npix: int
            Number of pixels in output spectrum
        conserve: bool
            Conserve flux (rather than interpolate)?

        Exactly 3 of (w1, w2, dw, npix) must be specified.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        w1 = params["w1"]
        w2 = params["w2"]
        dw = params["dw"]
        npix = params["npix"]
        conserve = params["conserve"]

        # Work out the missing variable from the others
        if npix is None:
            npix = int(np.ceil((w2 - w1) / dw)) + 1
            w2 = w1 + (npix-1) * dw
        elif w1 is None:
            w1 = w2 - (npix-1) * dw
        elif w2 is None:
            w2 = w1 + (npix-1) * dw
        else:
            dw = (w2 - w1) / (npix-1)

        for ad in adinputs:
            for ext in ad:
                attributes = [attr for attr in ('data', 'mask', 'variance')
                              if getattr(ext, attr) is not None]
                try:
                    coeffs = ext.WAVECAL["coefficients"]
                except AttributeError:
                    log.warning("{}:{} has no WAVECAL. Cannot linearize.".
                                format(ad.filename, ext.hdr['EXTVER']))
                    continue

                # Recreate wavelength solution and construct inverse
                order = int(coeffs[2])
                kwargs = {"domain": [*coeffs[:2].data.astype(int)]}
                kwargs.update({"c{}".format(i): value
                               for i, value in enumerate(coeffs[3: 4+order])})
                cheb = models.Chebyshev1D(degree=order, **kwargs)
                cheb.inverse = _make_inverse_chebyshev1d(cheb, rms=0.1)
                transform = Transform(cheb)

                # Linearization (and inverse)
                transform.append(models.Shift(-w1))
                transform.append(models.Scale(1./dw))

                # If we resample to a coarser pixel scale, we may interpolate
                # over features. We avoid this by subsampling back to the
                # original pixel scale (approximately).
                input_dw = np.diff(cheb(cheb.domain))[0] / np.diff(cheb.domain)
                subsample = dw / abs(input_dw)
                if subsample > 1.1:
                    subsample = int(subsample + 0.5)

                dg = DataGroup([ext], [transform])
                dg.output_shape = (npix,)
                output_dict = dg.transform(attributes=attributes, subsample=subsample,
                                           conserve=conserve)
                print(dg.jfactors)
                for key, value in output_dict.items():
                    setattr(ext, key, value)

                ext.hdr["CRPIX1"] = 1.
                ext.hdr["CRVAL1"] = w1
                ext.hdr["CDELT1"] = dw
                ext.hdr["CUNIT1"] = "nanometers"

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

#-----------------------------------------------------------------------------
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



def _set_model(model, order=None, initial=False, kdfit=False):
    # TODO: I find this ugly. Do better!
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
    TODO: This is kind of hacky
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

def _make_inverse_chebyshev1d(model, sampling=1, rms=None):
    """
    This creates a Chebyshev1D model that attempts to be the inverse of
    the model provided.

    Parameters
    ----------
    model: Chebyshev1D
        The model to be inverted
    rms: float/None
        required maximum rms in input space (i.e., pixels)
    """
    order = model.degree
    max_order = order if rms is None else order+2
    incoords = np.arange(*model.domain, sampling)
    outcoords = model(incoords)
    while order <= max_order:
        m_init = models.Chebyshev1D(degree=order, domain=model(model.domain))
        fit_it = fitting.LinearLSQFitter()
        m_inverse = fit_it(m_init, outcoords, incoords)
        rms_inverse = np.std(m_inverse(outcoords) - incoords)
        if rms is None or rms_inverse <= rms:
            break
        order += 1
    return m_inverse