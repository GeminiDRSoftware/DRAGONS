import re
from itertools import product as cart_product
from functools import partial, reduce

import numpy as np
from astropy.modeling.polynomial import Chebyshev1D
from scipy.spatial import cKDTree
from bisect import bisect

from astropy import units as u
from astropy.modeling import fix_inputs, fitting, models, Model, CompoundModel
from astropy.table import Table
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS
from specutils.utils.wcs_utils import air_to_vac, vac_to_air
from matplotlib import pyplot as plt

from astrodata import wcs as adwcs

from . import matching, tracing, peak_finding
from .fitting import fit_1D
from . import astromodels as am

from ..utils import logutils
from ..utils.decorators import insert_descriptor_values


class FakeLog:
    """Simple class that suppresses all logging messages"""
    def __getattr__(self, attr):
        return self.null_func

    def null_func(*args, **kwargs):
        return


class LineList:
    """
    A container to hold a list of reference line wavelengths and allow
    conversions between air and vacuum wavelengths
    """
    def __init__(self, filename=None):
        self._lines = None
        self._weights = None
        self._ids = None
        self._units = None
        self._in_vacuo = None
        self._decimals = 3
        self.reference_spectrum = None
        if filename:
            self.read_linelist(filename)

    def __len__(self):
        try:
            return self._lines.size
        except AttributeError:
            return 0

    @property
    def units(self):
        if self._units:
            return self._units
        if min(self._lines) < 3000:
            return u.nm
        elif max(self._lines) > 12000:
            return u.AA
        raise ValueError("Units are not defined and cannot be calculated")

    @property
    def ids(self):
        """Identifications of the lines (not yet implemented)"""
        return self._ids

    @property
    def weights(self):
        """Weights of the individual lines for fitting routines"""
        return self._weights

    def read_linelist(self, filename):
        """
        Read a text file containing the reference line list

        Parameters
        ----------
        filename : str
            name of text file
        """
        r = re.compile(r".*\sunits\s+(.+)")
        is_air = False
        is_vacuo = False
        data_lines = []
        with open(filename, "r") as f:
            for line in f.readlines():
                # We accept any case if there's a space before it, or require
                # all caps, to avoid matching stuff like "Blair & Brown (2010)"
                if not data_lines:
                    words = [word.upper() for word in line.strip().split()]
                    is_air |= "AIR" in words
                    is_vacuo |= "VACUUM" in line
                    m = r.match(line)
                    if m:
                        try:
                            self._units = u.Unit(m.group(1))
                        except ValueError:
                            pass
                try:
                    float(line.strip().split()[0])
                except ValueError:
                    pass
                else:
                    data_lines.append(line.split('#')[0].strip())

        if is_air ^ is_vacuo:
            self._in_vacuo = is_vacuo
        else:
            raise OSError("AIR or VACUUM wavelengths not specified in "
                          f"{filename}")

        # If we're converting between air and vacuum, we want the transformed
        # wavelengths to have the same number of decimal places as the input
        self._decimals = max(len((line.strip().split()[0]+".").split(".")[1])
                             for line in data_lines)
        self._lines = np.genfromtxt(data_lines, usecols=[0])
        try:
            self._weights = np.genfromtxt(data_lines, usecols=[1])
        except ValueError:
            self._weights = None
        # np.genfromtxt() silently returns an array of NaNs if it finds a column
        # in the given location but which it can't coerce to numerical form.
        # The check below catches that situation.
        if (self._weights is not None) and (np.isnan(self._weights).all()):
            self._weights = None

    def vac_to_air(self, wavelengths):
        """
        Converts vacuum to air wavelengths

        Parameters
        ----------
        wavelength : `Quantity` object (number or sequence)
            Vacuum wavelengths with an astropy.unit.

        Returns
        -------
        air_wavelength : `Quantity` object (number or sequence)
            Air wavelengths with the same unit as wavelength.
        """

        return np.round(vac_to_air(wavelengths), decimals=self._decimals)

    def air_to_vac(self, wavelengths):
        """
        Converts air to vacuum wavelengths.

        Parameters
        ----------
        wavelength : `Quantity` object (number or sequence)
            Air wavelengths with an astropy.unit.

        Returns
        -------
        vac_wavelength : `Quantity` object (number or sequence)
            Vacuum wavelengths with the same unit as wavelength.
        """

        return np.round(air_to_vac(wavelengths), decimals=self._decimals)

    def wavelengths(self, in_vacuo=None, units=None):
        """Return line wavelengths in air/vacuum (possibly with particular units)"""
        if not in_vacuo in (True, False):
            raise ValueError(f"in_vacuo must be True or False, not '{in_vacuo}'")
        if in_vacuo:
            return self.vacuum_wavelengths(units=units)
        return self.air_wavelengths(units=units)

    def air_wavelengths(self, units=None):
        """
        Return wavelengths of lines in air

        Parameters
        ----------
        units : str/u.Unit/None
            if None, return a Quantity object
            otherwise return an array in the specified units
        """
        wavelengths = self._lines * self.units
        if self._in_vacuo:
            wavelengths = self.vac_to_air(wavelengths)
        if units is None:
            return wavelengths
        elif isinstance(units, str):
            units = u.Unit(units)
        return wavelengths.to(units).value

    def vacuum_wavelengths(self, units=None):
        """
        Return wavelengths of lines in vacuo

        Parameters
        ----------
        units : str/u.Unit/None
            if None, return a Quantity object
            otherwise return an array in the specified units
        """
        wavelengths = self._lines * self.units
        if not self._in_vacuo:
            wavelengths =  self.air_to_vac(wavelengths)
        if units is None:
            return wavelengths
        elif isinstance(units, str):
            units = u.Unit(units)
        return wavelengths.to(units).value

    def convert_refplot_to_air(self):
        """Convert refplot wavelengths to air (in nanometers)"""
        units = u.Unit("nm")
        if self.reference_spectrum is not None:
            if "refplot_spec" in self.reference_spectrum:
                refplot_spec = self.reference_spectrum["refplot_spec"]
                refplot_spec[:, 0] = self.vac_to_air(refplot_spec[:, 0] * units)
                self.reference_spectrum["refplot_spec"] = refplot_spec
            if "refplot_linelist" in self.reference_spectrum:
                refplot_linelist = self.reference_spectrum["refplot_linelist"]
                refplot_linelist[:, 0] = self.vac_to_air(refplot_linelist[:, 0] * units)
                self.reference_spectrum["refplot_linelist"] = refplot_linelist


def find_line_peaks(data, mask=None, variance=None, fwidth=None, min_snr=3,
                    min_sep=2, reject_bad=False, nbright=0):
    """
    Find peaks in a 1D spectrum and return their locations and weights for
    a variety of weighting schemes.

    Parameters
    ----------
    data : ndarray
        1D array representing the data
    mask : ndarray / None
        mask to be applied to the data
    variance : ndarray / None
        variance of the data
    fwidth : float
        feature width (FWHM) in pixels
    min_snr : float
        minimum signal-to-noise ratio for line detection
    min_sep : float
        minimum separation in pixels between adjacent peaks
    reject_bad : bool
        reject peaks identified as bad by "forensic accounting"?
    nbright : int
        reject this number of the brightest peaks

    Returns
    -------
    peaks : ndarray
        the pixel locations of peaks
    weights : dict
        weight for each line for each of the weighting schemes
    """
    # Find peaks; convert width FWHM to sigma
    widths = 0.42466 * fwidth * np.arange(0.75, 1.26, 0.05)  # TODO!
    peaks, peak_snrs = peak_finding.find_wavelet_peaks(
        data, widths=widths, mask=mask, variance=variance, min_snr=min_snr,
        min_sep=min_sep, reject_bad=reject_bad)
    fit_this_peak = peak_snrs > min_snr
    fit_this_peak[np.argsort(peak_snrs)[len(peaks) - nbright:]] = False
    peaks = peaks[fit_this_peak]
    peak_snrs = peak_snrs[fit_this_peak]

    # Compute all the different types of weightings so we can
    # change between them as needs require
    weights = {"uniform": np.ones((len(peaks),)),
               "global": np.sqrt(peak_snrs)}
    # The "local" weights compares each line strength to
    # those of the lines close to it
    tree = cKDTree(np.array([peaks]).T)
    # Find lines within 10% of the array size
    indices = tree.query(np.array([peaks]).T, k=10,
                         distance_upper_bound=0.1 * len(data))[1]
    snrs = np.array(list(peak_snrs) + [np.nan])[indices]
    # Normalize weights by the median of these lines
    weights["local"] = peak_snrs / np.nanmedian(snrs, axis=1)

    return peaks, weights


def find_alternative_solutions(peaks, arc_lines, model, kdsigma, weights=None):
    """
    Searches for an alternative initial wavelength model, in case the
    descriptor return values are wrong.

    Parameters
    ----------
    peaks : array
        list of line peaks (pixels)
    arc_lines : array
        list of reference wavelengths
    model : Model
        initial solution for peaks->arc_lines mapping
    kdsigma : float
        KDTree fitter parameter
    weights : array / None
        weights assigned to peaks

    Returns
    -------
    Model / None
        an alternative initial model if one seems appropriate
    """
    fit_it = matching.KDTreeFitter(sigma=kdsigma, maxsig=5, k=1,
                                   method='differential_evolution')
    m_tweak = (models.Shift(0, bounds={"offset": (-100, 100)}) |
               models.Scale(1, bounds={"factor": (0.98, 1.02)}))
    peak_waves = model(peaks)
    m_out = fit_it(m_tweak, peak_waves, arc_lines, in_weights=weights)
    diffs = m_out(peak_waves) - peak_waves
    if abs(np.median(diffs)) > 10:
        # Linear approximation to modified model
        cheb_params = {k: v for k, v in zip(model.param_names,
                                            model.parameters)}
        cheb_params.update({'c0': m_out(model.c0),
                            'c1': m_out[1](model.c1)})
        new_model = Chebyshev1D(degree=model.degree, **cheb_params, domain=model.domain)
        return [new_model]


def get_center_from_correlation(data, arc_lines, peaks, sigma, c0, c1):
    len_data = len(data)
    m = models.Chebyshev1D(degree=1, c0=c0, c1=c1, domain=[0, len_data-1])
    w = m(np.arange(len_data))
    fake_arc = np.zeros_like(w)
    fake_data = np.zeros_like(w)
    for p in m(peaks):
        fake_data += np.exp(-0.5*(w-p)*(w-p)/(sigma*sigma))
    for p in arc_lines:
        fake_arc += np.exp(-0.5*(w-p)*(w-p)/(sigma*sigma))
    p = np.correlate(fake_data, fake_arc, mode='full').argmax() - len_data + 1
    return c0 - 2 * p * c1/(len_data - 1)



@insert_descriptor_values("dispersion_axis")
def initial_wavelength_model(ext, central_wavelength=None, dispersion=None,
                             dispersion_axis=None, axes={}):
    """
    Return the initial wavelength model for an NDData/NDAstroData object.

    This initially inspects the "wcs" attribute, and returns a model based on
    this, if it exists. If not, then a linear model is computed from the
    central_wavelength and dispersion parameters along the appropriate axis.
    The model need not be elegant since it is temporary and only going to be
    evaluated forwards.

    Parameters
    ----------
    ext : NDData-like
        data providing wavelength solution
    central_wavelength : float / None
        central wavelength in nm
    dispersion : float / None
        dispersion in nm/pixel
    dispersion_axis : int
        axis (python sense) along which data are dispersed
    axes : dict
        pixel locations along non-dispersion axes where WCS should be calculated

    Returns
    -------
    Model : a model with n_inputs=1 that returns the wavelength at that pixel
            along the 1D spectrum
    """
    npix = ext.shape[dispersion_axis]
    try:
        fwd_transform = ext.wcs.forward_transform.copy()
    except AttributeError:
        # Descriptors are not evaluated by the decorator to avoid overriding
        # the WCS (which may have done some tweaks, e.g., GMOS-S 1um offset)
        if central_wavelength is None:
            central_wavelength = ext.central_wavelength(asNanometers=True)
        if dispersion is None:
            dispersion = ext.dispersion(asNanometers=True)
        model = models.Chebyshev1D(degree=1, c0=central_wavelength,
                                   c1=0.5 * dispersion * (npix - 1),
                                   domain=[0, npix-1])
    else:
        ndim = len(ext.shape)
        #axis_dict = {ndim-i-1: axes.get(i, 0.5 * (length-1))
        #             for i, length in enumerate(ext.shape) if i != dispersion_axis}
        #model = (fix_inputs(fwd_transform, axis_dict) |
        #         models.Mapping((0,), n_inputs=fwd_transform.n_outputs))
        # Ugly hack until fix_inputs broadcasting is fixed
        # https://github.com/astropy/astropy/issues/12021
        axis_models = [models.Const1D(axes.get(i, 0.5 * (length-1)))
                       for i, length in enumerate(ext.shape)]
        for m in axis_models:
            m.amplitude.fixed = True
        if isinstance(fwd_transform, CompoundModel):
            for m in fwd_transform:
                if m.name == "SKY":
                    for p in m.param_names:
                        getattr(m, p).fixed = True
        axis_models[dispersion_axis] = models.Identity(1)
        model = (models.Mapping((0,) * ndim) |
                 reduce(Model.__and__, axis_models[::-1]) |
                 fwd_transform |
                 models.Mapping((0,), n_inputs=fwd_transform.n_outputs))
        if dispersion or central_wavelength:
            actual_cenwave = model(0.5 * (npix - 1))
            model |= models.Shift(-actual_cenwave)
            if dispersion:
                actual_dispersion = np.diff(model([0, npix - 1]))[0] / (npix - 1)
                model |= models.Scale(dispersion / actual_dispersion)
            model |= models.Shift(actual_cenwave if central_wavelength is None
                                  else central_wavelength)

    # The model might not have an actual domain but we want this information,
    # so stick it in the meta
    model.meta["domain"] = [0, npix - 1]
    return model


def create_interactive_inputs(ad, ui_params=None, p=None,
                              linelist=None, bad_bits=0):
    data = {"x": [], "y": [], "meta": []}
    for ext in ad:
        input_data, fit1d, _ = get_automated_fit(
            ext, ui_params, p=p, linelist=linelist, bad_bits=bad_bits)
        # peak locations and line wavelengths of matched peaks/lines
        data["x"].append(fit1d.points[~fit1d.mask])
        data["y"].append(fit1d.image[~fit1d.mask])

        # Get the data necessary for the reference spectrum plot, to be displayed
        # in a separate plot in the interactive mode
        lnlist = input_data["linelist"]
        if lnlist.reference_spectrum is not None:
            if "refplot_linelist" not in lnlist.reference_spectrum:
                # We need to calculate the intensity of the reference spectrum
                # at the locations of the lines
                params = ui_params.toDict()
                refspec = lnlist.reference_spectrum["refplot_spec"]
                wavelengths = lnlist.vacuum_wavelengths(units="nm")
                # LineList might extend beyond edges of displayed spectrum
                fluxes = np.interp(wavelengths, *refspec.T,
                                   left=np.nan, right=np.nan)
                lnlist.reference_spectrum["refplot_linelist"] = np.array(
                    [wavelengths, fluxes]).T[~np.isnan(fluxes)]
            if not params["in_vacuo"]:
                lnlist.convert_refplot_to_air()
            input_data.update(lnlist.reference_spectrum)

        input_data["init_models"] = [fit1d.model] + input_data["init_models"]
        data["meta"].append(input_data)
    return data


def get_automated_fit(ext, ui_params, p=None, linelist=None, bad_bits=0):
    """
    Produces a wavelength fit for a given slice of an AstroData object.
    In non-interactive mode, this is the final result; in interactive mode
    it provides the starting point with a list of matched peaks and arc
    lines.

    Parameters
    ----------
    ext: single-slice AstroData
        the extension
    ui_params: UIParameters object
        class holding parameters for the UI, passed from the primitive's Config
    p: PrimitivesBASE object
        (needed to get the correct linelist... perhaps only need to pass that fn)
    linelist: str
        user-supplied linelist filename
    bad_bits : int
        bitwise-and the mask with this to produce the mask

    Returns
    -------
    input_data : a dict containing useful information
        (see get_all_input_data)
    fit1d : a fit_1D object
        containing the wavelength solution, plus an "image" attribute that
        lists the matched arc line wavelengths
    acceptable_fit : bool
        whether this fit is likely to be good
    """
    input_data = get_all_input_data(
        ext, p, ui_params.toDict(), linelist=linelist, bad_bits=bad_bits)
    spectrum = input_data["spectrum"]
    init_models = input_data["init_models"]
    peaks, weights = input_data["peaks"], input_data["weights"]
    fwidth = input_data["fwidth"]
    dw = np.diff(init_models[0](np.arange(spectrum.size))).mean()
    kdsigma = fwidth * abs(dw)
    k = 1 if kdsigma < 3 else 2
    fit1d, acceptable_fit = find_solution(
        init_models, ui_params.toDict(), peaks=peaks,
        peak_weights=weights[ui_params.weighting],
        linelist=input_data["linelist"], fwidth=fwidth, kdsigma=kdsigma, k=k,
        bounds_setter=input_data["bounds_setter"], filename=ext.filename)

    input_data["fit"] = fit1d
    return input_data, fit1d, acceptable_fit


def create_chebyshev(waves, central_wavelength=None, dispersion=None,
                     max_order=1):
    """
    Create a Chebyshev1D Model instance that fits the pixel->wavelength
    mapping of a 1D AstroNDData object, with the option of overriding the
    central wavelength and/or dispersion

    Parameters
    ----------
    waves: array-like
        wavelengths of pixels
    central_wavelength: float
        estimated wavelength of central pixel (nm)
    dispersion: float
        estimated dispersion (nm/pixel)
    config: Config object
        contains the maximum order of fit

    Returns
    -------
    model: Chebyshev1D
    """
    npix = waves.size
    x = np.arange(npix)
    dw = abs(np.median(np.diff(waves)))
    order = 0
    while order < max_order:
        order += 1
        fit1d = fit_1D(waves, points=x, function="chebyshev",
                       order=order, domain=[0, npix - 1],
                       niter=0, sigma_lower=None, sigma_upper=None)
        if fit1d.rms < 0.001 * dw:
            break
    model = fit1d.model
    if central_wavelength is not None:
        model.c0 = central_wavelength
    if dispersion is not None:
        model.c1 = 0.5 * dispersion * (npix - 1)
    return model


def get_all_input_data(ext, p, config, linelist=None, bad_bits=0,
                       skylines=False, loglevel='stdinfo'):
    """
    There's a specific order needed to do things:
    1) The initial model and 1D spectrum give us the wavelength extrema
       and dispersion
    2) That allows us to read the linelist (if not user-supplied)
    3) The linelist and peak locations are needed before we can look
       for alternative models


    Parameters
    ----------
    ext : AstroData single slice
    p : PrimitivesBASE object
    bad_bits : int
        bitwise-and the mask with this to produce the mask
    config : dict
        dictionary of parameters
    skylines : bool
        True if the reference lines being used are skylines, othewise False if
        they are arc lines
    loglevel : str, ('stdinfo', 'fullinfo', 'debug')
        Sets the log level at which to print some output from the function. If
        left at the default 'stdinfo', all information will be printed to the
        terminal; setting it to a lower level will cause it by default to only
        appear in the reduction log (though it can appear in the terminal if
        the logging level specified there allows it).

    Returns
    -------
    dict : all input information, namely the following:
    "spectrum" : np.ma.masked_array of the 1D spectrum
    "init_models" : list of initial wavelength solution model(s)
    "peaks" : array of peak locations
    "weights" : dict of peaks weights in various weighting schemes
    "linelist" : LineList object
    "fwidth" : feature width (pixels)
    "location" : extraction location (if 2D spectrum)
     "bounds_setter" : a callable to set the uncertainty on polynomial parameters
    """
    cenwave = config["central_wavelength"]
    dispersion = config["dispersion"]

    log = FakeLog() if config["interactive"] == True else p.log
    # This allows suppression of the terminal log output by calling the function
    # with loglevel='debug'.
    logit = getattr(log, loglevel)

    # Create 1D spectrum for calibration
    if ext.data.ndim > 1:
        dispaxis = 2 - ext.dispersion_axis()  # python sense
        direction = "row" if dispaxis == 1 else "column"
        const_slit = 'LS' in ext.tags or "TRANSFRM" in ext.phu
        center = config["center"] or int(0.5 * (ext.shape[1 - dispaxis] - 1))
        data, mask, variance, extract_info = peak_finding.average_along_slit(
            ext, center=center, nsum=config["nsum"],
            combiner=config["combine_method"])
        if const_slit:
            logit("Extracting 1D spectrum from {}s {} to {}".
                  format(direction, extract_info.start + 1, extract_info.stop))
            middle = 0.5 * (extract_info.start + extract_info.stop - 1)
            location = f"{direction} {int(middle)}"
        else:
            # For non-straight slits, `extract_info` is the 1D
            # Chebyshev polynomial that traces the center of the slit.
            coeffs = [f"{key}: {value:.2f}" for key, value in
                      zip(extract_info.param_names,
                          extract_info.parameters)]
            logit(f"Extracting 1D spectrum for extension {ext.id}")
            logit(f"  ±{config['nsum']/2:.1f} {direction}s "
                  "around polynomial with " + ", ".join(coeffs))
            middle = extract_info(0.5 * (ext.shape[dispaxis] - 1))
            # TODO: this isn't strictly correct, since it implies extraction
            # at a fixed location.
            location = f"{direction} {int(middle)}"
    else:
        data = ext.data.copy()
        mask = ext.mask.copy()
        variance = ext.variance
        location = ""
    # Mask bad columns but not saturated/non-linear data points
    if mask is not None:
        mask &= bad_bits
        data[mask > 0] = 0.

    if config["fwidth"] is None:
        fwidth = peak_finding.estimate_peak_width(data, mask=mask, boxcar_size=30)
        logit(f"Estimated feature width: {fwidth:.2f} pixels")
    else:
        fwidth = config["fwidth"]

    # If we are doing wavecal from sky absorption lines in object spectrum,
    # the most realistic variance estimation comes from pixel-to-pixel
    # variations, as done in `peak_finding.find_wavelet_peaks`.
    # The variance estimations coming from `tracing.average_along_slit` don't
    # provide sensible values in this particular case.
    if config.get("absorption") is True:
        variance = None
    peaks, weights = find_line_peaks(
        data, mask=mask, variance=variance,
        fwidth=fwidth, min_snr=config["min_snr"], min_sep=config["min_sep"],
        reject_bad=False, nbright=config.get("nbright", 0))
    if len(peaks) == 0:
        raise ValueError(f"No peaks were found; perhaps try a lower min_snr value?")
    # Do the second iteration of fwidth estimation and peak finding, this time using the number of peaks
    # found after the first fwidth estimation, in order to get more accurate
    # line widths. This step is mostly necessary when doing wavecal from sky lines, as for those
    # the brightest peaks also tend to be the widest, thus estimation from 10 brightest lines tends
    # to be too high.
    if config["fwidth"] is None:
        fwidth = peak_finding.estimate_peak_width(data, mask=mask, boxcar_size=30, nlines=len(peaks))
        log.stdinfo(f"Estimated feature width is {fwidth:.2f} pixels")
        peaks, weights = find_line_peaks(
            data, mask=mask, variance=variance,
            fwidth=fwidth, min_snr=config["min_snr"], min_sep=config["min_sep"],
            reject_bad=False, nbright=config.get("nbright", 0))

    # Determine extent of data in spectrum
    x1 = mask.astype(bool).argmin()
    x2 = mask.size - mask.astype(bool)[::-1].argmin()

    if dispaxis == 1:
        _slice = (center, slice(None))
    else:
        _slice = (slice(None), center)
    ndd = ext.nddata[_slice]

    # Get the initial wavelength solution
    npix = ndd.shape[0]
    try:
        ndd.wcs.forward_transform
    except AttributeError:
        m_init = models.Chebyshev1D(degree=1, c0=cenwave or ext.central_wavelength(asNanometers=True),
                                    c1=0.5 * (dispersion or ext.dispersion(asNanometers=True)) * (npix - 1),
                                    domain=[0, npix - 1])
    else:
        m_init = create_chebyshev(
            ndd.wcs(np.arange(npix))[0], central_wavelength=cenwave,
            dispersion=dispersion, max_order=config["order"])

    # Convert to appropriate domain
    cheb = np.polynomial.chebyshev.Chebyshev(m_init.parameters, m_init.domain)
    coef = {f'c{i}': v for i, v in enumerate (cheb.convert(domain=(x1, x2 - 1)).coef)}
    m_init = Chebyshev1D(degree=m_init.degree, **coef, domain=(x1, x2 - 1))

    waves = m_init([0, 0.5 * (npix - 1), npix - 1])
    dw0 = (waves[2] - waves[0]) / (npix - 1)
    logit("Wavelengths at start, middle, end (nm), and dispersion "
          f"(nm/pixel):\n{waves} {dw0:.4f}")

    if linelist is None:
        linelist = p._get_linelist(wave_model=m_init, ext=ext, config=config)

    # This wants to be logged even in interactive mode
    sky_or_arc = 'reference sky' if skylines else 'arc'
    msg = f"Found {len(peaks)} peaks and {len(linelist)} {sky_or_arc} lines"
    p.log.stdinfo(msg) if config["interactive"] == True else logit(msg)

    m_init = [m_init]
    kdsigma = fwidth * abs(dw0)
    if cenwave is None:
        if config["debug_alternative_centers"]:
            alt_models = find_alternative_solutions(
                peaks, linelist.wavelengths(in_vacuo=config["in_vacuo"], units="nm"),
                m_init[0], 2.5 * kdsigma, weights=weights["global"])
            if alt_models is not None:
                m_init.extend(alt_models)
                log.warning("Alternative model(s) found")
                for i, m in enumerate(alt_models, start=1):
                    waves = m([0, 0.5 * (npix - 1), npix - 1])
                    dw0 = (waves[2] - waves[0]) / (npix - 1)
                    log.warning(f"{i}. Wavelength at middle, and dispersion "
                                f"(nm/pixel):\n{waves[1]} {dw0:.4f}")

    try:
        peak_to_centroid_func = p._convert_peak_to_centroid(ext)
    except AttributeError:
        peak_to_centroid_func = lambda x: x
    else:
        p.log.stdinfo("Applying peak-to-centroid shifts to lines.")
        peaks = peak_to_centroid_func(peaks)

    return {"spectrum": np.ma.masked_array(data[x1:x2], mask=mask[x1:x2]),
            "init_models": m_init, "peaks": peaks, "weights": weights,
            "linelist": linelist, "fwidth": fwidth, "location": location,
            "peak_to_centroid_func": peak_to_centroid_func,
            "bounds_setter": partial(p._wavelength_model_bounds, ext=ext)}

def find_solution(init_models, config, peaks=None, peak_weights=None,
                  linelist=None, fwidth=4,
                  kdsigma=1, k=1, filename=None, bounds_setter=None):
    """
    Find the best wavelength solution from the set of initial models.

    init_models : list of Chebyshev1D instances
        starting models
    config : dict
        dictionary of parameters
    peaks : list of floats
        list of peaks found in image
    peak_weights : list of floats
        list of peak weights
    filename : str
        name of file being checked
    bounds_setter : callable
        function to provide bounds for the wavelength model

    Returns
    -------
    length-2 tuple
        A tuple of the best-fit model and a boolean denoting whether or not it
        is an acceptable fit.
    """
    log = logutils.get_logger(__name__)
    min_lines = [int(x) for x in str(config["debug_min_lines"]).split(',')]
    best_score = np.inf
    arc_lines = linelist.wavelengths(in_vacuo=config["in_vacuo"], units="nm")
    arc_weights = linelist.weights
    # This allows suppression of the terminal log output by calling the function
    # with loglevel='debug'.
    loglevel = "stdinfo" if config["verbose"] else "fullinfo"
    logit = getattr(log, loglevel)

    best_fit1d = None
    # Iterate over start position models most rapidly
    for min_lines_per_fit, model, loc_start in cart_product(
            min_lines, init_models, (0.5, 0.3, 0.7)):
        domain = model.domain
        #print("STARTING", model.parameters, loc_start, domain)
        len_data = np.diff(domain)[0]  # actually len(data)-1
        pixel_start = domain[0] + loc_start * len_data

        matches = perform_piecewise_fit(model, peaks, arc_lines, pixel_start,
                                        kdsigma, order=config["order"],
                                        min_lines_per_fit=min_lines_per_fit,
                                        k=k, bounds_setter=bounds_setter)

        # We perform a regular least-squares fit to all the matches
        # we've made. This allows a high polynomial order to be
        # used without the risk of it going off the rails
        matched = np.where(matches > -1)[0]
        fit_it = fitting.LinearLSQFitter()
        if len(matched) > 1:  # need at least 2 lines, right?
            m_init = models.Chebyshev1D(degree=min(config["order"], len(matched)-1),
                                        domain=domain)
            for p, v in zip(model.param_names, model.parameters):
                if p in m_init.param_names:
                    setattr(m_init, p, v)
            #bounds_setter(m_init)
            #for i in range(len(matched), m_init.degree + 1):
            #    m_init.fixed[f"c{i}"] = True
            matched_peaks = peaks[matched]
            matched_arc_lines = arc_lines[matches[matched]]
            m_final = fit_it(m_init, matched_peaks, matched_arc_lines)
            #for p, l in zip(matched_peaks, matched_arc_lines):
            #    print(f"{p:.2f} => {l:.2f}")

            # We're close to the correct solution, perform a KDFit
            m_init = models.Chebyshev1D(degree=config["order"], domain=domain)
            for p, v in zip(m_final.param_names, m_final.parameters):
                setattr(m_init, p, v)
            dw = abs(np.diff(m_final(m_final.domain))[0] / np.diff(m_final.domain)[0])
            fit_it = matching.KDTreeFitter(sigma=2 * abs(dw), maxsig=5,
                                           k=k, method='Nelder-Mead')
            m_final = fit_it(m_init, peaks, arc_lines, in_weights=peak_weights,
                             ref_weights=arc_weights)
            logit(f'{repr(m_final)} {fit_it.statistic}')

            # And then recalculate the matches
            match_radius = 4 * fwidth * abs(m_final.c1) / len_data  # 2*fwidth pixels
            try:
                matched = matching.match_sources(m_final(peaks), arc_lines,
                                                 radius=match_radius)
                incoords, outcoords = zip(*[(peaks[i], arc_lines[m])
                                            for i, m in enumerate(matched) if m > -1])
                # Probably incoords and outcoords as defined here should go to
                # the interactive fitter, but cull to derive the "best" model
                fit1d = fit_1D(outcoords, points=incoords, function="chebyshev",
                               order=min(m_final.degree, len(incoords)-1),
                               domain=m_final.domain,
                               niter=config["niter"], sigma_lower=config["lsigma"],
                               sigma_upper=config["hsigma"])
                fit1d.image = np.asarray(outcoords)
            except ValueError:
                log.warning("Line-matching failed")
                continue
            nmatched = np.sum(~fit1d.mask)
            logit(f"{filename} {repr(fit1d.model)} {nmatched} {fit1d.rms}")

            # Wavelength solution models need to be monotonic. Make that check.
            waves = fit1d.evaluate(np.arange(len_data))
            if not (np.all(np.diff(waves) > 0) or np.all(np.diff(waves) < 0)):
                continue

            # Calculate how many lines *could* be fit. We require a constrained
            # fit but also that it fits some reasonable number of lines
            nfittable_lines = np.sum(np.logical_and(arc_lines > waves.min(), arc_lines < waves.max()))
            min_matches_required = max(config["order"] + min(nfittable_lines // 2, 3), 2)

            # Trial and error suggests this criterion works well
            if fit1d.rms < 0.8 / config["order"] * fwidth * abs(dw) and nmatched >= min_matches_required:
                #print("RETURNING", fit1d.model.parameters)
                return fit1d, True

            # This seems to be a reasonably ranking for poor models
            if nmatched > config["order"] + 1:
                score = fit1d.rms / (nmatched - config["order"] - 1)
            else:
                score = np.inf
            if score < best_score or np.isinf(score) and (best_fit1d is None or fit1d.model.degree > best_fit1d.model.degree):
                best_score = score
                best_fit1d = fit1d

    if best_fit1d is None:
        # Hack a fit1D object that represents the original model with no fitted lines
        best_fit1d = fit_1D(np.arange(5), function="chebyshev", order=1,
                            niter=0)
        best_fit1d._models = init_models[0]
        best_fit1d.image = np.array([])
        best_fit1d.points = np.array([])
        best_fit1d.mask = np.array([], dtype=bool)
    return best_fit1d, True


def perform_piecewise_fit(model, peaks, arc_lines, pixel_start, kdsigma,
                          order=3, min_lines_per_fit=15, k=1,
                          arc_weights=None, bounds_setter=None):
    """
    This function performs fits in multiple regions of the 1D arc spectrum.
    Given a starting location, a suitable fitting region is "grown" outwards
    until it has at least the specified number of both input and output
    coordinates to fit. A fit (usually linear, but quadratic if more than
    half the array is being used and the final fit is order >= 2) is made
    to this region and coordinate matches are found. The matches at the
    extreme ends are then used as the starts of subsequent fits, moving
    outwards until the edges of the data are reached.

    Parameters
    ----------
    model: Model

    peaks : array-like
        pixel locations of detected arc lines
    arc_lines : array-like
        wavelengths of arc lines to be identified
    pixel_start : float
        pixel location from which to make initial regional fit
    kdsigma : float
        scale length for KDFitter (wavelength units)
    order : int
        order of Chebyshev fit providing complete solution
    min_lines_per_fit : int
        minimum number of peaks and arc lines needed to perform a regional fit
    k : int
        maximum number of arc lines to match each peak
    arc_weights : array-like/None
        weights of output coordinates
    bounds_setter : callable
        function to provide bounds for the wavelength model

    Returns
    -------
    array : index in arc_lines that each peak has been matched to (the
            value -1 means no match)
    """
    from datetime import datetime
    start_time = datetime.now()
    matches = np.full_like(peaks, -1, dtype=int)
    len_data = np.diff(model.domain)[0] + 1
    wave_start = model(pixel_start)
    dw_start = np.diff(model([pixel_start - 0.5, pixel_start + 0.5]))[0]
    match_radius = 2 * abs(dw_start)
    fits_to_do = [(pixel_start, wave_start, dw_start, min_lines_per_fit)]

    first = True
    while fits_to_do:
        start = datetime.now()
        p0, c0, dw, min_lines_this_fit = fits_to_do.pop()
        if min(len(arc_lines), len(peaks)) <= min_lines_this_fit:
            p1 = p0
        else:
            p1 = 0
        npeaks = narc_lines = 0
        while (min(npeaks, narc_lines) < min_lines_this_fit and
               not (p0 - p1 < model.domain[0] and p0 + p1 >= model.domain[1])):
            p1 += 1
            i1 = bisect(peaks, p0 - p1)
            i2 = bisect(peaks, p0 + p1)
            #npeaks = i2 - i1
            npeaks = (matches[i1:i2] == -1).sum()  # only count unmatched peaks
            i1 = bisect(arc_lines, c0 - p1 * abs(dw))
            i2 = bisect(arc_lines, c0 + p1 * abs(dw))
            #narc_lines = i2 - i1
            narc_lines = [x not in matches for x in range(i1, i2)].count(True)
        c1 = p1 * dw
        #print(f"Pixel={p0:6.1f} p1={p1:6.1f} c0={c0:9.4f} dw={dw:8.4f} {min_lines_this_fit}")

        if p1 > 0.25 * len_data and order >= 2:
            m_init = models.Chebyshev1D(2, c0=c0, c1=c1,
                                        domain=[p0 - p1, p0 + p1])
        else:
            m_init = models.Chebyshev1D(1, c0=c0, c1=c1,
                                        domain=[p0 - p1, p0 + p1])
        m_init.bounds.update(bounds_setter(m_init))
        if not first:
            m_init.c0.bounds = (c0 - 5 * abs(dw), c0 + 5 * abs(dw))
        #print("INPUT MODEL")
        #print(m_init.parameters, m_init.domain, m_init(np.arange(0,1001,200)))
        #print(m_init.bounds)
        #print(datetime.now() - start)

        # Need to set in_weights=None as there aren't many lines so
        # the fit could be swayed by a single very bright line
        m_this = _fit_region(m_init, peaks, arc_lines, kdsigma,
                             in_weights=None, ref_weights=arc_weights,
                             matches=matches, k=k, first=first)
        #print("OUTPUT", m_this.parameters)
        dw = 2 * m_this.c1 / np.diff(m_this.domain)[0]

        # Add new matches to the list
        new_matches = matching.match_sources(m_this(peaks), arc_lines,
                                             radius=match_radius)
        found_new_matches = False
        for i, (m, p) in enumerate(zip(new_matches, peaks)):
            if matches[i] == -1 and m > -1:
                if p0 - p1 <= p <= p0 + p1:
                    # automatically removes old (bad) match
                    matches[i] = m
                    found_new_matches = True
                    #print(f"Pixel {p} => {arc_lines[m]}")
        try:
            p_lo = peaks[matches > -1].min()
        except ValueError:
            pass
        else:
            # The commented-out line below speeds up fitting in some situations,
            # but also breaks regression tests. More investigation needed. DB
            # if min(len(arc_lines), len(peaks)) > min_lines_per_fit:
                if p_lo < p0 <= pixel_start:
                    arc_line = arc_lines[matches[list(peaks).index(p_lo)]]
                    fits_to_do.append((p_lo, arc_line, dw, min_lines_per_fit))
                p_hi = peaks[matches > -1].max()
                if p_hi > p0 >= pixel_start:
                    arc_line = arc_lines[matches[list(peaks).index(p_hi)]]
                    fits_to_do.append((p_hi, arc_line, dw, min_lines_per_fit))
        #dc0 = 5 * abs(dw)
        #if not found_new_matches and min_lines_this_fit < 2 * min_lines_per_fit:
        #    fits_to_do.append((p0, c0, dw, 2 * min_lines_this_fit))
        first = False

    return matches


def _fit_region(m_init, peaks, arc_lines, kdsigma, in_weights=None,
                ref_weights=None, matches=None, k=1, first=False):
    """
    This function fits a region of a 1D spectrum (delimited by the domain of
    the input Chebyshev model) using the KDTreeFitter. Only detected peaks
    and arc lines within this domain (and a small border to prevent mismatches
    when a feature is near the edge) are matched. An improved version of the
    input model is returned.

    Parameters
    ----------
    m_init : Model
        initial model desccribing the wavelength solution
    peaks : array-like
        pixel locations of detected arc lines
    arc_lines : array-like
        wavelengths of plausible arc lines
    kdsigma : float
        scale length for KDFitter (wavelength units)
    in_weights : array-like/None
        weights of input coordinates
    ref_weights : array-like/None
        weights of output coordinates
    matches : array, same length as peaks
        existing matches (each element points to an index in arc_lines)
    k : int
        maximum number of arc lines to match each peak

    Returns
    -------
    Model : improved model fit
    """
    p0 = np.mean(m_init.domain)
    p1 = 0.5 * np.diff(m_init.domain)[0]
    # We're only interested in fitting lines in this region
    new_in_weights = (abs(peaks - p0) <= 1.05 * p1).astype(float)
    if in_weights is not None:
        new_in_weights *= in_weights
    w0 = m_init.c0.value
    w1 = abs(m_init.c1.value)
    max_range = max(1.05 * w1, np.sum([abs(np.diff(v)[0])
                                       for v in m_init.bounds.values()]))
    new_ref_weights = (abs(arc_lines - w0) < max_range).astype(float)
    if ref_weights is not None:
        new_ref_weights *= ref_weights
    #new_ref_weights = ref_weights
    # Maybe consider two fits here, one with a large kdsigma, and then
    # one with a small one (perhaps the second could use weights)?
    fit_it = matching.KDTreeFitter(sigma=kdsigma, maxsig=10, k=k, method='direct')
    m_init.linear = False  # supress warning
    m_this = fit_it(m_init, peaks, arc_lines, in_weights=new_in_weights,
                    ref_weights=new_ref_weights, matches=matches,
                    locally_biased=not first, eps=1e-2,
                    maxfun=80000, maxiter=20000, vol_tol=1e-6 if first else 1e-4)
    m_this.linear = True
    return m_this


def fit1d_from_kdfit(input_coords, output_coords, model,
                     match_radius, sigma_clip=None):
    """
    Creates a fit_1D object from a KDTree-fitted model. This does
    the matching between input and output coordinates and, if
    requested, iteratively sigma-clips.

    Parameters
    ----------
    input_coords: array-like
        untransformed input coordinates
    output_coords: array-like
        output coordinates
    model: Model
        transformation
    match_radius: float
        maximum distance for matching coordinates
    sigma_clip: float/None
        if not None, iteratively sigma-clip using this number of
        standard deviations

    Returns
    -------
    fit_1D : the fit
    """
    num_matches = None
    init_match_radius = match_radius
    while True:
        matched = matching.match_sources(model(input_coords), output_coords,
                                         radius=match_radius)
        incoords, outcoords = zip(*[(input_coords[i], output_coords[m])
                                    for i, m in enumerate(matched) if m > -1])
        fit1d = fit_1D(outcoords, points=incoords, function="chebyshev",
                       order=model.degree, domain=model.domain, niter=0)
        if sigma_clip is None or num_matches == len(incoords):
            break
        num_matches = len(incoords)
        match_radius = min(init_match_radius, sigma_clip * fit1d.rms)
    fit1d.image = np.asarray(outcoords)
    return fit1d


def update_wcs_with_solution(ext, fit1d, input_data, config):
    """
    Attach a WAVECAL table and update the WCS of a single AstroData slice
    based on the result of the wavelength solution model.

    Parameters
    ----------
    ext : single-slice AstroData
        the extension to be updated
    fit1d : fit_1D
        the best-fitting model
    input_data : dict
        stuff
    config : config
    """
    log = logutils.get_logger(__name__)
    in_vacuo = config.in_vacuo

    # Because of the way the fit_1D object is constructed, there
    # should be no masking. But it doesn't hurt to make sure, or
    # be futureproofed in case we change things.
    incoords = fit1d.points[~fit1d.mask]
    outcoords = fit1d.image[~fit1d.mask]

    m_final = fit1d.model
    domain = m_final.domain
    rms = fit1d.rms
    nmatched = len(incoords)
    log.stdinfo("Chebyshev coefficients: "+" ".join(
        f"{p:.5f}" for p in m_final.parameters))
    # TODO: Do we need input_data? config.fwidth?
    log.stdinfo(f"Matched {nmatched}/{len(input_data['peaks'])} lines with "
                f"rms = {rms:.3f} nm")

    dw = np.diff(m_final(domain))[0] / np.diff(domain)[0]
    max_rms = max(0.2 * rms / abs(dw), 1e-4)  # in pixels
    max_dev = 3 * max_rms
    m_inverse = am.make_inverse_chebyshev1d(m_final, rms=max_rms,
                                            max_deviation=max_dev)
    if len(incoords) > 1:
        inv_rms = np.std(m_inverse(m_final(incoords)) - incoords)
        log.stdinfo(f"Inverse model has rms = {inv_rms:.3f} pixels.")
    m_final.name = "WAVE"  # always WAVE, never AWAV
    m_final.inverse = m_inverse

    if len(incoords):
        indices = np.argsort(incoords)
        # Add 1 to pixel coordinates so they're 1-indexed
        incoords = np.float32(incoords[indices]) + 1
        outcoords = np.float32(outcoords[indices])
    temptable = am.model_to_table(m_final, xunit=u.pixel, yunit=u.nm)

    #### Temporary to ensure all the old stuff is still there
    # while I refactor tests
    temptable.add_columns([[1], [m_final.degree], [domain[0]], [domain[1]]],
                          names=("ndim", "degree", "domain_start", "domain_end"))
    temptable.add_columns([[rms], [input_data["fwidth"]]],
                          names=("rms", "fwidth"))
    if ext.data.ndim > 1:
        # TODO: Need to update this from the interactive tool's values
        direction, location = input_data["location"].split()
        temptable[direction] = int(location)
        temptable["nsum"] = config.nsum
    pad_rows = nmatched - len(temptable.colnames)
    if pad_rows < 0:  # Really shouldn't be the case
        incoords = list(incoords) + [0] * (-pad_rows)
        outcoords = list(outcoords) + [0] * (-pad_rows)
        pad_rows = 0

    fit_table = Table([temptable.colnames + [''] * pad_rows,
                       list(temptable[0].values()) + [0] * pad_rows,
                       incoords, outcoords],
                      names=("name", "coefficients", "peaks", "wavelengths"),
                      units=(None, None, u.pix, u.nm),
                      meta=temptable.meta)
    medium = "vacuo" if in_vacuo else "air"
    fit_table.meta['comments'] = [
        'coefficients are based on 0-indexing',
        'peaks column is 1-indexed',
        f'calibrated with wavelengths in {medium}']
    ext.WAVECAL = fit_table

    spectral_frame = (ext.wcs.output_frame if ext.data.ndim == 1
                      else ext.wcs.output_frame.frames[0])
    axis_name = "WAVE" if in_vacuo else "AWAV"
    new_spectral_frame = cf.SpectralFrame(
        axes_order=spectral_frame.axes_order,
        unit=spectral_frame.unit, axes_names=(axis_name,),
        name=adwcs.frame_mapping[axis_name].description)

    if ext.data.ndim == 1:
        ext.wcs.set_transform(ext.wcs.input_frame,
                              new_spectral_frame, m_final)
    else:
        # Write out a simplified WCS model with a new "world" frame
        # so it's easier to extract what we need later
        dispaxis = 2 - ext.dispersion_axis()  # python sense
        spatial_frame = cf.CoordinateFrame(
            naxes=1, axes_type="SPATIAL", axes_order=(1,),
            unit=u.arcsec, name="SPATIAL")
        output_frame = cf.CompositeFrame(
            [new_spectral_frame, spatial_frame], name='world')
        slit_model = models.Scale(ext.pixel_scale())
        try:
            slit_model = ext.wcs.forward_transform[f'crpix{dispaxis + 1}'] | slit_model
        except IndexError:
            pass
        slit_model.name = 'SKY'
        transform = m_final & slit_model
        if dispaxis == 0:
            transform = models.Mapping((1, 0)) | transform
        #ext.wcs = gWCS([(ext.wcs.input_frame, transform),
        #                (output_frame, None)])
        ext.wcs = gWCS(ext.wcs.pipeline[:-2] +
                       [(ext.wcs.pipeline[-2].frame, transform),
                        (output_frame, None)])


def create_pdf_plot(input_data, peaks, arc_lines, title="",
                    absorption=False):
    """
    Create and save a simple pdf plot of the arc spectrum with line
    identifications, useful for checking the validity of the solution.

    Parameters
    ----------
    input_data: dict
        the dict from get_all_input_data()
    peaks: 1d array
        pixel locations of peaks
    arc_lines: 1d array
        wavelengths of arc lines
    title: str
        plot title
    absorption: bool
        has the wavecal been determined from absorption lines? If so, the
        input "spectrum" will be invered

    Returns
    -------
    fig: a matplotlib figure
    """
    if absorption:
        data = -input_data["spectrum"]
        spacing = -0.01
        vert_align = "top"
    else:
        data = input_data["spectrum"]
        spacing = 0.01
        vert_align = "bottom"
    xmin, xmax = input_data["init_models"][0].domain
    pixels = np.arange(xmin, xmax + 1)
    data_max = data.max()
    spacing *= data_max
    fig, ax = plt.subplots()
    ax.plot(pixels, data, 'b-')
    ax.set_ylim(0, data_max * 1.1)
    if len(arc_lines) and np.diff(arc_lines)[0] / np.diff(peaks)[0] < 0:
        ax.set_xlim(xmax + 1, xmin - 1)
    else:
        ax.set_xlim(xmin - 1, xmax + 1)
    for p, w in zip(peaks, arc_lines):
        j = int(p + 0.5)
        ax.plot([p, p], [data[j - xmin] + spacing,
                         data[j - xmin] + 2 * spacing], 'k-')
        ax.text(p, data[j - xmin] + 3 * spacing, str('{:.5f}'.format(w)),
                horizontalalignment='center', verticalalignment=vert_align,
                rotation=90, fontdict={'size': 8})
    ax.set_xlabel("Pixel number")
    ax.set_title(title)
    fig.set_size_inches(17, 11)
    return fig
