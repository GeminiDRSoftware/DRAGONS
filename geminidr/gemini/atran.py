import os
from importlib import import_module

from astropy.convolution import Gaussian1DKernel, convolve
from astropy.table import Table
import numpy as np
from pygments.lexers.arrow import TYPES
from scipy.signal import find_peaks

from gempy.library import peak_finding, wavecal
from gempy.utils import logutils

from .lookups import atran_models, qa_constraints


log = logutils.get_logger(__name__)


def get_atran_linelist(wave_model=None, ext=None, config=None):
    """
    Return a list of spectral lines to be matched in the wavelength
    calibration, and a reference plot of a convolved synthetic spectrum,
    to aid the user in making the correct identifications.

    The linelist can be generated on-the-fly by finding peaks in the
    convolved spectrum, or read from disk if there exists a suitable
    list for this instrumental setup.

    Parameters
    ----------
    wave_model: ``astropy.modeling.models.Chebyshev1D``
        the current wavelength model (pixel -> wavelength), with an
        appropriate domain describing the illuminated region
    ext: single-slice ``AstroData``
        the extension for which a sky spectrum is being constructed
    config: ``config.Config`` object
        containing various parameters

    Returns
    -------
    ``wavecal.Linelist``
        list of lines to match, including data for a reference plot
    """
    observatory = ext.telescope()
    site = {'Gemini-North': 'mk', 'Gemini-South': 'cp'}[observatory]
    try:
        wv_band = int(config.get("wv_band", ext.raw_wv()))
    except TypeError:  # ext.raw_wv() returned None
        req_wv = ext.requested_wv()
        log.stdinfo("Unknown RAWWV for this observation: "
                    f"using requested WV band ('{req_wv}'-percentile)")
        wv_band = int(req_wv)
    if wv_band == 100:
        # a WV value to use for the case of RAWWV='Any'
        wv_content = {'Gemini-North': 5, 'Gemini-South': 10}[observatory]
    else:
        wv_content = qa_constraints.wvBands[observatory].get(str(wv_band))

    atran_file = os.path.join(os.path.split(__file__)[0], "lookups",
                              "atran_spectra.fits")
    atran_models = Table.read(atran_file)
    waves = atran_models['wavelength']
    data = atran_models[f"{site}_wv{wv_content * 1000:.0f}_za48"]


    resolution = self._get_resolution(ext)

    # The wave_model's domain describes the illuminated region
    wave_model_bounds = self._wavelength_model_bounds(wave_model, ext)
    start_wvl, end_wvl = (np.sort(wave_model(wave_model.domain)) +
                          np.asarray(wave_model_bounds['c0']) -
                          wave_model.c0)


def _uses_atran_linelist(self, cenwave, absorption):
    """
    Returns True if the observation can use ATRAN line list for the wavecal.
    By default ATRAN line lists are used in two cases:
        1) wavecal from sky absorption lines in object spectrum (in XJHK-bands);
        2) wavecal from sky emission lines in L- and M-band.
    In these cases (unless the user is providing her own linelist) we
    generate the line list on-the-fly from an ATRAN model with the parameters
    closest to the frame's observing conditions, and convolve it to the
    resolution of the observation.

    Parameters
    ----------
    cenwave: float
        central wavelength in nm
    absorption: bool
        is wavecal done from absorption lines?
    """
    return absorption or (cenwave >= 2800)


def _get_atran_linelist(self, wave_model=None, ext=None, config=None):
    """
    Returns default filename of the line list generated from an ATRAN
    model spectrum with model parameters matching the observing conditions
    and spectral characteristics of the frame (or those, specified by the user).
    Also returns a dictionary with the data necessary for the reference plot (so
    that we don't have to re-calculate the linelist for making the reference plot).

    Parameters
    ----------
    wave_model: astropy.modeling.models.Chebyshev1D
        estimated wavelength solution (with domain)
    ext: single-slice AstroData
        the extension
    config: Config-like object containing parameters

    Returns
    -------
    atran_filename: str
        full path to the generated ATRAN line list
    refplot_dict: dict
        a dictionary containing data for the reference plot
    """
    lookup_dir = os.path.dirname(import_module('.__init__',
                                               self.inst_lookups).__file__)
    working_dir = os.getcwd()

    refplot_dict = None
    model_params = _get_model_params(wave_model, ext, config)

    atran_linelist = 'atran_linelist_{}_{:.0f}-{:.0f}_wv{:.0f}_r{:.0f}.dat' \
        .format(model_params["site"], model_params["start_wvl"],
                model_params["end_wvl"], model_params["wv_content"] * 1000,
                model_params["resolution"])

    if not os.path.exists(atran_filename):
        atran_filename = os.path.join(working_dir, atran_linelist)
        log.stdinfo(f"Generating a linelist from ATRAN synthetic spectrum")
        _, refplot_dict = _make_atran_linelist(ext, wave_model, filename=atran_filename,
                                                    model_params=model_params)
    log.stdinfo(f"Using linelist {atran_filename}")
    linelist = wavecal.LineList(atran_filename)
    linelist.reference_spectrum = refplot_dict
    return linelist


def _make_atran_linelist(ext, wave_model, filename, model_params):
    """
    Generates line list from a convolved ATRAN spectrum: finds peaks,
    assings weights, divides spectrum into 10 wavelength bins,
    selects nlines//10 best lines within each bin, pinpoints the peaks.
    Saves the linelist to a file, and creates a disctionary with
    convolved spectrum and line list data to use later for the reference plot.

    Parameters
    ----------
    ext: single-slice AstroData
        the extension
    filename: str or None
        full name of the linelist to be saved on disk. If None, the linelist is not saved.
    model_params: dict
     A dictionary of observation parameters generated by _get_model_params() function.

    Returns
    -------
    atran_linelist: 1d array
        A list of ATRAN line wavelengths in air/vacuum (nm)
    refplot_dict: dict
        a dictionary containing data for the reference plot
    """
    from datetime import datetime
    start = datetime.now()
    atran_spec = _get_convolved_atran(ext, model_params=model_params)
    print(datetime.now() - start, "CONVOLVED")
    sampling = abs(np.diff(atran_spec[:, 0]).mean())

    inverse_atran_spec = 1 - atran_spec[:, 1]
    fwhm = (model_params["cenwave"] / model_params["resolution"]) / sampling
    pixel_peaks, properties = find_peaks(
        inverse_atran_spec, prominence=0.001, width=(None, 5 * fwhm))
    print(datetime.now() - start, "FOUND PEAKS")
    weights = properties["prominences"] / properties["widths"]

    def trim_peaks(peaks, weights, bin_edges, nlargest=10, sort=True):
        """
        Filters the peaks list, binning it over the range of the whole
        signal, preserving only the N-largest ones on each bin

        peaks: array
            wavelengths of peaks
        weights: array
            strengths of peaks
        bin_edges: array of shape (N+1,)
            edges of the N desired bins
        nlargest: int
            number of largest peaks to extract from each bin

        Returns: array of shape (M, 2)
            the M (M <= N * nlargest) line wavelengths and weights
        """
        result = []
        for wstart, wend in zip(bin_edges[:-1], bin_edges[1:]):
            indices = np.logical_and(peaks >= wstart, peaks < wend)
            indices_to_keep = weights[indices].argsort()[-nlargest:]
            result.extend(list(zip(peaks[indices][indices_to_keep],
                                   weights[indices][indices_to_keep])))
        return np.array(sorted(result) if sort else result,
                        dtype=peaks.dtype)

    # For the final line list select n // 10 peaks with largest weights
    # within each of 10 wavelength bins.
    nbins = 10
    bin_edges = np.linspace(atran_spec[:, 0].min(),
                            atran_spec[:, 0].max() + sampling, nbins + 1)
    best_peaks = trim_peaks(atran_spec[pixel_peaks, 0], weights, bin_edges,
                            nlargest=model_params["nlines"] // nbins, sort=True)
    print(datetime.now() - start, "TRIMMED PEAKS")

    # Pinpoint peak positions (need pixel positions), and cull any
    # peaks that couldn't be fit (keep_bad will return location=None)
    best_pixel_peaks = [np.argmin(abs(atran_spec[:, 0] - p))
                        for p in best_peaks[:, 0]]
    atran_linelist = np.vstack(peak_finding.pinpoint_peaks(
        inverse_atran_spec, peaks=best_pixel_peaks,
        halfwidth=1, keep_bad=True)).T
    atran_linelist = atran_linelist[~np.isnan(atran_linelist).any(axis=1)]

    # Convert from peak locations in pixels to wavelengths
    atran_linelist[:, 0] = np.interp(atran_linelist[:, 0],
                                     np.arange(atran_spec.shape[0]),
                                     atran_spec[:, 0])
    print(datetime.now() - start, "MADE LINELIST")

    if model_params["absorption"]:
        atran_linelist[:, 1] = 1 - atran_linelist[:, 1]

    if filename is not None:
        header = (f"Sky emission line list: {model_params['start_wvl']:.0f}-{model_params['end_wvl']:.0f}nm   \n"
                  "Generated from ATRAN synthetic spectrum (Lord, S. D., 1992, NASA Technical Memorandum 103957) \n"
                  "Model parameters: \n"
                  f"Obs altitude: {model_params['alt']}, Obs latitude: 39 degrees,\n"
                  f"Water vapor overburden: {model_params['wv_content'] * 1000:.0f} microns, Number of atm. layers: 2,\n"
                  "Zenith angle: 48 deg, Wavelength range: 1.0 - 6.0 microns, Smoothing R:0 \n"
                  "units nanometer\n"
                  "wavelengths IN VACUUM")
        np.savetxt(filename, atran_linelist, fmt=['%.3f', '%.3f'], header=header)

    # Create data for reference plot using the convolved spectrum and the line
    # list generated here
    refplot_dict = _make_refplot_data(wave_model, ext, refplot_spec=atran_spec,
                                      refplot_linelist=atran_linelist, model_params=model_params)
    return atran_linelist, refplot_dict


def _get_convolved_atran(ext, model_params):
    """
    Reads a section of an appropriate ATRAN model spectrum file,
    convolves the spectrum to the resolution of the observation.

    Parameters
    ----------
    ext: single-slice AstroData
        the extension
    model_params: dict
     A dictionary of observation parameters generated by _get_model_params() function.

    Returns
    -------
    spectrum : ndarray
        2d array of wavelengths (nm) and atmospheric transmission values
    sampling: float
        sampling (nm) of the ATRAN model spectrum
    """
    path = list(atran_models.__path__).pop()

    atran_model_filename = os.path.join(path,
                                        'atran_{}_850-6000nm_wv{:.0f}_za48_r0.dat'
                                        .format(model_params["site"],
                                                model_params["wv_content"] * 1000))
    full_atran_spec = np.loadtxt(atran_model_filename)
    atran_spec = full_atran_spec[np.logical_and(
        full_atran_spec[:, 0] >= model_params["start_wvl"],
        full_atran_spec[:, 0] <= model_params["end_wvl"])]
    # Smooth the atran spectrum with a Gaussian with a constant FWHM value,
    # where FWHM = cenwave / resolution
    sampling = abs(np.diff(atran_spec[:, 0]).mean())
    fwhm = (model_params["cenwave"] / model_params["resolution"]) / sampling
    gauss_kernel = Gaussian1DKernel(fwhm / 2.35)  # std = FWHM / 2.35
    atran_spec[:, 1] = convolve(atran_spec[:, 1], gauss_kernel,
                                boundary='extend')
    return atran_spec


def _show_refplot(ext):
    """
    Reference plots are implemented for the wavecal from sky lines
    """
    return False if 'ARC' in ext.tags else True


def _make_refplot_data(wave_model, ext, refplot_linelist, config=None, model_params=None,
                       refplot_spec=None, refplot_name=None, refplot_y_axis_label=None):
    """
    Generate data for the reference plot (reference spectrum, reference plot name, and
    the label for the y-axis), depending on the type of spectrum used for wavelength calibration,
    using the supplied line list.

    Parameters
    ----------
    ext: single-slice AstroData
        the extension
    model_params: dict or None
     A dictionary of observation parameters generated by _get_model_params() function.
     Either model_params or config must be not None
    config: Config-like object containing parameters or None
        Either model_params or config must be not None
    refplot_linelist: LineList object or a 2-column nd-array with wavelengths
            (nm) and line intensities
    refplot_spec: 2d-array or None
        two-column nd.array containing reference spectrum wavelengths (nm) and intensities
    refplot_name: str or None
        reference plot name string
    refplot_y_axis_label: str or None
        reference plot y-axis label string

    Returns
    -------
    dict : all the information needed to construct reference spectrum plot:
    "refplot_spec" : two-column nd.array containing reference spectrum wavelengths
                    (nm) and intensities
    "refplot_linelist" : two-column nd.array containing line wavelengths (nm) and
                        intensities
    "refplot_name" : reference plot name string
    "refplot_y_axis_label" : reference plot y-axis label string
    """

    if config is None and model_params is None:
        raise ValueError('Either "config" or "model_params" must be not None')
    if model_params is None:
        model_params = _get_model_params(wave_model, ext, config)
    resolution = model_params["resolution"]
    absorption = model_params["absorption"]

    if _show_refplot(ext):
        # Use ATRAN models for generating reference plots
        # for wavecal from sky absorption lines in science spectrum,
        # and for wavecal from sky emission lines in L- and M-bands
        if _uses_atran_linelist(absorption=absorption,
                                cenwave=ext.central_wavelength(asNanometers=True)):
            if refplot_spec is None:
                refplot_spec = _get_convolved_atran(ext, model_params)
            if refplot_y_axis_label is None:
                if absorption:
                    refplot_y_axis_label = "Atmospheric transmission"
                else:
                    refplot_y_axis_label = "Inverse atm. transmission"
                    refplot_spec[:, 1] = 1 - refplot_spec[:, 1]
            if refplot_name is None:
                refplot_name = 'ATRAN spectrum (Alt={}, WV={:.1f}mm, AM=1.5, R={:.0f})' \
                    .format(model_params["alt"], model_params["wv_content"], resolution)
        else:
            # Use a set of pre-calculated synthetic spectra for the reference plots
            # for wavecal from the OH emission sky lines
            raise NotImplementedError("SHOULDN'T GET HERE")

    # if isinstance(refplot_linelist, wavecal.LineList):
    #     # If the provided linelist is a LineList-type object, then it wasn't
    #     # generated on-the-fly and its format has to be adjusted.
    #     # For the refplot use only line wavelengths, and determine
    #     # reference spectrum intensities at the positions of lines in the linelist
    #     # (this is needed to determine the line label positions).
    #     line_intens = []
    #     line_wvls = refplot_linelist.wavelengths(in_vacuo=model_params["in_vacuo"], units="nm")
    #     for n, line in enumerate(line_wvls):
    #         subtracted = refplot_spec[:, 0] - line
    #         min_index = np.argmin(np.abs(subtracted))
    #         line_intens.append(refplot_spec[min_index, 1])
    #     refplot_linelist = np.array([line_wvls, line_intens]).T
    #     refplot_linelist = refplot_linelist[np.logical_and(refplot_linelist[:, 0] >= start_wvl,
    #                                                        refplot_linelist[:, 0] <= end_wvl)]

    return {"refplot_spec": refplot_spec, "refplot_linelist": refplot_linelist,
            "refplot_name": refplot_name, "refplot_y_axis_label": refplot_y_axis_label}


def _get_model_params(self, wave_model=None, ext=None, config=None):
    """
    Get the observation parameters needed to select the appropriate
    ATRAN (or other) model.

    Parameters
    ----------
    ext: single-slice AstroData
        the extension
    config: Config-like object containing parameters

    Returns
    ----------
    dict : all the parameters needed for ATRAN model selection and line
    list generation:
    "resolution" : resolution to which ATRAN model spectrum should be convolved (int)
    "cenwave" : *actual* central wavelength of the observation (float)
    "in_vacuo" : config["in_vacuo"]
    "nlines" : maximum number of highest-weight lines in the ATRAN spectrum to
        be included in the generated linelist (int)
    "absorption" : are absorption features used for wavecal? (bool)
    wv_content: float
        WV band constraints (in mm of precipitable H2O at zenith)
    site: str
        observation site
    alt: str
        observatory altitude
    start_wvl: float
         start wavelength of the spectrum (nm) with a small buffer
    end_wvl: float
        start wavelength of the spectrum (nm) with a small buffer
    spec_range: float
        spectral range with buffer (nm)
    """

    if config.get("resolution", None) is None:
        resolution = round(self._get_resolution(ext), -1)
    else:
        resolution = config["resolution"]

    if config["central_wavelength"] is None:
        cenwave = ext.wcs(*(0.5 * np.array(ext.shape[::-1])))[0]
    else:
        cenwave = config["central_wavelength"]

    in_vacuo = config["in_vacuo"]
    num_atran_lines = config.get("num_atran_lines", 50)
    absorption = config.get("absorption", False)

    dispaxis = 2 - ext.dispersion_axis()  # python sense
    npix = ext.shape[dispaxis]
    dcenwave = np.diff(self._wavelength_model_bounds(wave_model, ext)['c0'])[0]

    spec_range = dcenwave + abs(ext.dispersion(asNanometers=True)) * npix
    start_wvl = cenwave - (0.5 * spec_range)
    end_wvl = start_wvl + spec_range
    observatory = ext.phu['OBSERVAT']
    param_wv_band = config.get("wv_band", "header")
    if param_wv_band == "header":
        wv_band = ext.raw_wv()
    else:
        wv_band = int(param_wv_band)

    if wv_band is None:
        req_wv = ext.requested_wv()
        log.stdinfo("Unknown RAWWV for this observation; \n"
                    f" using the constraint value for the requested WV band: '{req_wv}'-percentile")
        wv_band = req_wv

    if wv_band == 100:
        # a WV value to use for the case of RAWWV='Any'
        if observatory == 'Gemini-North':
            wv_content = 5.
        elif observatory == 'Gemini-South':
            wv_content = 10.
    else:
        wv_content = qa_constraints.wvBands.get(observatory).get(str(wv_band))

    if observatory == 'Gemini-North':
        site = 'mk'
        alt = "13825ft"
    elif observatory == 'Gemini-South':
        site = 'cp'
        alt = "8980ft"
    return {"site": site, "alt": alt, "start_wvl": start_wvl, "end_wvl": end_wvl,
            "spec_range": spec_range, "wv_content": wv_content,
            "resolution": resolution, "cenwave": cenwave, "in_vacuo": in_vacuo,
            "absorption": absorption, "nlines": num_atran_lines}


