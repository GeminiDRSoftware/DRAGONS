#
#                                                                  gemini_python
#
#                                                        primtives_gnirs_spect.py
# ------------------------------------------------------------------------------
import os

from importlib import import_module

from geminidr.core import Spect

from .primitives_gnirs import GNIRS
from . import parameters_gnirs_spect

from gempy.gemini import gemini_tools as gt
from gempy.library import transform, wavecal

from recipe_system.utils.decorators import parameter_override, capture_provenance


@parameter_override
@capture_provenance
class GNIRSSpect(Spect, GNIRS):
    """
    This is the class containing all of the preprocessing primitives
    for the GNIRSSpect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "GNIRS", "SPECT"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_spect)

    def standardizeWCS(self, adinputs=None, **params):
        """
        This primitive updates the WCS attribute of each NDAstroData extension
        in the input AstroData objects. For spectroscopic data, it means
        replacing an imaging WCS with an approximate spectroscopic WCS.

        Parameters
        ----------
        suffix: str/None
            suffix to be added to output files

        """
        log = self.log
        timestamp_key = self.timestamp_keys[self.myself()]
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        super().standardizeWCS(adinputs, **params)

        for ad in adinputs:
            log.stdinfo(f"Adding spectroscopic WCS to {ad.filename}")
            cenwave = ad.central_wavelength(asNanometers=True)
            transform.add_longslit_wcs(ad, central_wavelength=cenwave,
                                       pointing=ad[0].wcs(512, 511))

            # Timestamp. Suffix was updated in the super() call
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
        return adinputs

    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        Determines the wavelength solution for an ARC and updates the wcs
        with this solution. In addition, the solution and pixel/wavelength
        matches are stored as an attached `WAVECAL` :class:`~astropy.table.Table`.

        2D input images are converted to 1D by collapsing a slice of the image
        along the dispersion direction, and peaks are identified. These are then
        matched to an arc line list, using piecewise-fitting of (usually)
        linear functions to match peaks to arc lines, using the
        :class:`~gempy.library.matching.KDTreeFitter`.

        The `.WAVECAL` table contains four columns:
            ["name", "coefficients", "peaks", "wavelengths"]

        The `name` and the `coefficients` columns contain information to
        re-create an Chebyshev1D object, plus additional information about
        the way the spectrum was collapsed. The `peaks` column contains the
        (1-indexed) position of the lines that were matched to the catalogue,
        and the `wavelengths` column contains the matched wavelengths.

        This GNIRS-specific primitive sets debug_min_lines, order, min_snr,
        num_atran_lines and average values depending on the
        observing mode, as the default value for these parameters is None.
        It then calls the generic version of the primitive.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
             Mosaicked Arc data as 2D spectral images or 1D spectra.

        suffix : str/None
            Suffix to be added to output files

        order : int
            Order of Chebyshev fitting function.

        center : None or int
            Central row/column for 1D extraction (None => use middle).

        nsum : int, optional
            Number of rows/columns to average.

        min_snr : float
            Minimum S/N ratio in line peak to be used in fitting.

        weighting : {'natural', 'relative', 'none'}
            How to weight the detected peaks.

        fwidth : float/None
            Expected width of arc lines in pixels. It tells how far the
            KDTreeFitter should look for when matching detected peaks with
            reference arcs lines. If None, `fwidth` is determined using
            `tracing.estimate_peak_width`.

        min_sep : float
            Minimum separation (in pixels) for peaks to be considered distinct

        central_wavelength : float/None
            central wavelength in nm (if None, use the WCS or descriptor)

        dispersion : float/None
            dispersion in nm/pixel (if None, use the WCS or descriptor)

        linelist : str/None
            Name of file containing arc lines. If None, then a default look-up
            table will be used.

        alternative_centers : bool
            Identify alternative central wavelengths and try to fit them?

        nbright : int (or may not exist in certain class methods)
            Number of brightest lines to cull before fitting

        absorption : bool
            If feature type is absorption (default: "False")

        interactive : bool
            Use the interactive tool?

        debug : bool
            Enable plots for debugging.

        num_atran_lines: int/None
            Number of lines with largest weigths (within a wvl bin) to be used for
            the generated ATRAN line list.

        wv_band: {'20', '50', '80', '100', 'header'}
            Water vapour content (as percentile) to be used for ATRAN model
            selection. If "header", then the value from the header is used.

        resolution: int/None
            Resolution of the observation (as l/dl), to which ATRAN spectrum should be
            convolved. If None, the default value for the instrument/mode is used.

        debug_combiner: {"mean", "median", "none"}
            Method to use for combining rows/columns when extracting 1D-spectrum.
            Default: "mean".

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Updated objects with a `.WAVECAL` attribute and improved wcs for
            each slice

        See Also
        --------
        :class:`~geminidr.core.primitives_visualize.Visualize.mosaicDetectors`,
        :class:`~gempy.library.matching.KDTreeFitter`,
        """
        adoutputs = []
        for ad in adinputs:
            these_params = params.copy()
            min_snr_isNone = True if these_params["min_snr"] is None else False
            order_isNone = True if these_params["order"] is None else False
            combine_method_isNone = True if these_params["combine_method"] == "optimal" else False

            disp = ad.disperser(pretty=True)
            filt = ad.filter_name(pretty=True)
            cam = ad.camera(pretty=True)
            cenwave = ad.central_wavelength(asMicrometers=True)
            log = self.log

            if 'ARC' in ad.tags:
                if these_params["min_snr"] is None:
                    these_params["min_snr"] = 20
                if these_params["debug_min_lines"] is None:
                    these_params["debug_min_lines"] = 100000

                if these_params["order"] is None:
                    if ((filt == "H" and cenwave >= 1.75) or (filt == "K" and cenwave >= 2.2)) \
                            and ((cam.startswith('Long') and disp.startswith('32')) or
                                 (cam.startswith('Short') and disp.startswith('111'))):
                            these_params["order"] = 1
                    elif disp.startswith('111') and cam.startswith('Long'):
                            these_params["order"] = 1
                    else:
                        these_params["order"] = 3

            elif these_params["absorption"] or ad.central_wavelength(asMicrometers=True) >= 2.8:
                # The case of wavecal from absorption, or wavecal from telluric
                # emission in L- and M-bands, both done using ATRAN lines
                self.generated_linelist = True
                # sigma=2 works better with ATRAN line lists
                these_params["lsigma"] = 2
                these_params["hsigma"] = 2

                if these_params["absorption"]:
                    # Telluric absorption case
                    if these_params["order"] is None:
                        these_params["order"] = 1

                    if these_params["min_snr"] is None:
                        these_params["min_snr"] = 1

                    if these_params["center"] is None:
                        try:
                            aptable = ad[0].APERTURE
                            these_params["center"] = int(aptable['c0'].data[0])
                        except (AttributeError, KeyError):
                            log.error("Could not find aperture locations in "
                                        f"{ad.filename} - continuing")
                            continue

                else:
                    # Telluric emission in L and M-bands
                    if these_params["order"] is None:
                        these_params["order"] = 3

                    if these_params["center"] is None:
                        these_params["center"] = 650

                    if these_params["min_snr"] is None:
                        if filt.startswith('L'):
                            # Use a lower min_snr for the regions with large illumination gradient,
                            # and for the region of "comb"-like lines beyond 3.8 um
                            if (disp.startswith('111') and 3.50 <= cenwave) or \
                                (disp.startswith('111') and cam.startswith('Short') and 3.80 <= cenwave) or \
                                (disp.startswith('32') and cam.startswith('Long') and 3.65 <= cenwave):
                                these_params["min_snr"] = 1
                            else:
                                these_params["min_snr"] = 10
                        else:
                            these_params["min_snr"] = 10

                    if these_params["num_atran_lines"] is None:
                        if filt.startswith('M'):
                            these_params["num_atran_lines"] = 150
                        elif filt.startswith('L'):
                            these_params["num_atran_lines"] = 100
                            if ((disp.startswith('111') and cam.startswith('Short')) or
                                (disp.startswith('32') and cam.startswith('Long'))) and \
                                    3.80 <= cenwave:
                                these_params["num_atran_lines"] = 300

                    if these_params["combine_method"] == "optimal":
                        # this is to reduce the impact of hot pixels
                        if filt.startswith('L') and cenwave >= 3.8:
                            these_params["combine_method"] = "median"
                        else:
                            these_params["combine_method"] = "mean"
            else:
                # OH emission
                if these_params["min_snr"] is None:
                    these_params["min_snr"] = 10
                if these_params["order"] is None:
                    these_params["order"] = 3
                if these_params["center"] is None:
                    these_params["center"] = 650

            if these_params["debug_min_lines"] is None:
                these_params["debug_min_lines"] = 15
            if these_params["num_atran_lines"] is None:
                these_params["num_atran_lines"] = 50
            if these_params["combine_method"] == "optimal":
                these_params["combine_method"] = "mean"

            if min_snr_isNone:
                self.log.stdinfo(f'Parameter "min_snr" is set to None. '
                                 f'Using min_snr={these_params["min_snr"]} for {ad.filename}')
            if order_isNone:
                self.log.stdinfo(f'Parameter "order" is set to None. '
                                 f'Using order={these_params["order"]} for {ad.filename}')
            if combine_method_isNone:
                self.log.stdinfo(f'Parameter "combine_method" is set to "optimal"".'
                                 f' Using "combine_method"={these_params["combine_method"]} for {ad.filename}')
            adoutputs.extend(super().determineWavelengthSolution([ad], **these_params))
        return adoutputs

    def determineDistortion(self, adinputs=None, **params):
        """
        Maps the distortion on a detector by tracing lines perpendicular to the
        dispersion direction. Then it fits a 2D Chebyshev polynomial to the
        fitted coordinates in the dispersion direction. The distortion map does
        not change the coordinates in the spatial direction.

        The Chebyshev2D model is stored as part of a gWCS object in each
        `nddata.wcs` attribute, which gets mapped to a FITS table extension
        named `WCS` on disk.

        This GNIRS-specific primitive sets default spectral order in case it's None
        (since there are only few lines available in H and K-bands in high-res mode, which
        requires setting order to 1), and minimum length of traced feature to be considered
        as a useful line for each pixel scale.
        It then calls the generic version of the primitive.


        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Arc data as 2D spectral images with the distortion and wavelength
            solutions encoded in the WCS.

        suffix :  str
            Suffix to be added to output files.

        spatial_order : int
            Order of fit in spatial direction.

        spectral_order : int
            Order of fit in spectral direction.

        id_only : bool
            Trace using only those lines identified for wavelength calibration?

        min_snr : float
            Minimum signal-to-noise ratio for identifying lines (if
            id_only=False).

        nsum : int
            Number of rows/columns to sum at each step.

        step : int
            Size of step in pixels when tracing.

        max_shift : float
            Maximum orthogonal shift (per pixel) for line-tracing (unbinned).

        max_missed : int
            Maximum number of steps to miss before a line is lost.

        min_line_length: float
            Minimum length of traced feature (as a fraction of the tracing dimension
            length) to be considered as a useful line.

        debug_reject_bad: bool
            Reject lines with suspiciously high SNR (e.g. bad columns)? (Default: True)

        debug: bool
            plot arc line traces on image display window?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has the
            appropriate `nddata.wcs` defined for each of its extensions. This
            provides details of the 2D Chebyshev fit which maps the distortion.
        """
        adoutputs = []
        for ad in adinputs:
            these_params = params.copy()
            disp = ad.disperser(pretty=True)
            cam = ad.camera(pretty=True)
            cenwave = ad.central_wavelength(asMicrometers=True)
            if these_params["spectral_order"] is None:
                if 'ARC' in ad.tags:
                    if disp.startswith('111') and cam.startswith('Long') and \
                            cenwave >= 1.65:
                            these_params["spectral_order"] = 1
                    else:
                        these_params["spectral_order"] = 2
                else:
                # sky line case
                    these_params["spectral_order"] = 3
                self.log.stdinfo(f'Parameter "spectral_order" is set to None. '
                                 f'Using spectral_order={these_params["spectral_order"]} for {ad.filename}')

            if these_params["min_line_length"] is None:
                if cam.startswith('Long'):
                    these_params["min_line_length"] = 0.8
                else:
                    these_params["min_line_length"] = 0.6
                self.log.stdinfo(f'Parameter "min_line_length" is set to None. '
                 f'Using min_line_length={these_params["min_line_length"]} for {ad.filename}')

            if these_params["max_missed"] is None:
                if "ARC" in ad.tags:
                    # In arcs with few lines tracing strong horizontal noise pattern can
                    # affect distortion model.Using a lower max_missed value helps to
                    # filter out horizontal noise.
                    these_params["max_missed"] = 2
                else:
                    # In science frames we want this parameter be set to a higher value, since
                    # otherwise the line might be abandoned when crossing a bright object spectrum.
                    these_params["max_missed"] = 5
                self.log.stdinfo(f'Parameter "max_missed" is set to None. '
                 f'Using max_missed={these_params["max_missed"]} for {ad.filename}')
            adoutputs.extend(super().determineDistortion([ad], **these_params))
        return adoutputs

    def _get_arc_linelist(self, ext, waves=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)

        is_lowres = ext.disperser(pretty=True).startswith('10') or \
                    (ext.disperser(pretty=True).startswith('32') and
                        ext.camera(pretty=True).startswith('Short'))

        if 'ARC' in ext.tags:
            if 'Xe' in ext.object():
                linelist ='Ar_Xe.dat'
            elif "Ar" in ext.object():
                if is_lowres:
                    linelist = 'lowresargon.dat'
                else:
                    linelist = 'argon.dat'
            else:
                raise ValueError(f"No default line list found for {ext.object()}-type arc. Please provide a line list.")

        else:
            # In case of wavecal from sky OH emission use this line list:
            linelist = 'nearIRsky.dat'

        self.log.stdinfo(f"Using linelist {linelist}")
        filename = os.path.join(lookup_dir, linelist)

        return wavecal.LineList(filename)

    def _apply_wavelength_model_bounds(self, model=None, ext=None):
        # Apply bounds to an astropy.modeling.models.Chebyshev1D to indicate
        # the range of parameter space to explore
        dispaxis = 2 - ext.dispersion_axis()
        npix = ext.shape[dispaxis]
        for i, (pname, pvalue) in enumerate(zip(model.param_names, model.parameters)):
            if i == 0:  # central wavelength
                if 'ARC' in ext.tags or ext.filter_name(pretty=True)[0] in 'LM':
                    prange = 10
                else:
                    prange = abs(ext.dispersion(asNanometers=True)) * npix * 0.07
            elif i == 1:  # half the wavelength extent (~dispersion)
                prange = 0.02 * abs(pvalue)
            else:  # higher-order terms
                prange = 1
            getattr(model, pname).bounds = (pvalue - prange, pvalue + prange)
