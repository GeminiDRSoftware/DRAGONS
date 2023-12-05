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

        This GNIRS-specific primitive sets the default order in case it's None.
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
            disp = ad.disperser(pretty=True)
            filt = ad.filter_name(pretty=True)
            cam = ad.camera(pretty=True)
            cenwave = ad.central_wavelength(asMicrometers=True)

            if 'ARC' in ad.tags:
                if these_params["min_snr"] is None:
                    these_params["min_snr"] = 20
                    self.log.stdinfo('Parameter "min_snr" is set to None. '
                                     f'Using min_snr={these_params["min_snr"]} for {ad.filename}')
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
                    self.log.stdinfo('Parameter "order" is set to None. '
                                     f'Using order={these_params["order"]} for {ad.filename}')
            else:
                these_params["lsigma"] = 2
                these_params["hsigma"] = 2

                if these_params["debug_min_lines"] is None:
                    these_params["debug_min_lines"] = 15

                if these_params["order"] is None:
                    if ad.camera(pretty=True).startswith('Long') and \
                            ad.disperser(pretty=True).startswith('111') and \
                            3.65 <= cenwave <= 3.75:
                            these_params["order"] = 1
                    else:
                        these_params["order"] = 3
                    self.log.stdinfo('Parameter "order" is set to None. '
                                     f'Using order={these_params["order"]} for {ad.filename}')
                if these_params["min_snr"] is None:
                    these_params["min_snr"] = 10
                    self.log.stdinfo('Parameter "min_snr" is set to None. '
                                     f'Using min_snr={these_params["min_snr"]} for {ad.filename}')
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
        for ad in adinputs:
            disp = ad.disperser(pretty=True)
            cam = ad.camera(pretty=True)
            cenwave = ad.central_wavelength(asMicrometers=True)
            if params["spectral_order"] is None:
                if disp.startswith('111') and cam.startswith('Long') and \
                        cenwave >= 1.65:
                        params["spectral_order"] = 1
                else:
                    params["spectral_order"] = 2
                self.log.stdinfo(f'Parameter "spectral_order" is set to None. '
                                 f'Using spectral_order={params["spectral_order"]}')

            if params["min_line_length"] is None:
                if cam.startswith('Long'):
                    params["min_line_length"] = 0.8
                else:
                    params["min_line_length"] = 0.6
                self.log.stdinfo(f'Parameter "min_line_length" is set to None. '
                 f'Using min_line_length={params["min_line_length"]}')
        adinputs = super().determineDistortion(adinputs, **params)
        return adinputs


    def _get_arc_linelist(self, waves=None, ad=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)

        is_lowres = ad.disperser(pretty=True).startswith('10') or \
                    (ad.disperser(pretty=True).startswith('32') and
                        ad.camera(pretty=True).startswith('Short'))

        if 'ARC' in ad.tags:
            if 'Xe' in ad.object():
                linelist ='Ar_Xe.dat'
            elif "Ar" in ad.object():
                if is_lowres:
                    linelist = 'lowresargon.dat'
                else:
                    linelist = 'argon.dat'
            else:
                raise ValueError(f"No default line list found for {ad.object()}-type arc. Please provide a line list.")

        else:
            if ad.filter_name(pretty=True).startswith('M'):
                resolution = self._get_resolution(ad)
                if resolution >= 5000:
                    linelist = 'sky_M_band_high_res.dat'
                elif (2000 <= resolution < 5000):
                    linelist = 'sky_M_band_med_res.dat'
                elif (500 <= resolution < 2000):
                    linelist = 'sky_M_band_low_res.dat'
                elif resolution < 500:
                    linelist = 'sky_M_band_very_low_res.dat'
            elif ad.filter_name(pretty=True).startswith('L'):
                resolution = self._get_resolution(ad)
                if resolution >=10000:
                    linelist = 'sky_L_band_high_res.dat'
                elif (3000 <= resolution < 10000):
                    linelist = 'sky_L_band_med_res.dat'
                elif (1000 <= resolution < 3000):
                    linelist = 'sky_L_band_low_res.dat'
                elif resolution < 1000:
                    linelist = 'sky_L_band_very_low_res.dat'

            else:
                linelist = 'nearIRsky.dat'

        self.log.debug(f"Using linelist '{linelist}'")
        filename = os.path.join(lookup_dir, linelist)

        return wavecal.LineList(filename)


    def _get_resolution(self, ad=None):
        resolution_2pix_slit = {('M, 10/mm, 0.05'): 1200,
                                ('M, 32/mm, 0.15'): 1240,
                                ('M, 32/mm, 0.05'): 3700,
                                ('M, 111/mm, 0.15'): 4300,
                                ('M, 111/mm, 0.05'): 12800,
                                ('L, 10/mm, 0.05'): 1800,
                                ('L, 32/mm, 0.15'): 1800,
                                ('L, 32/mm, 0.05'): 5400,
                                ('L, 111/mm, 0.15'): 6400,
                                ('L, 111/mm, 0.05'): 19000}

        filter = str(ad.filter_name(pretty=True))[0]
        grating = ad._grating(pretty=True, stripID=True)
        pix_scale = ad.pixel_scale()
        config = f"{filter}, {grating}, {pix_scale}"

        resolution_2pix = resolution_2pix_slit.get(config)
        slit_width_pix = ad.slit_width()/pix_scale

        return resolution_2pix * 2 / slit_width_pix

    def _get_cenwave_accuracy(self, ad=None):
        # Accuracy of central wavelength (nm) for a given instrument/setup.
        # According to GNIRS instrument pages "wavelength settings are accurate
        # to better than 5 percent of the wavelength coverage".
        # However using 7% covers more cases. For the arcs dc0=10 works just fine for all modes.

        mband = ad.filter_name(pretty=True).startswith('M')
        lband = ad.filter_name(pretty=True).startswith('L')

        if 'ARC' in ad.tags or not (mband or lband):
            dcenwave = 10
        else:
            dcenwave = abs(ad.dispersion(asNanometers=True)) * 1024 * 0.07
        return dcenwave
