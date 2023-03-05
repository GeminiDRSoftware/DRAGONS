#
#                                                                  gemini_python
#
#                                                        primtives_gnirs_spect.py
# ------------------------------------------------------------------------------
import os
import tempfile
import numpy as np


from importlib import import_module

from astropy.convolution import convolve, Gaussian1DKernel

from geminidr.core import Spect

from gemini_instruments.gnirs import lookup

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
        self.refplot_tempfile = None

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
        for ad in adinputs:
            disp = ad.disperser(pretty=True)
            filt = ad.filter_name(pretty=True)
            cam = ad.camera(pretty=True)
            cenwave = ad.central_wavelength(asMicrometers=True)
            log = self.log

            if 'ARC' in ad.tags:
                if params["min_snr"] is None:
                    params["min_snr"] = 20
                    self.log.stdinfo(f'Parameter "min_snr" is set to None. Using min_snr={params["min_snr"]}')
                if params["debug_min_lines"] is None:
                    params["debug_min_lines"] = 100000

                if params["order"] is None:
                    if ((filt == "H" and cenwave >= 1.75) or (filt == "K" and cenwave >= 2.2)) \
                            and ((cam.startswith('Long') and disp.startswith('32')) or
                                 (cam.startswith('Short') and disp.startswith('111'))):
                            params["order"] = 1
                    elif disp.startswith('111') and cam.startswith('Long'):
                            params["order"] = 1
                    else:
                        params["order"] = 3
                    self.log.stdinfo(f'Parameter "order" is set to None. Using order={params["order"]}')
            else:
                # works better with ATRAN line lists
                params["lsigma"] = 2
                params["hsigma"] = 2

                if params["debug_min_lines"] is None:
                    params["debug_min_lines"] = 15

                if params["absorption"] is True:
                    # Telluric absorption case
                    if params["order"] is None:
                        params["order"] = 1
                        self.log.stdinfo(f'Parameter "order" is set to None. Using order={params["order"]}')

                    if params["min_snr"] is None:
                        params["min_snr"] = 1
                        self.log.stdinfo(f'Parameter "min_snr" is set to None. Using min_snr={params["min_snr"]}')

                    if params["center"] is None:
                        try:
                            aptable = ad[0].APERTURE
                            params["center"] = int(aptable['c0'].data[0])
                        except (AttributeError, KeyError):
                            log.error("Could not find aperture locations in "
                                        f"{ad.filename} - continuing")
                            continue
                        self.log.stdinfo(f'Extracting spectrum from columns '
                                    f'[{params["center"]-params["nsum"]}:{params["center"]+params["nsum"]}]')
                else:
                    # Telluric emission case
                    if params["order"] is None:
                        if ad.camera(pretty=True).startswith('Long') and \
                                ad.disperser(pretty=True).startswith('111') and \
                                3.65 <= cenwave <= 3.75:
                                params["order"] = 1
                        else:
                         params["order"] = 3
                        self.log.stdinfo(f'Parameter "order" is set to None. Using order={params["order"]}')
                    if params["min_snr"] is None:
                        params["min_snr"] = 10
                        self.log.stdinfo(f'Parameter "min_snr" is set to None. Using min_snr={params["min_snr"]}')
        adinputs = super().determineWavelengthSolution(adinputs, **params)
        return adinputs

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
                if 'ARC' in ad.tags:
                    if disp.startswith('111') and cam.startswith('Long') and \
                            cenwave >= 1.65:
                            params["spectral_order"] = 1
                    else:
                        params["spectral_order"] = 2
                else:
                # sky line case
                    params["spectral_order"] = 3
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

    def _get_refplot_data(self, ad=None, config=None):
        """
        Generates data necessary for displaying reference plot used to verify line
        identification.

        Since the reference spectrum stays unchanged when resetting the view or
        reconstructing points, after generating the data for the plot gets saved to
        a temporary file, that then is read whenever the spectrum is being reloaded.

        So far this has been implemented only for the modes for which ATRAN model
        spectra can be used as a reference:
        - wavecal from telluric absorption in XJHK-bands;
        - wavecal from telluric "emission" (which in fact is absence of absorption) in
            L- and M-band.

        Returns
        -------
        dict : all the information needed to construct the reference spectrum plot:
        "refplot_spec" : two-column nd.array containing reference spectrum wavelengths
                        (nm) and intensities
        "refplot_linelist" : two-column nd.array containing line wavelengths (nm) and
                            intensities
        "refplot_name" : reference plot name string
        "refplot_y_axis_label" : reference plot y-axis label string
        """
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)
        if 'ARC' in ad.tags:
            # Not yet implemented for arc spectra.
            return None

        else:
            is_atran_spec = False
            if config.absorption:
                # Unconvolved ATRAN spectrum for XJHK-bands, wvl = 1000-2500nm,
                # resampled to 0.02 nm.
                # ATRAN model params: Alt: 14000ft, WV=2mm, ZA=48deg, R=0
                is_atran_spec = True
                ref_spec_file = "atran_1000-2500_uncolvolved_0_02_nm.dat"
                sampling = 0.02
                spec_start_wvl = 1000
                refplot_y_axis_label = "Atmospheric transmission"

            else:
                filter_name = ad.filter_name(pretty=True)
                if filter_name.startswith('M') or filter_name.startswith('L'):
                    # Unconvolved inverse ATRAN spectrum for L and M bands,
                    # wvl = 2800-6000nm, resampled to 0.02 nm.
                    # ATRAN model params: Alt: 14000ft, WV=2mm, ZA=48deg, R=0
                    is_atran_spec = True
                    ref_spec_file="atran_2800-6000_uncolvolved_inverse_0_02_nm.dat"
                    sampling = 0.02
                    spec_start_wvl = 2800
                    refplot_y_axis_label = "Inverse atm. transmission"
                else:
                    # Not yet implemented for wavecal from the OH emission lines in XJHK-bands.
                    return None

        if self.refplot_tempfile is None:
            # Set the wavelength range of the reference spectrum at the observation's
            # wvl range plus 10% on eah side.
            spec_range = 1.2 * abs(ad.dispersion(asNanometers=True)) * 1024
            central_wavelength = ad.central_wavelength(asNanometers=True)
            start_wvl = central_wavelength - (0.5 * spec_range)
            end_wvl = start_wvl + spec_range
            resolution = self._get_resolution(ad)

            linelist =self._get_arc_linelist(ad=ad,config=config).wavelengths(config.in_vacuo,
                                                                              units="nm")
            # Trim the line list to the wvl range of the reference spectrum
            refplot_linelist = linelist[np.logical_and(linelist >= start_wvl,
                                                       linelist <= end_wvl)]
            if is_atran_spec:
                refplot_name = f"ATRAN spectrum (WV=2mm, AM=1.5, Alt=14000ft, R={resolution})"
                ref_spec_file_path = os.path.join(lookup_dir, ref_spec_file)
                start_row = int((start_wvl - spec_start_wvl) / sampling)
                nrows = int(spec_range / sampling)

                # Read a portion of the unconvolved spectrum, then convolve to the resolution
                # of the observation.
                refplot_spec = np.loadtxt(ref_spec_file_path, skiprows=start_row,
                                          max_rows=nrows+1)

                # Smooth the reference spectrum with a Gaussian with a constant FWHM value,
                # where FWHM = cenwave / resolving_power
                fwhm = (central_wavelength / resolution) * (1 / sampling)
                gauss_kernel = Gaussian1DKernel(fwhm / 2.35) # FWHM = 2.35 * std
                refplot_spec[:,1] = convolve(refplot_spec[:,1], gauss_kernel,
                                             boundary='extend')

                # Get ref. spectrum intensities at the positions of lines in the line list
                # (this is needed to determine the line label positions).
                line_intens = []
                for n, line in enumerate(refplot_linelist):
                    subtracted = refplot_spec[:,0] - line
                    min_index = np.argmin(np.abs(subtracted))
                    line_intens.append(refplot_spec[min_index,1])
                refplot_linelist = np.array([refplot_linelist, line_intens]).T

                # Write the results to a temporary file, to be used when reloading the plot,
                # so that we don't have to redo the calculations.
                self.refplot_tempfile = tempfile.TemporaryFile()
                np.save(self.refplot_tempfile, refplot_spec)
                np.save(self.refplot_tempfile, refplot_linelist)
                np.save(self.refplot_tempfile, refplot_name)

            else:
                # Not yet implemented for the modes not using ATRAN spectrum as reference.
                return None
        else:
            # Read the precalculated data
            self.refplot_tempfile.seek(0)
            refplot_spec = np.load(self.refplot_tempfile)
            refplot_linelist = np.load(self.refplot_tempfile)
            refplot_name = np.load(self.refplot_tempfile)

        return {"refplot_spec": refplot_spec, "refplot_linelist": refplot_linelist,
                "refplot_name": refplot_name, "refplot_y_axis_label": refplot_y_axis_label}


    def _get_arc_linelist(self, waves=None, ad=None, config=None):
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
            resolution = self._get_resolution(ad)

            if config.absorption is True:
                if resolution >= 10000:
                    linelist = 'sky_absorp_XJHK_high_res.dat'
                elif (3000 <= resolution < 10000):
                    linelist = "sky_absorp_XJHK_med_res.dat"
                elif (1000 <= resolution < 3000):
                    linelist = "sky_absorp_XJHK_low_res.dat"
                elif resolution < 1000:
                    linelist = "sky_absorp_XJHK_very_low_res.dat"
            else:
                if ad.filter_name(pretty=True).startswith('M'):
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

                elif is_lowres:
                    linelist = 'sky.dat'
                else:
                    linelist = 'nearIRsky.dat'

        self.log.stdinfo(f"Using linelist {linelist}")
        filename = os.path.join(lookup_dir, linelist)

        return wavecal.LineList(filename)


    def _get_resolution(self, ad=None):
        if ad.pixel_scale() == lookup.pixel_scale_shrt:
            camera = "Short"
        elif ad.pixel_scale() == lookup.pixel_scale_long:
            camera = "Long"
        else:
            camera = None
        grating = ad._grating(pretty=True, stripID=True)
        filter = str(ad.filter_name(pretty=True))[0]
        config = f"{grating}, {camera}"

        resolution_2pix_slit = lookup.dispersion_and_resolution.get(config, {}).get(filter)[1]
        pix_scale = ad.pixel_scale()
        slit_width_pix = ad.slit_width()/pix_scale

        return resolution_2pix_slit * 2 / slit_width_pix

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
