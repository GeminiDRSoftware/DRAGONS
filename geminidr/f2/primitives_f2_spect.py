#
#                                                                  gemini_python
#
#                                                          primtives_f2_spect.py
# ------------------------------------------------------------------------------

import os

from importlib import import_module

from astropy.modeling import models

from geminidr.core import Spect, Telluric
from .primitives_f2 import F2
from . import parameters_f2_spect
from gemini_instruments.f2.lookup import dispersion_offset_mask, resolving_power

from gempy.gemini import gemini_tools as gt
from gempy.library import transform, wavecal
from gemini_instruments import gmu
from geminidr import CalibrationNotFoundError

from recipe_system.utils.decorators import parameter_override, capture_provenance
from ..gsaoi.tests.test_gsaoi_image import adinputs


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class F2Spect(Telluric, Spect, F2):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Spect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "F2", "SPECT"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2_spect)

    def darkCorrect(self, adinputs=None, **params):
        """
        This primitive performs dark subtraction on F2 spectroscopic data.
        It differs from the generic version in that it will look for a
        lamp-off flat in the list of inputs and use that to subtract the
        dark and thermal components instead of looking for a processed
        dark.  This case happens in HK-HK and R3K-Klong configurations as
        far as we know.

        Parameters
        ----------
        suffix: str
            The suffix to be added to the output file.
        dark: list of :class:`~astrodata.AstroData` or None
            List of dark frames to be used for the correction. If None,
            the calibration database will be searched for an appropriate
            dark frame.
        do_cal: str
            Whether to search for calibration frames in the calibration
            database. Options are 'yes', 'no', and 'procmode'. In the last
            case, calibration frames will be searched for only if the input
            frames are processed.
        """

        to_correct = []
        if 'ARC' in adinputs[0].tags and params['dark'] is None:
            lampoffs = []
            for ad in adinputs:
                if 'LAMPOFF' in ad.tags:
                    lampoffs.append(ad)
                else:
                    to_correct.append(ad)
            if lampoffs:
                log = self.log
                log.fullinfo("Using lamp-off flat(s) for dark correction.")
                # Check that all exposure times match
                exptimes = [ad.exposure_time() for ad in adinputs]
                if not all(abs(et - exptimes[0]) < 0.01 for et in exptimes[1:]):
                    raise ValueError("Lamp-off flats and arcs do not have the same exposure time")
                # Stack lamp-offs
                stack_params = self._inherit_params(params, "stackFrames")
                stack_params.update({'zero': False, 'scale': False})
                self.showInputs(lampoffs, purpose='lamp-off flats for dark correction')
                lampoff_stack = self.stackFrames(lampoffs, **stack_params)
                params['dark'] = [lampoff_stack[0]]
        else:
            to_correct = adinputs

        return super().darkCorrect(to_correct, **params)


    def makeLampFlat(self, adinputs=None, **params):
        """
        This produces an appropriate stacked F2 spectroscopic flat, based on
        the inputs. For F2 spectroscopy, lamp-on flats have the dark current
        removed by subtracting darks, but lamp-off flats are also allowed.

        Parameters
        ----------
        suffix: str
            The suffix to be added to the output file.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        suffix = params["suffix"]

        # Since this primitive needs a reference, it must no-op without any
        if not adinputs:
            return adinputs

        def check_exposure_times(adlist):
            return all(abs(ad.exposure_time() - adlist[0].exposure_time()) < 0.01
                       for ad in adlist[1:])

        flat_list = self.selectFromInputs(adinputs, tags='FLAT')
        flat_on_list = self.selectFromInputs(flat_list, tags='LAMPON')
        flat_off_list = self.selectFromInputs(flat_list, tags='LAMPOFF')
        dark_list = self.selectFromInputs(adinputs, tags='DARK')
        if not flat_on_list:
            raise ValueError("No lamp-on flats have been provided")
        if dark_list and flat_on_list and flat_off_list:
            log.warning("Darks have been provided as well as lamp-on and "
                        "lamp-off flats. Ignoring the darks.")
            dark_list = []
        stack_params = self._inherit_params(params, "stackFrames")
        if dark_list:
            self.showInputs(dark_list, purpose='darks')
            dark_list = self.stackDarks(dark_list, **stack_params)
        stack_params.update({'zero': False, 'scale': False})
        if flat_off_list:
            self.showInputs(flat_off_list, purpose='lamp-off flats')
            if not check_exposure_times(flat_off_list):
                raise ValueError("Lamp-off flats are not of equal exposure time")
            flat_off_list = self.stackFrames(flat_off_list, **stack_params)
        self.showInputs(flat_on_list, purpose='lamp-on flats')
        if not check_exposure_times(flat_on_list):
            raise ValueError("Lamp-on flats are not of equal exposure time")
        flat_on_list = self.stackFrames(flat_on_list, **stack_params)

        if flat_off_list or dark_list:
            if dark_list:
                subtype = "dark"
                sublist = dark_list
                if not check_exposure_times(flat_on_list + dark_list):
                    log.warning("Lamp-on flats and darks do not have the same exposure time.")
            else:
                subtype = "lamp-off flat"
                sublist = flat_off_list
                if not check_exposure_times(flat_on_list + flat_off_list):
                    log.warning("Lamp-on and lamp-off flats do not have the same exposure time.")
            log.stdinfo(f"Subtracting stacked {subtype} from stacked lamp-on flat")
            flat = flat_on_list.pop()
            flat.subtract(sublist.pop())
            flat.update_filename(suffix=suffix, strip=True)
            return [flat]

        else:  # Only lamp-on flats were passed.
            # Look for dark in calibration manager; if not found, crash.
            log.fullinfo("Only had flats to stack. Calling darkCorrect.")
            flat_on_list = self.darkCorrect(flat_on_list, suffix=suffix,
                                         dark=None, do_cal='procmode')
            if flat_on_list[0].phu.get('DARKIM') is None:
                # No dark was subtracted by darkCorrect:
                raise CalibrationNotFoundError(
                    "No processed dark found in calibration database. Please "
                    "either provide one, or include a list of darks as input.")
            return flat_on_list

    def standardizeWCS(self, adinputs=None, **params):
        """
        This primitive updates the WCS attribute of each NDAstroData extension
        in the input AstroData objects. For spectroscopic data, it means
        replacing an imaging WCS with an approximate spectroscopic WCS.

        This is an F2-specific primitive due to the need to apply an offset to the
        central wavelength derived from image header, which for F2 is specified for the middle of
        the grism+filter transmission window, not for the central row.

        Parameters
        ----------
        suffix: str/None
            suffix to be added to output files
        """
        super().standardizeWCS(adinputs, **params)
        for ad in adinputs:
            self._add_longslit_wcs(ad, pointing="center")
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

        This F2-specific primitive sets the max_missed value, since we want it to be
        low for arcs (to filter out horizontal noise), and larger for the
        science frames, to not to loose lines when crossing the object spectrum.
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

    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        This F2-specific primitive sets the default order in case it's None.
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
            `peak_finding.estimate_peak_width`.

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

        num_lines: int/None
            Number of lines with largest weigths (within a wvl bin) to be used for
            the generated line list.

        wv_band: {'20', '50', '80', '100', 'header'}
            Water vapour content (as percentile) to be used for ATRAN model
            selection. If "header", then the value from the header is used.

        resolution: int/None
            Resolution of the observation (as l/dl), to which ATRAN spectrum should be
            convolved. If None, the default value for the instrument/mode is used.

        combine_method: {"mean", "median", "optimal"}
            Method to use for combining rows/columns when extracting 1D-spectrum.
            Default: "optimal".

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Updated objects with a `.WAVECAL` attribute and improved wcs for
            each slice
        """
        log=self.log
        adoutputs = []
        for ad in adinputs:
            these_params = params.copy()

            # The peak locations are slightly dependent on the width of the
            # Ricker filter used in the peak finding, so control that
            try:
                slitwidth = int(ad.focal_plane_mask().replace('pix-slit', ''))
            except AttributeError:  # fpm() is returning None
                log.warning(f"Cannot determine slit width for {ad.filename}")
            else:
                these_params["fwidth"] = 2 + 0.5 * slitwidth
                log.stdinfo(f"Setting fwidth={these_params['fwidth']}")

            min_snr_isNone = True if these_params["min_snr"] is None else False

            if "ARC" in ad.tags:
                if these_params["min_snr"] is None:
                    these_params["min_snr"] = 10
            else:
                # Telluric absorption in object spectrum
                if these_params.get("absorption", False):
                    self.generated_linelist = "atran"
                    these_params["lsigma"] = 2
                    these_params["hsigma"] = 2
                    if these_params["min_snr"] is None:
                        these_params["min_snr"] = 1
                else:
                    # OH emission
                    self.generated_linelist = "airglow"
                    if these_params["min_snr"] is None:
                        these_params["min_snr"] = 10

            if min_snr_isNone:
                log.stdinfo('Parameter "min_snr" is set to None. '
                            f'Using min_snr={these_params["min_snr"]} '
                            f'for {ad.filename}')
            
            adoutputs.extend(super().determineWavelengthSolution([ad], **these_params))
        return adoutputs

    def _get_linelist(self, wave_model=None, ext=None, config=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)
        isHK_JH = (ext.disperser(pretty=True) == "HK" and
                   ext.filter_name(pretty=True) == "JH")
        if 'ARC' in ext.tags:
            # For HK grism + JH filter the second order is preserved by default
            # (as per IS request), so use the line list with the second order lines
            # for this mode
            filename = 'lowresargon_with_2nd_ord.dat' if isHK_JH else 'argon.dat'
        else:
            # In case of wavecal from sky OH emission use this line list:
            # filename = 'nearIRsky_with_2nd_order.dat' if isHK_JH else 'nearIRsky.dat'
            return self._get_airglow_linelist(wave_model=wave_model, ext=ext, config=config)

        self.log.stdinfo(f"Using linelist '{filename}'")
        linelist = wavecal.LineList(os.path.join(lookup_dir, filename))

        return linelist

    @staticmethod
    def _convert_peak_to_centroid(ext):
        """
        Returns a function that converts the location of the peak of an arc
        (or sky) line to the location of its centroid (also works on an array
        of locations). This is required due to the poor and spatially-varying
        line spread function of F2, which produces skewed lines. It's used in
        the wavecal routine to ensure that the wavelength solution is correct
        in terms of line centroids. This can be lead to incorrect results if
        the wavelength of a line in the science data is measured from its peak
        (or nadir).

        These cubic polynomials have all been derived by fitting to shifts
        measured from synthetic data using the LSF in geminidr.f2.lookups.lsf
        with the constraint that the shift is zero at row 1061 (therefore
        the Chebyshev coefficients c0 and c2 are the same, and c0 is not
        listed).

        Parameters
        ----------
        ext: single-slice AstroData
            the extension for which the shifts are to be calculation

        Returns
        -------
        callable:
            a callable that modifies a pixel value (or array thereof) of a
            line peak to the centroid
        """
        # (c1, c3) Chebyshev coefficients for each slit width
        coefficients = {1: (3.9029145352224286, 1.0409009914898588),
                        2: (3.700922980002846, 1.0316558532817541),
                        3: (3.425366901923241, 1.059424415261008),
                        4: (3.1280237101658357, 1.1019794225196382),
                        6: (2.455604480423782, 1.1844363798411677),
                        8: (1.781002390052844, 1.3316380498243399),
                        }
        try:
            slitwidth = int(ext.focal_plane_mask().replace('pix-slit', ''))
        except AttributeError:  # fpm() is returning None
            raise ValueError(f"Cannot determine slit width for {ext.filename}")
        try:
            c1, c3 = coefficients[slitwidth]
        except KeyError:
            raise ValueError(f"Slit width {slitwidth} for {ext.filename}"
                             "unknown")

        # Domain is (0, 2122) since slit projects on row 1061
        # The 1061 values are so the model returns the new pixel location
        # and not the shift (so we don't need to wrap the model)
        m_tweak = models.Chebyshev1D(degree=3, c0=1061, c1=c1+1061,
                                     c2=0, c3=c3, domain=(0, 2122))
        return m_tweak
