#
#                                                                  gemini_python
#
#                                                          primtives_f2_spect.py
# ------------------------------------------------------------------------------

import os

from importlib import import_module

from geminidr.core import Spect
from .primitives_f2 import F2
from . import parameters_f2_spect
from gemini_instruments.f2.lookup import dispersion_offset_mask, resolving_power

from gempy.gemini import gemini_tools as gt
from gempy.library import transform, wavecal
from gemini_instruments import gmu

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class F2Spect(Spect, F2):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Spect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "F2", "SPECT"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2_spect)

    def makeLampFlat(self, adinputs=None, **params):
        """
        This produces an appropriate stacked F2 spectroscopic flat, based on
        the inputs. For F2 spectroscopy, lamp-on flats have the dark current
        removed by subtracting darks.

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

        # This is basically the generic makeLampFlat code, but altered to
        # distinguish between FLATs and DARKs, not LAMPONs and LAMPOFFs
        flat_list = self.selectFromInputs(adinputs, tags='FLAT')
        dark_list = self.selectFromInputs(adinputs, tags='DARK')
        stack_params = self._inherit_params(params, "stackFrames")
        if dark_list:
            self.showInputs(dark_list, purpose='darks')
            dark_list = self.stackDarks(dark_list, **stack_params)
        self.showInputs(flat_list, purpose='flats')
        stack_params.update({'zero': False, 'scale': False})
        flat_list = self.stackFrames(flat_list, **stack_params)

        if flat_list and dark_list:
            log.fullinfo("Subtracting stacked dark from stacked flat")
            flat = flat_list[0]
            flat.subtract(dark_list[0])
            flat.update_filename(suffix=suffix, strip=True)
            return [flat]

        elif flat_list:  # No darks were passed.
            # Look for dark in calibration manager; if not found, crash.
            log.fullinfo("Only had flats to stack. Calling darkCorrect.")
            flat_list = self.darkCorrect(flat_list, suffix=suffix,
                                         dark=None, do_cal='procmode')
            if flat_list[0].phu.get('DARKIM') is None:
                # No dark was subtracted by darkCorrect:
                raise RuntimeError("No processed dark found in calibration "
                                   "database. Please either provide one, or "
                                   "include a list of darks as input.")
            return flat_list

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

        log = self.log
        timestamp_key = self.timestamp_keys[self.myself()]
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        super().standardizeWCS(adinputs, **params)

        for ad in adinputs:
            # Need to exclude darks from having a spectroscopic WCS added as
            # they don't have a SPECT tag and will gum up the works. This only
            # needs to be done for F2's makeLampFlat as it uses flats minus
            # darks to remove dark current.
            if 'DARK' in ad.tags:
                log.stdinfo(f"{ad.filename} is a DARK, continuing")
                continue

            log.stdinfo(f"Adding spectroscopic WCS to {ad.filename}")
            # Apply central wavelength offset
            if ad.dispersion() is None:
                raise ValueError(f"Unknown dispersion for {ad.filename}")
            cenwave = self._get_actual_cenwave(ad[0], asNanometers=True)
            transform.add_longslit_wcs(ad, central_wavelength=cenwave,
                                       pointing=ad[0].wcs(1024, 1024))

            # Timestamp. Suffix was updated in the super() call
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
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

        combine_method: {"mean", "median", "optimal"}
            Method to use for combining rows/columns when extracting 1D-spectrum.
            Default: "optimal".

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Updated objects with a `.WAVECAL` attribute and improved wcs for
            each slice
        """
        adoutputs = []
        for ad in adinputs:
            these_params = params.copy()
            min_snr_isNone = True if these_params["min_snr"] is None else False

            if "ARC" in ad.tags:
                if these_params["min_snr"] is None:
                    these_params["min_snr"] = 10
            else:
                # Telluric absorption in object spectrum
                if these_params.get("absorption", False):
                    self.generated_linelist = True
                    these_params["lsigma"] = 2
                    these_params["hsigma"] = 2
                    if these_params["min_snr"] is None:
                        these_params["min_snr"] = 1
                else:
                    # OH emission
                    if these_params["min_snr"] is None:
                        these_params["min_snr"] = 10

            if min_snr_isNone:
                self.log.stdinfo(f'Parameter "min_snr" is set to None. '
                                 f'Using min_snr={these_params["min_snr"]} for {ad.filename}')
            
            adoutputs.extend(super().determineWavelengthSolution([ad], **these_params))
        return adoutputs

    def _get_arc_linelist(self, ext, waves=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)
        isHK_JH = ext.disperser(pretty=True) == "HK" and \
                    ext.filter_name(pretty=True) == "JH"
        if 'ARC' in ext.tags:
            linelist = 'argon.dat'
            # For HK grism + JH filter the second order is preserved by default
            # (as per IS request), so use the line list with the second order lines
            # for this mode
            if isHK_JH:
                linelist = 'lowresargon_with_2nd_ord.dat'
        else:
            # In case of wavecal from sky OH emission use this line list:
            linelist = 'nearIRsky.dat'
            if isHK_JH:
                linelist = 'nearIRsky_with_2nd_order.dat'

        self.log.stdinfo(f"Using linelist '{linelist}'")
        filename = os.path.join(lookup_dir, linelist)
        return wavecal.LineList(filename)


    def _get_actual_cenwave(self, ext, asMicrometers=False, asNanometers=False, asAngstroms=False):
        """
        For some instruments (NIRI, F2) wavelenght at the central pixel
        can differ significantly from the descriptor value.

        Parameters
        ----------
        asMicrometers : bool
            If True, return the wavelength in microns
        asNanometers : bool
            If True, return the wavelength in nanometers
        asAngstroms : bool
            If True, return the wavelength in Angstroms

        Returns
        -------
        float
            Actual cenral wavelenght
        """
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        output_units = "meters" # By default
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # central wavelength in these units
            if asMicrometers:
                output_units = "micrometers"
            if asNanometers:
                output_units = "nanometers"
            if asAngstroms:
                output_units = "angstroms"
        index = (ext.disperser(pretty=True), ext.filter_name(keepID=True))
        mask = dispersion_offset_mask.get(index, None)
        cenwave_offset = mask.cenwaveoffset if mask else None
        actual_cenwave = ext.central_wavelength() + \
                  abs(ext.dispersion()) * cenwave_offset
        actual_cenwave = gmu.convert_units('meters', actual_cenwave, output_units)
        return actual_cenwave


    def _get_resolution(self, ext):
        # For F2 grisms resolution peaks in the middle of tthe filter and drops
        # dramatically on both sides. Use "average" resolution from the LUT,
        # (within 70% of filter's range, see F2 web pages).
        fpmask = ext.focal_plane_mask(pretty=True)
        if 'pix-slit' in fpmask:
            slit_width= int(fpmask.replace('pix-slit', ''))
        else:
            slit_width = fpmask
        disperser = ext.disperser(pretty=True)
        return resolving_power.get(f"{slit_width}", {}).get(f"{disperser}", None)

    def _apply_wavelength_model_bounds(self, model=None, ext=None):
        # Apply bounds to an astropy.modeling.models.Chebyshev1D to indicate
        # the range of parameter space to explore
        # The default here is 2% tolerance in central wavelength and dispersion
        super()._apply_wavelength_model_bounds(model, ext)
        model.c0.bounds = (model.c0 - 10, model.c0 + 10)
