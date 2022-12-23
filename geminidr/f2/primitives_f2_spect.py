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
from gemini_instruments.f2.lookup import dispersion_offset_mask

from gempy.gemini import gemini_tools as gt
from gempy.library import transform, wavecal

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
            cenwave = (ad.central_wavelength(asNanometers=True) +
                       abs(ad.dispersion(asNanometers=True)[0]) * self._get_cenwave_offset(ad))
            transform.add_longslit_wcs(ad, central_wavelength=cenwave,
                                       pointing=ad[0].wcs(1024, 1024))

            # Timestamp. Suffix was updated in the super() call
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
        return adinputs

    def _get_arc_linelist(self, waves=None, ad=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)

        if 'ARC' in ad.tags:
            linelist = 'argon.dat'
            if ad.disperser(pretty=True) == "HK" and \
                    ad.filter_name(pretty=True) == "JH":
                linelist = 'lowresargon_with_2nd_ord.dat'
        else:
            linelist = 'sky.dat'

        filename = os.path.join(lookup_dir, linelist)
        return wavecal.LineList(filename)

    def _get_cenwave_offset(self, ad=None):
        filter = ad.filter_name(pretty=True)
        if filter in {"HK", "JH"}:
            filter = ad.filter_name(keepID=True)
        # The following is needed since after the new HK and JH filters were installed, their WAVELENG
        # keywords wasn't updated until after the specified date, so the cenwave_offset for the
        # old filters has to be used.
        if ad.phu['DATE'] < '9999-99-99':
            if filter == "JH_G0816":
                filter = "JH_G0809"
            if filter == "HK_G0817":
                filter = "HK_G0806"
        index = (ad.disperser(pretty=True), filter)
        mask = dispersion_offset_mask.get(index, None)
        return mask.cenwaveoffset if mask else None

    def _get_cenwave_accuracy(self, ad=None):
        # Accuracy of central wavelength (nm) for a given setup.
        return 10
