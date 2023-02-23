#
#                                                                 gemini_python
#
#                                                      primitives_niri_spect.py
# -----------------------------------------------------------------------------

import os

from importlib import import_module

from geminidr.core import Spect
from gempy.gemini import gemini_tools as gt
from gempy.library import transform, wavecal
from recipe_system.utils.decorators import parameter_override, capture_provenance

from .primitives_niri import NIRI
from . import parameters_niri_spect


@parameter_override
@capture_provenance
class NIRISpect(Spect, NIRI):
    """
    This is the class containing all of the preprocessing primitives for the
    NIRISpect level of the hierarchy tree. It inherits all the primitives from
    the level above.
    """
    tagset = {"GEMINI", "NIRI", "SPECT"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_niri_spect)

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
            transform.add_longslit_wcs(ad, central_wavelength=cenwave)

            # Timestamp. Suffix was updated in the super() call
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
        return adinputs


    # MS: following two functions are largely a copy from primitives_gnirs_spect.py
    # for doing a basic wavecal.
    # Expect Olesja will improve it when she gets to NIRI.
    def determineWavelengthSolution(self, adinputs=None, **params):
        for ad in adinputs:
            if params["central_wavelength"] is None:
                    params["central_wavelength"] = ad.central_wavelength(asNanometers=True)

        adinputs = super().determineWavelengthSolution(adinputs, **params)
        return adinputs


    def _get_arc_linelist(self, waves=None, ad=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)

        if 'ARC' in ad.tags:
            if 'Xe' in ad.object():
                linelist ='Ar_Xe.dat'
            elif "Ar" in ad.object():
                linelist = 'argon.dat'
            else:
                raise ValueError(f"No default line list found for {ad.object()}-type arc. Please provide a line list.")
        else:
            if ad.filter_name(pretty=True).startswith('L'):
                linelist = 'skyLband.dat'
            elif ad.filter_name(pretty=True).startswith('L'):
                linelist = 'skyMband.dat'
            else:
                linelist = 'nearIRsky.dat'

        self.log.debug(f"Using linelist '{linelist}'")
        filename = os.path.join(lookup_dir, linelist)

        return wavecal.LineList(filename)

    def _get_cenwave_accuracy(self, ad=None):
        # Accuracy of central wavelength (nm) for a given instrument/setup.
        return 10
