#
#                                                                 gemini_python
#
#                                                      primitives_niri_spect.py
# -----------------------------------------------------------------------------


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
