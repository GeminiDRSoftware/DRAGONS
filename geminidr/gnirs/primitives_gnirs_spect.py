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

    # TODO: for data prior 2015 update WCS here? -OS
    def standardizeWCS(self, adinputs=None, suffix=None):
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

        for ad in adinputs:
            log.stdinfo(f"Adding spectroscopic WCS to {ad.filename}")
            cenwave = ad.central_wavelength(asNanometers=True)
            transform.add_longslit_wcs(ad, central_wavelength=cenwave,
                                       pointing=ad[0].wcs(512, 511))

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
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
            if ad.filter_name(pretty=True).startswith('L'):
                linelist = 'skyLband.dat'
            elif ad.filter_name(pretty=True).startswith('M'):
                linelist = 'skyMband.dat'
            elif is_lowres:
                linelist = 'sky.dat'
            else:
                linelist = 'nearIRsky.dat'

        self.log.stdinfo(f"Using linelist {linelist}")
        filename = os.path.join(lookup_dir, linelist)

        return wavecal.LineList(filename)
