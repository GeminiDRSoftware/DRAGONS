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


    def _get_cenwave_accuracy(self, ext):
        # Accuracy of central wavelength (nm) for a given instrument/setup.
        # According to GNIRS instrument pages "wavelength settings are accurate
        # to better than 5 percent of the wavelength coverage".
        # However using 7% covers more cases. For the arcs dc0=10 works just fine for all modes.

        mband = ext.filter_name(pretty=True).startswith('M')
        lband = ext.filter_name(pretty=True).startswith('L')
        dispaxis = 2 - ext.dispersion_axis()  # python sense
        npix = ext.shape[dispaxis]

        if 'ARC' in ext.tags or not (mband or lband):
            dcenwave = 10
        else:
            dcenwave = abs(ext.dispersion(asNanometers=True)) * npix * 0.07
        return dcenwave
