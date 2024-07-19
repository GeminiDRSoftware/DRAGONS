#
#                                                                  gemini_python
#
#                                                        primtives_gnirs_spect.py
# ------------------------------------------------------------------------------
import os

from importlib import import_module

from geminidr.core import Spect, Telluric

from .primitives_gnirs import GNIRS
from . import parameters_gnirs_spect

from gempy.gemini import gemini_tools as gt
from gempy.library import transform, wavecal

from recipe_system.utils.decorators import parameter_override, capture_provenance


@parameter_override
@capture_provenance
class GNIRSSpect(Telluric, GNIRS):
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

            # Exception needed to produce GNIRS BPMs from spec flats and darks.
            if "DARK" in ad.tags:
                log.debug(f"Skipping {ad.filename} as it is a DARK")
                continue

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

    def _apply_wavelength_model_bounds(self, model=None, ext=None):
        # Apply bounds to an astropy.modeling.models.Chebyshev1D to indicate
        # the range of parameter space to explore
        # GNIRS has a different central wavelength uncertainty
        super()._apply_wavelength_model_bounds(model, ext)
        if 'ARC' in ext.tags or not (ext.filter_name(pretty=True)[0] in 'LM'):
            prange = 10
        else:
            dispaxis = 2 - ext.dispersion_axis()
            npix = ext.shape[dispaxis]
            prange = abs(ext.dispersion(asNanometers=True)) * npix * 0.07
        model.c0.bounds = (model.c0 - prange, model.c0 + prange)
        dx = 0.02 * abs(model.c1.value)
        model.c1.bounds = (model.c1 - dx, model.c1 + dx)
