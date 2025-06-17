#
#                                                                  gemini_python
#
#                                                        primtives_gnirs_spect.py
# ------------------------------------------------------------------------------
import os
import numpy as np

from importlib import import_module

from geminidr.core import Telluric

from .primitives_gnirs import GNIRS
from . import parameters_gnirs_spect

from gempy.gemini import gemini_tools as gt
from gempy.library import wavecal

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

    def normalizeFlat(self, adinputs=None, **params):
        """
        A GNIRS-specific primitive to normalize the spectroscopic flatfield.
        Because of the odd/even row effect, the trace of the brightness has
        two loci and can be difficult to fit, sometimes jumping from one locus
        to the other. This primitive will attempt to remove the odd/even
        offset temporarily while the fit takes place, by scaling the odd rows
        by a constance value to match the even rows.

        Parameters
        ----------
        suffix : str/None
            suffix to be added to output files
        center : int/None
            central row/column for 1D extraction (None => use middle)
        nsum : int
            number of rows/columns to average (about "center")
        function : str
            type of function to fit (splineN or polynomial types)
        order : int
            Order of the spline fit to be performed
        lsigma : float/None
            lower rejection limit in standard deviations
        hsigma : float/None
            upper rejection limit in standard deviations
        niter : int
            maximum number of rejection iterations
        grow : float/False
            growth radius for rejected pixels
        interactive : bool
            set to activate an interactive preview to fine tune the input parameters
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            # XD will have multiple extensions at this point
            all_scaling_data = []
            for ext in ad:
                row_values = np.ma.median(np.ma.masked_array(
                    ext.data, mask=ext.mask), axis=1)
                bright_enough = row_values[1::2] > 50 * ext.read_noise()
                if bright_enough.any():
                    scaling = np.ma.median((row_values[::2] /
                                            row_values[1::2])[bright_enough])
                else:
                    scaling = 1.0
                log.debug(f"Scaling for {ad.filename}:{ext.id}: {scaling:8.6f}")
                scaling_data = np.ones_like(ext.data)
                scaling_data[1::2] *= scaling
                ext.multiply(scaling_data)
                all_scaling_data.append(scaling_data)

            # normalizeFlat() operates in-place; we don't need a return value
            super().normalizeFlat([ad], **params)
            for ext, scaling_data in zip(ad, all_scaling_data):
                ext.divide(scaling_data)

        return adinputs

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
        super().standardizeWCS(adinputs, **params)
        for ad in adinputs:
            self._add_longslit_wcs(ad, pointing="center")
        return adinputs

    def _get_linelist(self, wave_model=None, ext=None, config=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)

        is_lowres = (ext.disperser(pretty=True).startswith('10') or
                     ext.disperser(pretty=True).startswith('32') and
                     ext.camera(pretty=True).startswith('Short'))

        if 'ARC' in ext.tags:
            if 'Xe' in ext.object():
                filename ='Ar_Xe.dat'
            elif "Ar" in ext.object():
                filename = 'lowresargon.dat' if is_lowres else 'argon.dat'
            else:
                raise ValueError(f"No default line list found for {ext.object()}"
                                 "-type arc. Please provide a line list.")
        elif config.get("absorption", False) or wave_model.c0 > 2800:
            return self._get_atran_linelist(wave_model=wave_model, ext=ext, config=config)
        else:
            # In case of wavecal from sky OH emission use this line list
            filename = 'nearIRsky.dat'

        self.log.stdinfo(f"Using linelist {filename}")
        linelist = wavecal.LineList(os.path.join(lookup_dir, filename))
        if 'ARC' not in ext.tags:
            # Attach a synthetic sky spectrum if using sky lines or absorption
            linelist.reference_spectrum = self._get_sky_spectrum(wave_model, ext)

        return linelist


    def _wavelength_model_bounds(self, model=None, ext=None):
        # Apply bounds to an astropy.modeling.models.Chebyshev1D to indicate
        # the range of parameter space to explore
        # GNIRS has a different central wavelength uncertainty
        bounds = super()._wavelength_model_bounds(model, ext)
        if 'ARC' in ext.tags or not (ext.filter_name(pretty=True)[0] in 'LM'):
            prange = 10
        else:
            dispaxis = 2 - ext.dispersion_axis()
            npix = ext.shape[dispaxis]
            prange = abs(ext.dispersion(asNanometers=True)) * npix * 0.07
        # Recovering the values from the bounds protects against a Shift/Scale model
        cenwave = np.mean(bounds['c0'])
        bounds['c0'] = (cenwave - prange, cenwave + prange)
        c1 = np.mean(bounds['c1'])
        dx = 0.02 * abs(c1)
        bounds['c1'] = (c1 - dx, c1 + dx)
        return bounds