# This parameter file contains the parameters related to the primitives located
# in the primitives_crossdispersed.py file, in alphabetical order.

from geminidr.core import parameters_generic
from gempy.library import config
from astrodata import AstroData

from geminidr.core import parameters_spect


class flatCorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_flatCorrected", optional=True)
    flat = config.ListField("Flatfield frame", (str, AstroData), None, optional=True, single=True)


class resampleToCommonFrameConfig(parameters_spect.resampleToCommonFrameConfig):
    """
    For cross-dispersed spectra, the `force_linear` parameter is problematic if
    set to `False`. A linear scale for the entire wavelength range of the
    observation can be created from any individual order, but a non-linear scale
    created from an arbitrary order may or may not be able to cover the entire
    wavelength range. (And a good inverse may not be able to be created, even if
    it does.) For these reasons, for cross-dispersed use we remove the
    `force_linear` parameter entirely (it's set to `True` if not found).
    """
    def setDefaults(self):
        del self.force_linear
