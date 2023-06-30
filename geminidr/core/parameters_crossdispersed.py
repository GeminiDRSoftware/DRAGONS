# This parameter file contains the parameters related to the primitives located
# in the primitives_crossdispersed.py file, in alphabetical order.

from geminidr.core import parameters_generic
from gempy.library import config
from astrodata import AstroData


class flatCorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_flatCorrected", optional=True)
    flat = config.ListField("Flatfield frame", (str, AstroData), None, optional=True, single=True)
