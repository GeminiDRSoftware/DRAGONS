# This parameter file contains the parameters related to the primitives located
# in the primitives_ccd.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData
from . import parameters_generic

class biasCorrectConfig(parameters_generic.calRequirementConfig):
    suffix = config.Field("Filename suffix", str, "_biasCorrected", optional=True)
    bias = config.ListField("Bias(es) to subtract", (AstroData, str), None,
                            optional=True, single=True)


class subtractOverscanConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_overscanSubtracted", optional=True)
    function = config.ChoiceField("Fitting function", str,
                           allowed={"none": "Row-by-row values",
                                    "spline3": "Cubic spline",
                                    "chebyshev": "Chebyshev polynomial"},
                           default="spline3", optional=False)
    order = config.RangeField("Order of fitting function", int, None, min=0,
                              optional=True)
    nbiascontam = config.RangeField("Number of columns to exclude from averaging",
                                    int, 0, min=0)
    bias_type = config.ChoiceField("Overscan type", str,
                                   allowed={"serial":"Serial", "parallel":"Parallel"},
                                   optional=True)

    def validate(self):
        config.Config.validate(self)
        if self.function == "poly" and self.order is None:
            raise ValueError("Polynomial order must be specified")
        if self.function == "spline" and self.order == 0:
            raise ValueError("Must specify a positive spline order, or None")


class trimOverscanConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_overscanTrimmed", optional=True)


class overscanCorrectConfig(subtractOverscanConfig, trimOverscanConfig):
    def setDefaults(self):
        self.suffix = "_overscanCorrected"
