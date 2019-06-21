# This parameter file contains the parameters related to the primitives located
# in the primitives_ccd.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData

class biasCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_biasCorrected", optional=True)
    bias = config.ListField("Bias(es) to subtract", (AstroData, str), None,
                            optional=True, single=True)
    do_bias = config.Field("Perform bias subtraction?", bool, True)

class subtractOverscanConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_overscanSubtracted", optional=True)
    niterate = config.RangeField("Maximum number of iterations", int, 2, min=1)
    high_reject = config.RangeField("High rejection limit (standard deviations)",
                               float, 3., min=0., optional=True)
    low_reject = config.RangeField("Low rejection limit (standard deviations)",
                              float, 3., min=0., optional=True)
    function = config.ChoiceField("Type of function", str,
                                  allowed = {"spline": "Cublic spline",
                                             "poly":   "Polynomial",
                                             "none":   "Row-by-row"},
                                  default="spline", optional=True)
    nbiascontam = config.RangeField("Number of columns to exclude from averaging",
                               int, 0, min=0)
    order = config.RangeField("Order of fitting function", int, None, min=0,
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
