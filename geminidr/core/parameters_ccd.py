# This parameter file contains the parameters related to the primitives located
# in the primitives_ccd.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData

class biasCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_biasCorrected")
    bias = config.Field("Name of bias", (AstroData, str), None, optional=True)

class overscanCorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_overscanCorrected")
    # Inherits everything else from subtractOverscanConfig()

class subtractOverscanConfig(overscanCorrectConfig):
    suffix = config.Field("Filename suffix", str, "_overscanSubtracted")
    niterate = config.RangeField("Maximum number of interations", int, 2, min=1)
    high_reject = config.RangeField("High rejection limit (standard deviations)",
                               float, 3., min=0., optional=True)
    low_reject = config.RangeField("Low rejection limit (standard deviations)",
                              float, 3., min=0., optional=True)
    function = config.ChoiceField("Type of function", str,
                                  allowed = {"spline": "Cublic spline",
                                             "poly":   "Polynomial",
                                             "none": "Row-by-row"},
                                  default="spline", optional=True)
    nbiascontam = config.RangeField("Number of columns to exclude from averaging",
                               int, None, min=0, optional=True)
    order = config.RangeField("Order of fitting function", int, None, min=0,
                              optional=True)

class trimOverscanConfig(overscanCorrectConfig):
    suffix = config.Field("Filename suffix", str, "_overscanTrimmed")
