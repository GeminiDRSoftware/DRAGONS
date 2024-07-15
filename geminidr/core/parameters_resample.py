# This parameter file contains the parameters related to the primitives located
# in the primitives_resample.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData

class shiftImagesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_shift", optional=True)
    shifts = config.Field("Shift(s) to apply (or filename)", str, None)
    interpolant = config.ChoiceField("Type of interpolant", str,
                                     allowed={"nearest": "Nearest neighbour",
                                              "linear": "Linear interpolation",
                                              "poly3": "Cubic polynomial interpolation",
                                              "poly5": "Quintic polynomial interpolation",
                                              "spline3": "Cubic spline interpolation",
                                              "spline5": "Quintic spline interpolation"},
                                     default="poly3", optional=False)
    trim_data = config.Field("Trim to field of view of reference image?", bool, False)
    clean_data = config.Field("Clean bad pixels before interpolation?", bool, False)
    dq_threshold = config.RangeField("Fraction from DQ-flagged pixel to count as 'bad'",
                                     float, 0.001, min=0.)
