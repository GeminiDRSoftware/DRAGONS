# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.
from gempy.library import config

class resampleToCommonFrameConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_oldalign", optional=True)
    interpolator = config.ChoiceField("Type of pixel interpolation", str,
                                      allowed={"nearest": "nearest pixel",
                                               "linear": "linear interpolation",
                                               "spline2": "quadratic spline",
                                               "spline3": "cubic spline",
                                               "spline4": "quartic spline",
                                               "spline5": "qunitic spline"},
                                      default="linear")
    trim_data = config.Field("Trim to field of view of reference image?", bool, False)
