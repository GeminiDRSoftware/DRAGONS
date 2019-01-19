# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.
from gempy.library import config

class resampleToCommonFrameConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_align", optional=True)
    order = config.RangeField("Order of interpolation", int, 3, min=0, max=5)
    trim_data = config.Field("Trim to field of view of reference image?", bool, False)
    clean_data = config.Field("Clean bad pixels before interpolation?", bool, False)
