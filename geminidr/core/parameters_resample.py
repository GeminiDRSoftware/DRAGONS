# This parameter file contains the parameters related to the primitives located
# in the primitives_resample.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData

class resampleToCommonFrameConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_align", optional=True)
    order = config.RangeField("Order of interpolation", int, 1, min=0, max=5, inclusiveMax=True)
    trim_data = config.Field("Trim to field of view of reference image?", bool, False)
    clean_data = config.Field("Clean bad pixels before interpolation?", bool, False)
    reference = config.Field("Name of reference image (optional)", (str, AstroData), None, optional=True)

class shiftImagesConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_shift", optional=True)
    shifts = config.Field("Shift(s) to apply (or filename)", str, None)
    order = config.RangeField("Order of interpolation", int, 1, min=0, max=5, inclusiveMax=True)
    trim_data = config.Field("Trim to field of view of reference image?", bool, False)
    clean_data = config.Field("Clean bad pixels before interpolation?", bool, False)

class applyStackedObjectMaskConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_stackedObjMaskApplied", optional=True)
    source = config.Field("Filename of/Stream containing stacked image", str, None)
    order = config.RangeField("Order of interpolation", int, 1, min=0, max=5, inclusiveMax=True)
    threshold = config.RangeField("Threshold for flagging pixels", float, 0.01, min=0., max=1.)
