# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_image.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_stack, parameters_photometry, parameters_standardize

class addOIWFSToDQConfig(config.Config):
    pass

class makeFringeFrameConfig(parameters_stack.stackFramesConfig, parameters_photometry.detectSourcesConfig):
    subtract_median_image = config.Field("Subtract median image?", bool, True)
    def setDefaults(self):
        self.suffix = "_fringe"

class scaleFringeToScienceConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_fringeScaled", optional=True)
    science = None # TODO
    stats_scale = config.Field("Scale by statistics rather than exposure time?", bool, False)

class stackFlatsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_stack", optional=True)
    apply_dq = config.Field("Use DQ to mask bad pixels?", bool, True)
    scale = config.Field("Scale images to the same intensity?", bool, False)
    operation = config.ChoiceField("Averaging operation", str,
                                   allowed = {"mean": "arithmetic mean",
                                              "wtmean": "variance-weighted mean",
                                              "median": "median",
                                              "lmedian": "low-median"},
                                   default="mean", optional=False)
    reject_method = config.ChoiceField("Pixel rejection method", str,
                                       allowed={"none": "no rejection",
                                                "minmax": "reject highest and lowest pixels",
                                                "sigclip": "reject pixels based on scatter",
                                                "varclip": "reject pixels based on variance array"},
                                       default="minmax", optional=False)
    hsigma = config.RangeField("High rejection threshold (sigma)", float, 3., min=0)
    lsigma = config.RangeField("Low rejection threshold (sigma)", float, 3., min=0)
    mclip = config.Field("Use median for sigma-clipping?", bool, True)
    max_iters = config.RangeField("Maximum number of clipping iterations", int, None, min=1, optional=True)
    nlow = config.RangeField("Number of low pixels to reject", int, 0, min=0)
    nhigh = config.RangeField("Number of high pixels to reject", int, 0, min=0)
    memory = config.RangeField("Memory available for stacking (GB)", float, None, min=0.1, optional=True)
