# This parameter file contains the parameters related to the primitives located
# in the primitives_ccd.py file, in alphabetical order.
from gempy.library import config
from .parameters_register import matchWCSToReferenceConfig
from .parameters_resample import resampleToCommonFrameConfig

class core_stacking_config(config.Config):
    """Parameters relevant to ALL stacking primitives"""
    suffix = config.Field("Filename suffix", str, "_stack")
    apply_dq = config.Field("Use DQ to mask bad pixels?", bool, True)
    separate_ext = config.Field("Handle extensions separately?", bool, True)
    statsec = config.Field("Section for statistics", str, None, optional=True)
    operation = config.Field("Averaging operation", str, "mean")
    reject_method = config.Field("Pixel rejection method", str, "varclip")
    hsigma = config.RangeField("High rejection threshold (sigma)", float, 3., min=0)
    lsigma = config.RangeField("Low rejection threshold (sigma)", float, 3., min=0)
    mclip = config.Field("Use median for sigma-clipping?", bool, True)
    nlow = config.RangeField("Number of low pixels to reject", int, 0, min=0)
    nhigh = config.RangeField("Number of high pixels to reject", int, 0, min=0)

class stackFramesConfig(core_stacking_config):
    scale = config.Field("Scale images to the same intensity?", bool, False)
    zero = config.Field("Apply additive offsets to images to match intensity?", bool, True)

class stackSkyFramesConfig(stackFramesConfig):
    mask_objects = config.Field("Mask objects before stacking?", bool, True)
    dilation = config.RangeField("Object dilation radius (pixels)", float, 2., min=0)
    def setDefaults(self):
        self.suffix = "_skyStacked"
        self.operation = "median"
        self.reject_method = "minmax"
        self.nhigh = 1
        self.scale = True
        self.zero = False

class alignAndStackConfig(stackFramesConfig, resampleToCommonFrameConfig,
                          matchWCSToReferenceConfig):
    # TODO: Think about whether we need all these
    pass

class stackFlatsConfig(core_stacking_config):
    scale = config.Field("Scale images to the same intensity?", bool, False)
    def setDefaults(self):
        self.reject_method = "minmax"
        self.nlow = 1
        self.nhigh = 1
        self.scale = True
        self.separate_ext = False

# TODO: Do we want stackSkyFlats with object removal?