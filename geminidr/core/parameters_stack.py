# This parameter file contains the parameters related to the primitives located
# in the primitives_ccd.py file, in alphabetical order.
import re
from gempy.library import config
from .parameters_register import matchWCSToReferenceConfig
from .parameters_resample import resampleToCommonFrameConfig

def statsec_check(value):
    """Confirm that the statsec Field is given a value consisting of 4
       integers, x1:x2,y1:y2 with optional []"""
    m = re.match('\[?(\d+):(\d+),(\d+):(\d+)\]?', value)
    if m is None:
        return False
    try:
        coords = [int(v) for v in m.groups()]
    except ValueError:
        return False
    try:
        assert len(coords) == 4
        assert coords[0] < coords[1]
        assert coords[2] < coords[3]
    except AssertionError:
        return False
    return True

class core_stacking_config(config.Config):
    """Parameters relevant to ALL stacking primitives"""
    suffix = config.Field("Filename suffix", str, "_stack", optional=True)
    apply_dq = config.Field("Use DQ to mask bad pixels?", bool, True)
    separate_ext = config.Field("Handle extensions separately?", bool, True)
    statsec = config.Field("Section for statistics", str, None, optional=True, check=statsec_check)
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
                                       default="varclip", optional=False)
    hsigma = config.RangeField("High rejection threshold (sigma)", float, 3., min=0)
    lsigma = config.RangeField("Low rejection threshold (sigma)", float, 3., min=0)
    mclip = config.Field("Use median for sigma-clipping?", bool, True)
    nlow = config.RangeField("Number of low pixels to reject", int, 0, min=0)
    nhigh = config.RangeField("Number of high pixels to reject", int, 0, min=0)

class stackFramesConfig(core_stacking_config):
    scale = config.Field("Scale images to the same intensity?", bool, False)
    zero = config.Field("Apply additive offsets to images to match intensity?", bool, False)

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
    save = config.Field("Save aligned images to disk?", bool, False)

class stackFlatsConfig(core_stacking_config):
    scale = config.Field("Scale images to the same intensity?", bool, False)
    def setDefaults(self):
        self.reject_method = "minmax"
        self.nlow = 1
        self.nhigh = 1
        self.separate_ext = False

# TODO: Do we want stackSkyFlats with object removal?