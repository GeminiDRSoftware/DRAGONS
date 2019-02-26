# This parameter file contains the parameters related to the primitives located
# in the primitives_gsaoi.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_photometry, parameters_preprocess, parameters_visualize, parameters_nearIR, parameters_stack

class addDQConfig(parameters_nearIR.addDQConfig):
    def setDefaults(self):
        self.latency = True
        self.non_linear = True
        self.time = 900.

class addReferenceCatalogConfig(parameters_photometry.addReferenceCatalogConfig):
    def setDefaults(self):
        self.radius = 0.033
        self.source = "2mass"

class associateSkyConfig(parameters_preprocess.associateSkyConfig):
    def setDefaults(self):
        self.distance = 1.
        self.time = 900.

class makeBPMConfig(parameters_nearIR.makeBPMConfig):
    override_thresh = config.Field("Override GSAOI default calculation with a user-specified dark_hi_thresh?", bool, False)

    def setDefaults(self):
        del self.dark_lo_thresh, self.flat_hi_thresh
        self.dark_hi_thresh = None  # (when using default calc. instead)
        self.flat_lo_thresh = 0.5

    def validate(self):
        config.Config.validate(self)

class stackSkyFramesConfig(parameters_stack.stackSkyFramesConfig):
    def setDefaults(self):
        self.dilation = 6.
        self.suffix = "_skyStacked"
        self.operation = "median"
        self.reject_method = "minmax"
        self.nlow = 1
        self.nhigh = 1
        self.scale = False  # darks not taken routinely for GSAOI
        self.zero = True

class subtractSkyConfig(parameters_preprocess.subtractSkyConfig):
    def setDefaults(self):
        self.reset_sky = False
        self.scale_sky = False  # darks not taken routinely for GSAOI
        self.offset_sky = True

