# This parameter file contains the parameters related to the primitives located
# in the primitives_gsaoi.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_photometry, parameters_preprocess, parameters_visualize, parameters_nearIR, parameters_stack

class addReferenceCatalogConfig(parameters_photometry.addReferenceCatalogConfig):
    def setDefaults(self):
        self.radius = 0.033
        self.source = "2mass"

class associateSkyConfig(parameters_preprocess.associateSkyConfig):
    def setDefaults(self):
        self.distance = 1.
        self.time = 900.

# class tileArraysConfig(parameters_visualize.tileArraysConfig):
#     def setDefaults(self):
#         self.tile_all = True

class makeBPMConfig(parameters_nearIR.makeBPMConfig):
    def setDefaults(self):
        del self.dark_lo_thresh, self.flat_hi_thresh
        self.dark_hi_thresh = None  # (use default)
        self.flat_lo_thresh = 0.5

    def validate(self):
        config.Config.validate(self)

class stackSkyFramesConfig(parameters_stack.stackSkyFramesConfig):
    def setDefaults(self):
        self.suffix = "_skyStacked"
        self.operation = "median"
        self.reject_method = "minmax"
        self.nhigh = 1
        self.scale = False  # darks not taken routinely for GSAOI
        self.zero = True

class subtractSkyConfig(parameters_preprocess.subtractSkyConfig):
    def setDefaults(self):
        self.reset_sky = False
        self.scale_sky = False  # darks not taken routinely for GSAOI
        self.offset_sky = False

