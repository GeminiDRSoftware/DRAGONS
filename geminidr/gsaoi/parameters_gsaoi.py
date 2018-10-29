# This parameter file contains the parameters related to the primitives located
# in the primitives_gsaoi.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_photometry, parameters_preprocess, parameters_visualize, parameters_nearIR

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
        self.dark_lo_thresh = float('NaN')  # not used
        self.dark_hi_thresh = float('NaN')  # means use default
        self.flat_lo_thresh = 0.5
        self.flat_hi_thresh = float('Inf')  # not used, limits disallow NaN

    def validate(self):
        config.Config.validate(self)

