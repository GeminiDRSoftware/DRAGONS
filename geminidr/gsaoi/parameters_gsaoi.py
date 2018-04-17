# This parameter file contains the parameters related to the primitives located
# in the primitives_gsaoi.py file, in alphabetical order.
from geminidr.core import parameters_photometry, parameters_preprocess, parameters_visualize

class addReferenceCatalogConfig(parameters_photometry.addReferenceCatalogConfig):
    def setDefaults(self):
        self.radius = 0.033
        self.source = "2mass"

class associateSkyConfig(parameters_preprocess.associateSkyConfig):
    def setDefaults(self):
        self.distance = 1.
        self.time = 900.

class tileArraysConfig(parameters_visualize.tileArraysConfig):
    def setDefaults(self):
        del self.tile_all
