# This parameter file contains the parameters related to the primitives located
# in the primitives_niri.py file, in alphabetical order.
from geminidr.core import parameters_preprocess, parameters_photometry, parameters_standardize

class addDQConfig(parameters_standardize.addDQConfig):
    def setDefaults(self):
        self.latency = True

class addReferenceCatalogConfig(parameters_photometry.addReferenceCatalogConfig):
    def setDefaults(self):
        self.radius = 0.033
        self.source = "2mass"

class associateSkyConfig(parameters_preprocess.associateSkyConfig):
    def setDefaults(self):
        self.distance = 1.
        self.time = 900.

class detectSourcesConfig(parameters_photometry.detectSourcesConfig):
    def setDefaults(self):
        self.set_saturation = True
