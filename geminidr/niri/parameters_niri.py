# This parameter file contains the parameters related to the primitives located
# in the primitives_niri.py file, in alphabetical order.
from geminidr.core import parameters_preprocess, parameters_photometry, parameters_nearIR

class addDQConfig(parameters_nearIR.addDQConfig):
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

class makeBPMConfig(parameters_nearIR.makeBPMConfig):
    def setDefaults(self):
        self.dark_lo_thresh = -20.
        self.dark_hi_thresh = 1300.
        self.flat_lo_thresh = 0.8
        self.flat_hi_thresh = 1.25
