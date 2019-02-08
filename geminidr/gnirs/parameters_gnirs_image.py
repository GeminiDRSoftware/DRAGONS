# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_image.py file, in alphabetical order.
from geminidr.core import parameters_photometry, parameters_register, parameters_nearIR, parameters_standardize

class addDQConfig(parameters_standardize.addDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True

class addReferenceCatalogConfig(parameters_photometry.addReferenceCatalogConfig):
    def setDefaults(self):
        self.radius = 0.033
        self.source = "2mass"

class adjustWCSToReferenceConfig(parameters_register.adjustWCSToReferenceConfig):
    def setDefaults(self):
        self.first_pass = 2.
        self.min_sources = 1
        self.rotate = True

class detectSourcesConfig(parameters_photometry.detectSourcesConfig):
    def setDefaults(self):
        self.detect_minarea = 40
        self.detect_thresh = 3.
        self.deblend_mincont = 0.001
        self.back_filtersize = 3

class determineAstrometricSolutionConfig(parameters_register.determineAstrometricSolutionConfig):
    def setDefaults(self):
        self.initial = 15.

class makeBPMConfig(parameters_nearIR.makeBPMConfig):
    def setDefaults(self):
        self.dark_lo_thresh = -20.
        self.dark_hi_thresh = 100.
        self.flat_lo_thresh = 0.1
        self.flat_hi_thresh = 1.25
