# This parameter file contains the parameters related to the primitives located
# in the primitives_f2.py file, in alphabetical order.
from gempy.library import config
from astrodata import AstroData
from geminidr.core import parameters_photometry, parameters_stack, parameters_nearIR

class addDQConfig(parameters_nearIR.addDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True

class detectSourcesConfig(parameters_photometry.detectSourcesConfig):
    def setDefaults(self):
        self.mask = True

#class makeLampFlatConfig(parameters_nearIR.makeLampFlatConfig):
#    dark = config.Field("Name of dark frame (for K-band flats)", (str, AstroData), None, optional=True)

class makeBPMConfig(parameters_nearIR.makeBPMConfig):
    def setDefaults(self):
        self.dark_lo_thresh = -150.
        self.dark_hi_thresh = 3050.
        self.flat_lo_thresh = 0.68
        self.flat_hi_thresh = 1.28
