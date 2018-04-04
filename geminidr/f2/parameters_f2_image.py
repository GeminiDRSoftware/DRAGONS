# This parameter file contains the parameters related to the primitives located
# in the primitives_f2.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_standardize, parameters_photometry

class addDQConfig(parameters_standardize.addDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True

class detectSourcesConfig(parameters_photometry.detectSourcesConfig):
    def setDefaults(self):
        self.mask = True

class makeLampFlatConfig(config.Config):
    pass
