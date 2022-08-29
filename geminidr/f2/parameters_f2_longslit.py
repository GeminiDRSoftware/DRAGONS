# This parameter file contains the parameters related to the primitives located
# in the primitives_f2_longslit.py file, in alphabetical order.
from geminidr.core import parameters_standardize
from geminidr.core.parameters_standardize import addIllumMaskToDQConfig

class addDQConfig(parameters_standardize.addDQConfig, addIllumMaskToDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True
