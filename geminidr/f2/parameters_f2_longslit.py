# This parameter file contains the parameters related to the primitives located
# in the primitives_f2_longslit.py file, in alphabetical order.
from geminidr.core import parameters_standardize
from gempy.library import config


class addDQConfig(parameters_standardize.addDQConfig):
    keep_second_order = config.Field("Don't apply second order light mask?", bool, False)
    def setDefaults(self):
        self.add_illum_mask = True

class addIllumMaskToDQConfig(parameters_standardize.addIllumMaskToDQConfig):
    keep_second_order = config.Field("Don't apply second order light mask?", bool, False)
