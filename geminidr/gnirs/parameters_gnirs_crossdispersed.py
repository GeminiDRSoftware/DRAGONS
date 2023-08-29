# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_crossdispersed.py file, in alphabetical order.
from geminidr.core import parameters_spect
from geminidr.core import parameters_standardize
from geminidr.core.parameters_standardize import addIllumMaskToDQConfig
from gempy.library import config

def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True


class addDQConfig(parameters_standardize.addDQConfig, addIllumMaskToDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    order = config.RangeField("Order of fitting function", int, 3, min=0,
                              optional=True)
    debug_min_lines = config.Field("Minimum number of lines to fit each segment",
                                   (str, int), '50,20',
                                   check=list_of_ints_check)

    def setDefaults(self):
        self.in_vacuo = True
