# This parameter file contains the parameters related to the primitives located
# in the primitives_f2_spect.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_spect

def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True

class determineDistortionConfig(parameters_spect.determineDistortionConfig):
    id_only = config.Field("Use only lines identified for wavelength calibration?", bool, True)

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    nbright = config.RangeField("Number of bright lines to eliminate", int, 0, min=0)
    in_vacuo = config.Field("Use vacuum wavelength scale (rather than air)?", bool, True)
    debug_min_lines = config.Field("Minimum number of lines to fit each segment", (str, int), 100000,
                                   check=list_of_ints_check)
    def setDefaults(self):
        self.order = 3
