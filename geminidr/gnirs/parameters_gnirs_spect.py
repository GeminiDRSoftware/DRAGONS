# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_spect.py file, in alphabetical order.
from astrodata import AstroData
from gempy.library import config
from geminidr.core import parameters_spect

def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True

class determineDistortionConfig(parameters_spect.determineDistortionConfig):
    def setDefaults(self):
        self.id_only = True

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    def setDefaults(self):
        self.order = None
        self.in_vacuo = True
        self.nbright = 0
        self.debug_min_lines = 100000
