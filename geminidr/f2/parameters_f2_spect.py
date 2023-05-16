# This parameter file contains the parameters related to the primitives located
# in the primitives_f2_spect.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_spect
from geminidr.core import parameters_preprocess


def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    def setDefaults(self):
        self.order = 3
        self.in_vacuo = True
        self.min_snr = 10
        self.debug_min_lines = 100000

class determineDistortionConfig(parameters_spect.determineDistortionConfig):
    def setDefaults(self):
        self.spectral_order = 3
        self.min_snr = 7.
        self.min_line_length = 0.3
        self.max_missed = 2
        self.debug_reject_bad = False

class skyCorrectConfig(parameters_preprocess.skyCorrectConfig):
    def setDefaults(self):
        self.scale_sky = False #MS: IF for whatever reason the exposure times are different between frames being subtracted, one should have a check to turn this on.  
        self.offset_sky = False
        self.mask_objects = False
        self.dilation = 0.

class flatCorrectConfig(parameters_preprocess.flatCorrectConfig):
    def setDefaults(self):
        self.rectify = True
