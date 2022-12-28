# This parameter file contains the parameters related to the primitive located
# in the primitives_niri_spect.py file, in alphabetical order.

from astrodata import AstroData
from gempy.library import config
from geminidr.core import parameters_spect
from geminidr.core import parameters_preprocess


def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True

class determineDistortionConfig(parameters_spect.determineDistortionConfig):
    def setDefaults(self):
        # params not checked:
        self.id_only = True
        self.spectral_order = 3
        self.min_snr = 10
        self.debug_reject_bad = False

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    def setDefaults(self):
        self.order = 3
        self.in_vacuo = True
        self.debug_min_lines = 100000

class skyCorrectConfig(parameters_preprocess.skyCorrectConfig):
    def setDefaults(self):
        self.scale_sky = False #MS: IF for whatever reason the exposure times are different between frames being subtracted, that case may require a special treatment
        self.offset_sky = False
        self.mask_objects = False
        self.dilation = 0.
