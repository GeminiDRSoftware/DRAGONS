# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_spect.py file, in alphabetical order.
from astrodata import AstroData
from gempy.library import config
from geminidr.core import parameters_spect
from geminidr.core import parameters_preprocess


def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    order = config.RangeField("Order of fitting function", int, None, min=0,
                              optional=True)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, None, min=1., optional=True)
    debug_min_lines = config.Field("Minimum number of lines to fit each segment", (str, int), None,
                                   check=list_of_ints_check, optional=True)
    def setDefaults(self):
        self.in_vacuo = True

class skyCorrectConfig(parameters_preprocess.skyCorrectConfig):
    def setDefaults(self):
        self.scale_sky = False #MS: IF for whatever reason the exposure times are different between frames being subtracted, that case may require a special treatment
        self.offset_sky = False
        self.mask_objects = False
        self.dilation = 0.
