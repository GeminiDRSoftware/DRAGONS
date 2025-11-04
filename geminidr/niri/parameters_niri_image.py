# This parameter file contains the parameters related to the primitives located
# in the primitives_niri_image.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_photometry

def check_pattern_size(value):
    """Confirm that the pattern size is a factor of 256"""
    return (256 % value == 0)

class detectSourcesConfig(parameters_photometry.detectSourcesConfig):
    def setDefaults(self):
        self.set_saturation = False
        self.detect_minarea = 40
        self.detect_thresh = 1.5
        self.analysis_thresh = 1.5
        self.back_filtersize = 3
