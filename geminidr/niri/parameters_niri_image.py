# This parameter file contains the parameters related to the primitives located
# in the primitives_niri_image.py file, in alphabetical order.
from geminidr.core import parameters_photometry

class detectSourcesConfig(parameters_photometry.detectSourcesConfig):
    def setDefaults(self):
        self.set_saturation = False
        self.detect_minarea = 40
        self.detect_thresh = 1.5
        self.analysis_thresh = 1.5
        self.back_filtersize = 3
