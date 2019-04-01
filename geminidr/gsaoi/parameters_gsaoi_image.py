# This parameter file contains the parameters related to the primitives located
# in the primitives_gsaoi_image.py file, in alphabetical order.
from geminidr.core import parameters_photometry


class detectSourcesConfig(parameters_photometry.detectSourcesConfig):

    def setDefaults(self):

        self.detect_minarea = 20
        self.detect_thresh = 1.
        self.analysis_thresh = 1.
        self.phot_min_radius = 1.
        self.back_size = 256
        self.back_filtersize = 5
