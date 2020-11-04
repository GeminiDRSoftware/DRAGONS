# This parameter file contains the parameters related to the primitives located
# in the primitives_gsaoi_image.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_photometry, parameters_resample


class detectSourcesConfig(parameters_photometry.detectSourcesConfig):
    def setDefaults(self):
        self.detect_minarea = 20
        self.detect_thresh = 1.
        self.analysis_thresh = 1.
        self.phot_min_radius = 1.
        self.back_size = 256
        self.back_filtersize = 5

class resampleToCommonFrameConfig(parameters_resample.resampleToCommonFrameConfig):
    pixel_scale = config.RangeField("Output pixel scale (arcseconds) if no reference provided",
                                    float, 0.02, min=0.01, max=1.0)
    pa = config.RangeField("Output position angle (E of N) if no reference provided",
                           float, 0.0, min=-180, max=360)

    def setDefaults(self):
        self.force_affine = False
