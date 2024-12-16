# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_spect.py file, in alphabetical order.
from astrodata import AstroData
from gempy.library import config
from geminidr.core import parameters_preprocess


class addMDFConfig(config.Config):
    # Does not use MDF files
    suffix = config.Field("Filename suffix", str, "_mdfAdded", optional=True)


class skyCorrectConfig(parameters_preprocess.skyCorrectConfig):
    def setDefaults(self):
        # self.scale_sky = False #MS: IF for whatever reason the exposure times are different between frames being subtracted, that case may require a special treatment
        # self.offset_sky = False
        del self.scale_sky
        del self.offset_sky
        self.mask_objects = False
        self.dilation = 0.
