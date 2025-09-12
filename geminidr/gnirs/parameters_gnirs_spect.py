# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_spect.py file, in alphabetical order.
from astrodata import AstroData
from gempy.library import config
from geminidr.core import parameters_preprocess
from . import parameters_gnirs


class addMDFConfig(config.Config):
    # Does not use MDF files
    suffix = config.Field("Filename suffix", str, "_mdfAdded", optional=True)


class associateSkyConfig(parameters_gnirs.associateSkyConfig):
    def setDefaults(self):
        self.min_skies = 2


class determineDistortionConfig(parameters_spect.determineDistortionConfig):
    spectral_order = config.RangeField("Fitting order in spectral direction", int, None, min=1, optional=True)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost",
                               int, None, min=0, optional=True)
    def setDefaults(self):
        self.min_snr = 10
        self.debug_reject_bad = False


class skyCorrectConfig(parameters_preprocess.skyCorrectConfig):
    def setDefaults(self):
        # self.scale_sky = False #MS: IF for whatever reason the exposure times are different between frames being subtracted, that case may require a special treatment
        # self.offset_sky = False
        del self.scale
        del self.zero
        del self.scale_sky
        del self.offset_sky
        self.mask_objects = False
        self.dilation = 0.
