# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_image.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_stack, parameters_photometry, parameters_standardize

class addOIWFSToDQConfig(config.Config):
    pass

class makeFringeFrameConfig(parameters_stack.stackFramesConfig, parameters_photometry.detectSourcesConfig):
    subtract_median_image = config.Field("Subtract median image?", bool, True)
    def setDefaults(self):
        self.suffix = "_fringe"

class scaleFringeToScience(config.Config):
    suffix = config.Field("Filename suffix", str, "_fringeScaled")
    science = None # TODO
    stats_scale = config.Field("Scale by statistics rather than exposure time?", bool, False)

class standardizeStructureConfig(parameters_standardize.standardizeStructureConfig):
    def setDefaults(self):
        self.attach_mdf = False
