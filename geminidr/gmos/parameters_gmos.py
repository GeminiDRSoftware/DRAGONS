# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_visualize, parameters_ccd, parameters_standardize
from geminidr.gemini import parameters_qa

class displayConfig(parameters_visualize.displayConfig):
    remove_bias = config.Field("Remove estimated bias level before displaying?", bool, False)

class mosaicDetectorsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_mosaicked")
    tile = config.Field("Tile rather than mosaic?", bool, False)
    interpolate_gaps = config.Field("Interpolate across chip gaps?", bool, False)
    interpolator = config.Field("Type of interpolation", str, "linear")

class measureBGConfig(parameters_qa.measureBGConfig):
    remove_bias = config.Field("Remove bias level?", bool, False)

class measureIQConfig(parameters_qa.measureIQConfig):
    remove_bias = config.Field("Remove bias level before displaying?", bool, False)

    def setDefaults(self):
        self.remove_bias = True

class subtractOverscanConfig(parameters_ccd.subtractOverscanConfig):
    def setDefaults(self):
        self.function = None

class validateDataConfig(parameters_standardize.validateDataConfig):
    def setDefaults(self):
        self.num_exts = [1,2,3,4,6,12]
