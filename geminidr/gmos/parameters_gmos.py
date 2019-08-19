# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_visualize, parameters_ccd
from geminidr.gemini import parameters_qa

class displayConfig(parameters_visualize.displayConfig):
    remove_bias = config.Field("Remove estimated bias level before displaying?", bool, True)

class measureBGConfig(parameters_qa.measureBGConfig):
    remove_bias = config.Field("Remove estimated bias level?", bool, True)

class measureIQConfig(parameters_qa.measureIQConfig):
    remove_bias = config.Field("Remove estimated bias level before displaying?", bool, True)

class subtractOverscanConfig(parameters_ccd.subtractOverscanConfig):
    nbiascontam = config.RangeField("Number of columns to exclude from averaging",
                               int, None, min=0, optional=True)
    def setDefaults(self):
        self.function = None

    def validate(self):
        config.Config.validate(self)
        if self.function == "spline" and self.order == 0:
            raise ValueError("Must specify a positive spline order, or None")

# We need to redefine this to ensure it inherits this version of
# subtractOverscanConfig.validate()
class overscanCorrectConfig(subtractOverscanConfig, parameters_ccd.trimOverscanConfig):
    def setDefaults(self):
        self.suffix = "_overscanCorrected"
