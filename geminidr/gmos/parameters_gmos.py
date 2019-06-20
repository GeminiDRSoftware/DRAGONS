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
