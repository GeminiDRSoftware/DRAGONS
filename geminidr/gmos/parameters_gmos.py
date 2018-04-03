# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_visualize, parameters_ccd, parameters_standardize
from geminidr.gemini import parameters_qa

class displayConfig(parameters_visualize.displayConfig):
    def setDefaults(self):
        self.remove_bias = True

class measureBGConfig(parameters_qa.measureBGConfig):
    def setDefaults(self):
        self.remove_bias = True

class subtractOverscanConfig(parameters_ccd.subtractOverscanConfig):
    def setDefaults(self):
        self.function = None

class validateDataConfig(parameters_standardize.validateDataConfig):
    def setDefaults(self):
        self.num_exts = [1,2,3,4,6,12]
