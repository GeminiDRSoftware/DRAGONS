# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs.py file, in alphabetical order.
from geminidr.core import parameters_preprocess

class associateSkyConfig(parameters_preprocess.associateSkyConfig):
    def setDefaults(self):
        self.distance = 1.
