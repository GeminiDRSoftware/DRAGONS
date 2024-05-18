# This parameter file contains the parameters related to the primitives located
# in the primitives_ghost.py file, in alphabetical order.

from geminidr.core import parameters_standardize


class validateDataConfig(parameters_standardize.validateDataConfig):
    def setDefaults(self):
        self.require_wcs = False
