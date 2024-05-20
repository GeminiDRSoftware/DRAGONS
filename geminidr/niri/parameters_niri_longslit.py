# This parameter file contains the parameters related to the primitive located
# in the primitives_niri_longslit.py file, in alphabetical order.
from geminidr.core import parameters_nearIR


class cleanReadoutConfig(parameters_nearIR.cleanReadoutConfig):
    # Need a larger extent to cope with a bright spectrum down the middle
    def setDefaults(self):
        self.smoothing_extent = 300
