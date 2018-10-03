# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_spect.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_spect

class findAcquisitionSlitsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_acqSlitsAdded", optional=True)

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    def setDefaults(self):
        self.order = 5
