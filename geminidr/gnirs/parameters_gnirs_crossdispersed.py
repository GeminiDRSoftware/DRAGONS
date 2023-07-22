# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_crossdispersed.py file, in alphabetical order.
from geminidr.core import parameters_spect
from geminidr.core import parameters_standardize
from geminidr.core.parameters_standardize import addIllumMaskToDQConfig
from gempy.library import config


class addDQConfig(parameters_standardize.addDQConfig, addIllumMaskToDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    order = config.RangeField("Order of fitting function", int, 3, min=0,
                              optional=True)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, 2,
                                min=1., optional=True)

    def setDefaults(self):
        self.in_vacuo = True
