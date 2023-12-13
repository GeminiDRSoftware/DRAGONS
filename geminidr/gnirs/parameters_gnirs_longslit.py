# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_longslit.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_standardize
from geminidr.core import parameters_spect
from geminidr.core.parameters_standardize import addIllumMaskToDQConfig


def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True

class determineDistortionConfig(parameters_spect.determineDistortionConfig):
    spectral_order = config.RangeField("Fitting order in spectral direction", int, None, min=1, optional=True)
    min_line_length = config.RangeField("Exclude line traces shorter than this fraction of spatial dimension",
                                        float, None, min=0., max=1., optional=True)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost",
                               int, None, min=0, optional=True)
    def setDefaults(self):
        self.min_snr = 10
        self.debug_reject_bad = False

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    order = config.RangeField("Order of fitting function", int, None, min=0,
                              optional=True)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, None, min=0.1, optional=True)
    debug_min_lines = config.Field("Minimum number of lines to fit each segment", (str, int), None,
                                   check=list_of_ints_check, optional=True)
    num_atran_lines = config.RangeField("Max number of lines in ATRAN line list", int, None,
                                              min=10, max=300, inclusiveMax=True, optional=True)
    combine_method = config.ChoiceField("Combine method to use in 1D spectrum extraction", str,
                                   allowed={"mean": "mean",
                                            "median": "median",
                                            "optimal" : "auto-select depending on the mode"},
                                   default="optimal")
    def setDefaults(self):
        self.in_vacuo = True

class addDQConfig(parameters_standardize.addDQConfig):
    keep_second_order = config.Field("Don't apply second order light mask?", bool, False)
    def setDefaults(self):
        self.add_illum_mask = True

class addIllumMaskToDQConfig(parameters_standardize.addIllumMaskToDQConfig):
    keep_second_order = config.Field("Don't apply second order light mask?", bool, False)
