# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_crossdispersed.py file, in alphabetical order.
from geminidr.core import parameters_spect
from geminidr.core import parameters_crossdispersed
from geminidr.core import parameters_standardize
from geminidr.core.parameters_standardize import addIllumMaskToDQConfig
from gempy.library import config


def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True


class addDQConfig(parameters_standardize.addDQConfig, addIllumMaskToDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True


class determineSlitEdgesConfig(parameters_spect.determineSlitEdgesConfig):
    # GNIRS XD has narrow slits with more curvature than the longslit flats
    # the default values were calibrated to, so adjust some values.

    def setDefaults(self):
        self.debug_max_missed = 4
        self.debug_max_shift = 0.4
        self.debug_step = 10
        self.debug_nsum = 10
        del self.edge1
        del self.edge2

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    order = config.RangeField("Order of fitting function", int, 3, min=1,
                              optional=True)
    debug_min_lines = config.Field("Minimum number of lines to fit each segment",
                                   (str, int), '50,20',
                                   check=list_of_ints_check)
    def setDefaults(self):
        self.in_vacuo = True

class findAperturesConfig(parameters_crossdispersed.findAperturesConfig):
    # For cross-dispersed, allow the user to specify the extension to use for
    # finding apertures in. Will try to be a "best" one if not provided.
    ext = config.RangeField("Extension (1 - 6) to use for finding apertures",
                            int, None, optional=True, min=1, max=6,
                            inclusiveMin=True, inclusiveMax=True)

class normalizeFlatConfig(parameters_spect.normalizeFlatConfig):
    # Set flatfield threshold a little lower to avoid masking a region in
    # order 8.
    def setDefaults(self):
        self.threshold = 0.005


class skyCorrectFromSlitConfig(parameters_spect.skyCorrectFromSlitConfig):
    # Sky subtraction is difficult due to the short slit
    def setDefaults(self):
        self.function = "chebyshev"
        self.order = 2
        self.debug_allow_skip = True


class tracePinholeAperturesConfig(parameters_spect.tracePinholeAperturesConfig):
    """
    Configuration for the tracePinholeApertures() primitive.
    """
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.4, min=0.001, max=0.5, inclusiveMax=True)
