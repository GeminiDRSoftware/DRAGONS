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
    # the default values were calibrated to, so adjust the values.
    debug_max_missed = config.RangeField("Maximum missed steps when tracing edges",
                                         int, 4, min=1)
    debug_max_shift = config.RangeField("Maximum perpendicular shift (in pixels) per pixel",
                                        float, 0.3, min=0.)
    debug_step = config.RangeField("Step size (in pixels) for fitting edges",
                                   int, 10, min=5)
    debug_nsum = config.RangeField("Columns/rows to sum each step when fitting edges",
                                   int, 10, min=5)

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    order = config.RangeField("Order of fitting function", int, 3, min=0,
                              optional=True)
    debug_min_lines = config.Field("Minimum number of lines to fit each segment",
                                   (str, int), '50,20',
                                   check=list_of_ints_check)
    def setDefaults(self):
        self.in_vacuo = True

class findAperturesConfig(parameters_crossdispersed.findAperturesConfig):
    ext = config.RangeField("Extension (1 - 6) to use for finding apertures",
                            int, None, optional=True, min=1, max=6,
                            inclusiveMin=True, inclusiveMax=True)

class normalizeFlatConfig(parameters_spect.normalizeFlatConfig):
    # Set flatfield threshold a little lower to avoid masking a region in
    # order 8.
    threshold = config.RangeField("Threshold for flagging unilluminated pixels",
                                  float, 0.005, min=0.005, max=1.0)


class tracePinholeAperturesConfig(parameters_spect.tracePinholeAperturesConfig):
    """
    Configuration for the tracePinholeApertures() primitive.
    """
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.4, min=0.001, max=0.5, inclusiveMax=True)
