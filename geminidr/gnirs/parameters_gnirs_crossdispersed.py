# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_crossdispersed.py file, in alphabetical order.
from geminidr.core import parameters_spect
from geminidr.core import parameters_standardize
from geminidr.core.parameters_standardize import addIllumMaskToDQConfig
from gempy.library import config

def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True


class addDQConfig(parameters_standardize.addDQConfig, addIllumMaskToDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    order = config.RangeField("Order of fitting function", int, 3, min=0,
                              optional=True)
    debug_min_lines = config.Field("Minimum number of lines to fit each segment",
                                   (str, int), '50,20',
                                   check=list_of_ints_check)

    def setDefaults(self):
        self.in_vacuo = True

class tracePinholeAperturesConfig(config.core_1Dfitting_config):
    """
    Configuration for the tracePinholeApertures() primitive.
    """
    suffix = config.Field("Filename suffix",
                          str, "_pinholeAperturesTraced", optional=True)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost",
                                   int, 4, min=0)
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.4, min=0.001, max=0.5, inclusiveMax=True)
    min_line_length = config.RangeField("Minimum line length as a fraction of array",
                                        float, 0, min=0, max=1, inclusiveMin=True,
                                        inclusiveMax=True)
    min_snr = config.RangeField("Minimum SNR for apertures", float, 10., min=0.)
    nsum = config.RangeField("Number of lines to sum", int, 6, min=1)
    step = config.RangeField("Step in rows/columns for tracing", int, 6, min=1)
    spectral_order = config.RangeField("Order of fit in spectral direction",
                                       int, 3, min=1)
    # GNIRS has 4 good pinholes, and a fifth one which is partial vignetted
    # and which doesn't produce smooth traces. It's also only found in some
    # orders, so the default parameters here ensure only the first 4 traces are
    # used.
    debug_min_trace_pos = config.Field("Use traces above this number",
                                       dtype=int, default=None, optional=True)
    debug_max_trace_pos = config.Field("Use traces below this number",
                                       dtype=int, default=4, optional=True)