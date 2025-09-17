# This parameter file contains the parameters related to the primitives located
# in the primitives_gnirs_crossdispersed.py file, in alphabetical order.
from geminidr.core import parameters_spect
from geminidr.core import parameters_crossdispersed
from geminidr.core import parameters_standardize
from geminidr.core.parameters_standardize import addIllumMaskToDQConfig
from . import parameters_gnirs_spect
from gempy.library import config


def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True


class addDQConfig(parameters_standardize.addDQConfig, addIllumMaskToDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True


class determineDistortionConfig(parameters_gnirs_spect.determineDistortionConfig):
    debug_min_relative_peak_height = config.RangeField("Minimum relative peak height for tracing", float, 0.7, min=0., max=1.)

    def setDefaults(self):
        self.spatial_order = 2
        self.step = 5
        self.min_line_length = 0.8  # need to keep high to avoid false positives


class determinePinholeRectificationConfig(parameters_spect.determinePinholeRectificationConfig):
    """
    Configuration for the determinePinholeRectification() primitive.
    """
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.4, min=0.001, max=0.5, inclusiveMax=True)


class determineSlitEdgesConfig(parameters_spect.determineSlitEdgesConfig):
    # GNIRS XD has narrow slits with more curvature than the longslit flats
    # the default values were calibrated to, so adjust some values.

    def setDefaults(self):
        self.min_snr = 0.5
        self.debug_max_missed = 4
        self.debug_max_shift = 0.4
        self.debug_step = 10
        self.debug_nsum = 10
        del self.edge1
        del self.edge2


class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    order = config.RangeField("Order of fitting function", int, None, min=1,
                              optional=True)
    min_snr = config.RangeField("Minimum SNR for peak detection", float, None, min=0.1, optional=True)

    debug_min_lines = config.Field("Minimum number of lines to fit each segment",
                                   (str, int), '50,20',
                                   check=list_of_ints_check)
    num_lines = config.RangeField("Number of lines in the generated line list", int, 50,
                                              min=10, max=1000, inclusiveMax=True, optional=True)
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
        self.order = 1
        self.aperture_growth = 1
        self.debug_allow_skip = True
        self.grow = 1


class traceAperturesConfig(parameters_spect.traceAperturesConfig):
    # GNIRS XD benefits from light sigma clipping.
    def setDefaults(self):
        self.niter = 1
