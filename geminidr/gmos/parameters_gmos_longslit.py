# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_longslit.py file, in alphabetical order.
from geminidr.core import parameters_standardize
from gempy.library import config
from astrodata import AstroData
from geminidr.core import parameters_generic
from geminidr.core import parameters_stack


def flat_order_check(value):
    try:
        orders = [int(x) for x in value.split(',')]
    except AttributeError:  # not a str, must be int
        return value > 0
    except ValueError:  # items are not int-able
        return False
    else:
        return len(orders) == 3 and min(orders) > 0


class addIllumMaskToDQConfig(parameters_standardize.addIllumMaskToDQConfig):
    shift = config.RangeField("User-defined shift for illumination mask", int, None,
                              min=-100, max=100, inclusiveMax=True, optional=True)
    max_shift = config.RangeField("Maximum (unbinned) pixel shift for illumination mask",
                                  int, 20, min=0, max=100, inclusiveMax=True)


class addDQConfig(parameters_standardize.addDQConfig, addIllumMaskToDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True   # adds bridges in longslit full frame


class makeSlitIllumConfig(config.Config):
    bins = config.Field("Either total number of bins across the dispersion axis, "
                        "or a comma-separated list \n                                          "
                        "of pixel coordinate pairs defining then dispersion bins, e.g. 1:300,301:500",
                        (int, str), None, optional=True)
    debug_plot = config.Field("Create diagnosis plots?",
                              bool, False, optional=True)
    suffix = config.Field("Filename suffix",
                          str, "_slitIllum", optional=True)
    interactive = config.Field("Set to activate an interactive preview to fine tune the input parameters",
                               bool, False, optional=True)
    regions = config.Field("Sample regions along the slit", str, None, optional=True)
    spat_function = config.ChoiceField("Fitting function to use for bin fitting (spatial direction)", str,
                           allowed={"spline3": "Cubic spline",
                                    "chebyshev": "Chebyshev polynomial",
                                    "legendre": "Legendre polynomial",
                                    "spline1": "Linear spline"},
                           default="spline3", optional=False)
    spat_order = config.Field("Order of the bin fitting function",
                                int, 20, optional=True)
    disp_function = config.ChoiceField("Fitting function to use for row fitting (dispersion direction)", str,
                           allowed={"spline3": "Cubic spline",
                                    "chebyshev": "Chebyshev polynomial",
                                    "legendre": "Legendre polynomial",
                                    "spline1": "Linear spline"},
                           default="spline3", optional=False)
    disp_order = config.Field("Order of the row fitting function",
                                int, 6, optional=True)
    hsigma = config.RangeField("High rejection threshold (sigma) of the bin fit",
                               float, 3., min=0)
    lsigma = config.RangeField("Low rejection threshold (sigma) of the bin fit",
                               float, 3., min=0)
    niter = config.RangeField("Maximum number of iterations",
                              int, 3, min = 0, optional=True)
    grow = config.RangeField("Growth radius for rejected pixels of the bin fit",
                             float, 0, min=0, optional=True)
    border = config.Field("Size of the border added to the reconstructed slit illumination image",
                          int, 2, optional=True)

class normalizeFlatConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_normalized", optional=True)
    order = config.Field("Fitting order in spectral direction",
                                  (int, str), 20, check=flat_order_check)
    threshold = config.RangeField("Threshold for flagging unilluminated pixels",
                                  float, 0.01, min=0, inclusiveMin=False)
    interactive = config.Field("Interactive fitting?", bool, False)
    debug_plot = config.Field("Create diagnosis plots?",
                              bool, False, optional=True)

    def setDefaults(self):
        self.niter = 3


class slitIllumCorrectConfig(parameters_generic.calRequirementConfig):
    slit_illum = config.ListField("Slit Illumination Response",
                                  (str, AstroData), None, optional=True, single=True)
    suffix = config.Field("Filename suffix",
                          str, "_illumCorrected", optional=True)

class stackFramesConfig(parameters_stack.stackFramesConfig):
    def setDefaults(self):
        self.reject_method = "varclip"
