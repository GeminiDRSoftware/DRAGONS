# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_longslit.py file, in alphabetical order.
from geminidr.core import parameters_standardize
from gempy.library import config
from astrodata import AstroData


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
    bins = config.Field("Total number of bins across the dispersion axis.",
                        int, None, optional=True)
    border = config.Field("Size of the border added to the reconstructed slit illumination image",
                          int, 0, optional=True)
    debug_plot = config.Field("Create diagnosis plots?",
                              bool, False, optional=True)
    smooth_order = config.Field("Spline order to smooth binned data",
                                int, 3, optional=True)
    suffix = config.Field("Filename suffix",
                          str, "_slitIllum", optional=True)
    x_order = config.Field("Order of the x-component of the Chebyshev2D model used to reconstruct data",
                           int, 4, optional=True)
    y_order = config.Field("Order of the y-component of the Chebyshev2D model used to reconstruct data",
                           int, 4, optional=True)


class normalizeFlatConfig(config.core_1Dfitting_config):
    suffix = config.Field("Filename suffix", str, "_normalized", optional=True)
    order = config.Field("Fitting order in spectral direction",
                                  (int, str), 20, check=flat_order_check)
    threshold = config.RangeField("Threshold for flagging unilluminated pixels",
                                  float, 0.01, min=0, inclusiveMin=False)
    interactive = config.Field("Interactive spline selection?", bool, False)
    debug_plot = config.Field("Create diagnosis plots?",
                              bool, False, optional=True)


class slitIllumCorrectConfig(config.Config):
    do_illum = config.Field("Perform Slit Illumination Correction?",
                            bool, True, optional=True)
    slit_illum = config.ListField("Slit Illumination Response",
                                  (str, AstroData), None, optional=True, single=True)
    suffix = config.Field("Filename suffix",
                          str, "_illumCorrected", optional=True)
