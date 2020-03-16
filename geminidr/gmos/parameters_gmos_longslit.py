# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_longslit.py file, in alphabetical order.
from geminidr.core import parameters_standardize
from gempy.library import config

def flat_order_check(value):
    try:
        orders = [int(x) for x in value.split(',')]
    except AttributeError:  # not a str, must be int
        return value > 0
    except ValueError:  # items are not int-able
        return False
    else:
        return len(orders) == 3 and min(orders) > 0

class addDQConfig(parameters_standardize.addDQConfig):
    def setDefaults(self):
        self.add_illum_mask = True

class normalizeFlatConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_normalized", optional=True)
    spectral_order = config.Field("Fitting order in spectral direction",
                                  (int, str), 20, check=flat_order_check)
    threshold = config.RangeField("Threshold for flagging unilluminated pixels", float, 0.01, min=0)
    hsigma = config.RangeField("High rejection threshold (sigma)", float, 3., min=0)
    lsigma = config.RangeField("Low rejection threshold (sigma)", float, 3., min=0)
    grow = config.RangeField("Growth radius for bad pixels", int, 0, min=0)
