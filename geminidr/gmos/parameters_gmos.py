# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos.py file, in alphabetical order.
from gempy.library import config
from geminidr.core import parameters_visualize, parameters_ccd, parameters_standardize
from geminidr.gemini import parameters_qa

def badamps_check(value):
    try:
        badamps = [int(x) for x in value.split(',')]
    except AttributeError:  # not a str, must be int
        if value is None:
            return True
        return value > 0
    except ValueError:  # items are not int-able
        return False
    else:
        return len(badamps) >= 1 and min(badamps) > 0


class displayConfig(parameters_visualize.displayConfig):
    remove_bias = config.Field("Remove estimated bias level before displaying?", bool, True)


class measureBGConfig(parameters_qa.measureBGConfig):
    remove_bias = config.Field("Remove estimated bias level?", bool, True)


class measureIQConfig(parameters_qa.measureIQConfig):
    remove_bias = config.Field("Remove estimated bias level before displaying?", bool, True)


class maskFaultyAmpConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_badAmpMasked", optional=True)
    instrument = config.Field("Applicable instrument", str, None, optional=True)
    bad_amps = config.Field("Amps to mask as a list", (int, str), None,
                                optional=True, check=badamps_check)
    valid_from = config.Field("Mask data taken after this date (YYYYMMDD)", str, None, optional=True)
    valid_to = config.Field("Mask data taken before this date (YYYYMMDD)", str, None, optional=True)


class standardizeWCSConfig(parameters_standardize.standardizeWCSConfig):
    pass


class subtractOverscanConfig(parameters_ccd.subtractOverscanConfig):
    nbiascontam = config.RangeField("Number of columns to exclude from averaging",
                               int, None, min=0, optional=True)
    def setDefaults(self):
        self.function = "none"

    def validate(self):
        config.Config.validate(self)
        if self.function == "spline" and self.order == 0:
            raise ValueError("Must specify a positive spline order, or None")


# We need to redefine this to ensure it inherits this version of
# subtractOverscanConfig.validate()
class overscanCorrectConfig(subtractOverscanConfig, parameters_ccd.trimOverscanConfig):
    def setDefaults(self):
        self.suffix = "_overscanCorrected"
