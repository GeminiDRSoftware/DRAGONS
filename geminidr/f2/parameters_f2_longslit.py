# This parameter file contains the parameters related to the primitives located
# in the primitives_f2_longslit.py file, in alphabetical order.
from geminidr.core import parameters_spect, parameters_standardize, parameters_telluric
from geminidr.f2 import parameters_f2_spect
from gempy.library import config


class addDQConfig(parameters_standardize.addDQConfig):
    keep_second_order = config.Field("Don't apply second order light mask?", bool, False)
    def setDefaults(self):
        self.add_illum_mask = True


class addIllumMaskToDQConfig(parameters_standardize.addIllumMaskToDQConfig):
    keep_second_order = config.Field("Don't apply second order light mask?", bool, False)


class addMDFConfig(config.Config):
    # Does not use MDF files
    suffix = config.Field("Filename suffix", str, "_mdfAdded", optional=True)

class fitTelluricConfig(parameters_telluric.fitTelluricConfig):
    order = config.RangeField("Order of fitting function", int, None, min=1, max=30,
                       inclusiveMax=True, optional=True)

class maskBeyondRegionsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_regionsMasked", optional=True)
    regions = config.Field('Wavelength regions (nm) to keep, eg. "1888:2200,2250:"',
                           str, ":",
                           check=parameters_spect.validate_regions_float)
    aperture = config.RangeField("Aperture to mask", int, 1, min=1, optional=True)

class normalizeFlatConfig(parameters_f2_spect.normalizeFlatConfig):
    def setDefaults(self):
        self.regions = "recommended"
        self.order = 2

class traceAperturesConfig(parameters_spect.traceAperturesConfig):
    def setDefaults(self):
        self.order = 3
