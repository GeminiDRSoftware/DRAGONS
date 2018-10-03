# This parameter file contains the parameters related to the primitives located
# in the primitives_spect.py file, in alphabetical order.
from gempy.library import config

class determineWavelengthSolutionConfig(config.Config):
    center = config.RangeField("Central row/column to extract", int, None, min=1, optional=True)
    nsum = config.RangeField("Number of lines to sum", int, 10, min=1)
    order = config.RangeField("Order of fitting polynomial", int, 2, min=1)
    central_wavelength = config.RangeField("Estimated central wavelength (A)", float, None,
                                           min=3500., max=20000., optional=True)
    dispersion = config.Field("Estimated dispersion (A/pixel)", float, None, optional=True)