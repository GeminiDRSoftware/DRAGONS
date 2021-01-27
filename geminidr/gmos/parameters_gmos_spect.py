# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_spect.py file, in alphabetical order.
from astrodata import AstroData
from gempy.library import config
from geminidr.core import parameters_spect

class QECorrectConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_QECorrected", optional=True)
    arc = config.ListField("Arc(s) with distortion map", (AstroData, str), None,
                           optional=True, single=True)
    use_iraf = config.Field("Use IRAF polynomial fits for Hamamatsu CCDs?",
                            bool, False)

class findAcquisitionSlitsConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_acqSlitsAdded", optional=True)

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    nbright = config.RangeField("Number of bright lines to eliminate", int, 0, min=0)

    def setDefaults(self):
        self.order = 3


class traceAperturesConfig(config.Config):

    debug = config.Field("Draw aperture traces on image display?",
                         bool, False)
    interactive = config.Field("Run primitive interactively?",
                               bool, False)
    max_missed = config.RangeField("Maximum number of steps to miss before a line is lost",
                                   int, 5, min=0)
    max_shift = config.RangeField("Maximum shift per pixel in line position",
                                  float, 0.05, min=0.001, max=0.1)
    niter = config.RangeField("Maximum number of rejection iterations",
                              int, None, min=0, optional=True)
    nsum = config.RangeField("Number of lines to sum",
                             int, 10, min=1)
    order = config.RangeField("Order of fitting function",
                              int, 2, min=1)
    sigma = config.RangeField("Rejection in sigma of fit",
                              float, 3, min=0, optional=True)
    step = config.RangeField("Step in rows/columns for tracing",
                             int, 10, min=1)
    suffix = config.Field("Filename suffix",
                          str, "_aperturesTraced", optional=True)

    def setDefaults(self):
        self.order = 2
