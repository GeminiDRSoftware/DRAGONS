# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos_spect.py file, in alphabetical order.
from astrodata import AstroData
from gempy.library import config
from geminidr.core import parameters_spect

def list_of_ints_check(value):
    [int(x) for x in str(value).split(',')]
    return True

class determineDistortionConfig(parameters_spect.determineDistortionConfig):
    spatial_order = config.RangeField("Fitting order in spatial direction", int, 3, min=1)
    spectral_order = config.RangeField("Fitting order in spectral direction", int, 1, min=1)
    id_only = config.Field("Use only lines identified for wavelength calibration?", bool, True)
    in_vacuo = config.Field("Use vacuum wavelength scale (rather than air)?", bool, True)
    #remove
    debug = config.Field("Display line traces on image display?", bool, True)

class determineWavelengthSolutionConfig(parameters_spect.determineWavelengthSolutionConfig):
    nbright = config.RangeField("Number of bright lines to eliminate", int, 0, min=0)
    threshold = config.RangeField("Intensity threshold for line detection", int, None, min=0, optional=True)
    debug_min_lines = config.Field("Minimum number of lines to fit each segment", (str, int),
                                  100000,
                                #'15,20',
                                   check=list_of_ints_check)
    def setDefaults(self):
        self.order = 1
