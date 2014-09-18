import sys

import numpy as np
from astrodata import AstroData
from astrodata.adutils import logutils
from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at
from astrodata import Lookups

import pywcs

from primitives_NIRI import NIRIPrimitives

class NIRI_IMAGEPrimitives(NIRIPrimitives):
    """
    This is the class containing all of the primitives for the NIRI_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'NIRIPrimitives'.
    """
    astrotype = "NIRI_IMAGE"
    
    def init(self, rc):
        NIRIPrimitives.init(self, rc)
        return rc
