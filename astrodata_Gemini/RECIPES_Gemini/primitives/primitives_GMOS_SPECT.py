#Author: Kyle Mede, June 2010
# Skeleton originally written by Craig Allen, callen@gemini.edu

import sys
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import calibrate as cal
from gempy.science import standardization as sdz
from primitives_GMOS import GMOSPrimitives

class GMOS_SPECTPrimitives(GMOSPrimitives):
    """
    This is the class of all primitives for the GMOS_SPECT level of the type 
    hierarchy tree.  It inherits all the primitives to the level above
    , 'GMOSPrimitives'.
    """
    astrotype = "GMOS_SPECT"
    
    def init(self, rc):
        GMOSPrimitives.init(self, rc)
        return rc



