#Author: Kyle Mede, June 2010
# Skeleton originally written by Craig Allen, callen@gemini.edu
#from Reductionobjects import Reductionobject
from primitives_GMOS import GMOSPrimitives, pyrafLoader
# All GEMINI IRAF task wrappers.
import time
from astrodata.adutils import filesystem
from astrodata.adutils import gemLog
from astrodata import IDFactory
from astrodata import Descriptors
from astrodata.data import AstroData
from astrodata.Errors import PrimitiveError
from gempy import geminiTools as gemt

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



