from primitives_GMOS import GMOSPrimitives

import sys, StringIO, os

from astrodata.adutils import gemLog
from astrodata import Descriptors
from astrodata.data import AstroData
from astrodata.Errors import PrimitiveError
from gempy.instruments import geminiTools as gemt
from gempy.instruments import gmosTools as gmost
from primitives_GMOS import GMOSPrimitives, pyrafLoader
import primitives_GEMINI
import primitives_GMOS

import numpy as np
import pyfits as pf
import shutil

class GMOS_LS_ARCPrimitives(GMOSPrimitives):
    """ 
    This is the class of all primitives for the GMOS level of the type 
    hierarchy tree.  It inherits all the primitives to the level above
    , 'GEMINIPrimitives'.
    
    """
    astrotype = 'GMOS_LS_ARC'
    
    def init(self, rc):
        GMOSPrimitives.init(self, rc)
        return rc
     
    def gtransform(self, rc):
        log = gemLog.getGeminiLog(logName=rc['logName'], 
                                  logLevel=rc['logLevel'])
        try:
            print "Starting gtrans GMOS_LS_ARC"
            #gtrans.gtrans.Gtrans('gsN20011222S027.fits',minsep=4,ntmax=50)
            #gg = gtrans.Gtrans(rc.inputsAsStr(), minsep=4,ntmax=50)

        except:
            raise PrimitiveError("Problem with gtransform")

        yield rc
