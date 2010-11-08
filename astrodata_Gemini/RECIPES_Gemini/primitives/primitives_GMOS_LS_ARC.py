from primitives_GMOS import GMOSPrimitives

import sys, StringIO, os

from astrodata.adutils import gemLog
from astrodata import Descriptors
from astrodata.data import AstroData
from gempy.instruments import geminiTools as gemt
from gempy.instruments import gmosTools as gmost
from primitives_GMOS import GMOSPrimitives, pyrafLoader
import primitives_GEMINI
import primitives_GMOS

import numpy as np
import pyfits as pf
import shutil

log=gemLog.getGeminiLog()

class GMOS_LS_ARCException:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    
    """
    def __init__(self, message='Exception Raised in Recipe System'):
        """This constructor takes a message to print to the user."""
        self.message = message
    def __str__(self):
        """This str conversion member returns the message given 
        by the user (or the default message)
        when the exception is not caught."""
        return self.message

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
        try:
            print "Starting gtrans GMOS_LS_ARC"
            #gtrans.gtrans.Gtrans('gsN20011222S027.fits',minsep=4,ntmax=50)
            #gg = gtrans.Gtrans(rc.inputsAsStr(), minsep=4,ntmax=50)

        except:
            raise GMOSException("Problem with gtransform")

        yield rc
