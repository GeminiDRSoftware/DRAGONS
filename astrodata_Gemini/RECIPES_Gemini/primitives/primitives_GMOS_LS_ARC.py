from primitives_GMOS import GMOSPrimitives

import sys, StringIO, os

from astrodata.adutils import gemLog
from astrodata import Descriptors
from astrodata.data import AstroData
from astrodata.Errors import PrimitiveError
from gempy import geminiTools as gemt
from gempy.instruments import gmosTools as gmost
from primitives_GMOS import GMOSPrimitives, pyrafLoader
import primitives_GEMINI
import primitives_GMOS

import numpy as np
import pyfits as pf
import shutil

from gwavecal import GmosLONGSLIT


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
     
    def wavecal(self, rc):
        """
          Wavelength calibration primitive
        """
        log = gemLog.createGeminiLog(logName=rc['logName'], logLevel=rc['logLevel'])

        adOutputs = []
        log.info( "STARTING Wavecal")

        try:

            for ad in rc.getInputs(style='AD'):

                log.info('\n*** Wavecal primitive. Processing file:'+ad.filename)
                gls = GmosLONGSLIT(ad, reffile=rc['reffile'], wrdb=rc['wrdb'], fitfunction=rc['fitfunction'],
                                   fitorder=rc['fitorder'], ntmax=rc['ntmax'], fwidth=rc['fwidth'], 
                                   cradius=rc['cradius'], match=rc['match'], minsep=rc['minsep'], 
                                   clip=rc['clip'], nsum=rc['nsum'], debug=rc['debug'], logfile=rc['logfile'])
                gls.wavecal()
                gls.save_features()
                adOutputs.append(gls.outad)

            rc.reportOutput(adOutputs)

            log.status('wavecal  completed successfully')

        except:
            raise PrimitiveError("Problems with wavecal")

        yield rc

    def gtransform(self, rc):
        log = gemLog.getGeminiLog(logName=rc['logName'], 
                                  logLevel=rc['logLevel'])
        try:
            print "Starting gtrans GMOS_LS_ARC"
            #gtrans.gtrans.Gtrans('gsN20011222S027.fits',minsep=4,ntmax=50)
            #gg = gtrans.Gtrans(rc.inputsAsStr(), minsep=4,ntmax=50)

        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general PrimitiveError 
            # with message.
            log.critical(repr(sys.exc_info()[1]))
            raise PrimitiveError("Problem with gtransform")

        yield rc
