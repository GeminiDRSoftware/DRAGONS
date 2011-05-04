from primitives_GMOS import GMOSPrimitives

import sys, StringIO, os

from astrodata.adutils import gemLog
from astrodata import Descriptors
from astrodata.data import AstroData
from astrodata.Errors import PrimitiveError
from gempy import geminiTools as gemt
from primitives_GMOS import GMOSPrimitives, pyrafLoader
import primitives_GEMINI
import primitives_GMOS

import numpy as np
import pyfits as pf
import shutil

from gwavecal import Wavecal


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
                wc = Wavecal(ad, reffile=rc['reffile'], wrdb=rc['wrdb'], fitfunction=rc['fitfunction'],
                                   fitorder=rc['fitorder'], ntmax=rc['ntmax'], fwidth=rc['fwidth'], 
                                   cradius=rc['cradius'], match=rc['match'], minsep=rc['minsep'], 
                                   clip=rc['clip'], nsum=rc['nsum'], debug=rc['debug'], logfile=rc['logfile'])
                wc.wavecal()
                wc.save_features()
                adOutputs.append(wc.outad)

            rc.reportOutput(adOutputs)

            log.status('wavecal  completed successfully')

        except:
            raise 
        yield rc

