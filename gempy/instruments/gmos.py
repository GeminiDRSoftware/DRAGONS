#Author: Kyle Mede, May 2010
#this module is being used as the workhorse for the prepare primitives.

#!/usr/bin/env python

import os
import pyfits as pf
import numpy as np
import time
from datetime import datetime
from astrodata.adutils import mefutil, paramutil
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData

log=gemLog.getGeminiLog() 

def stdInstHdrs(ad):  
    # adding the missing/needed keywords into the headers 
    if not ad.isType('GMOS_IMAGE'):
    # do the stuff to the headers that is for the MOS, those for IMAGE are taken care of with stdObsHdrs all ready 
        for ext in ad["SCI"]:
            log.critical('gmostools22: still need to get the dispersion axis descriptor working you know!!','critical')
            #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'DISPAXIS',  , 'Dispersion axis')
            #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'',, "")
            pass

def valInstData(ad):
    # to ensure structure is the normal PHU followed by 3 SCI extensions for GMOS
    if len(ad)==1 or len(ad)==3 or len(ad)==6 or len(ad)==12:
        pass
    else: 
        log.critical("gmostools33: there are NOT 1, 3, 6 or 12 extensions in file = "+ad.filename,'critical')
        raise 
        
    
                 
def addMDF(ad,mdf): 
        # so far all of this is done in the primitive, will figure this out later
        pass
        
         
       