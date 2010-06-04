#Author: Kyle Mede, May 2010
#this class file is being used as the workhorse for the prepare primitives.


#!/usr/bin/env python

import os
import pyfits as pf
import numpy as np
import time
from astrodata.adutils import mefutil, paramutil, geminiLogger
from astrodata.AstroData import AstroData


def stdInstHdrs(self, ad):  
    ##adding the missing/needed keywords into the headers 
 

        if not ad.isType('GMOS_IMAGE'):
        ##do the stuff to the headers that is for the MOS 
            for ext in ad["SCI"]:
                #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'DISPAXIS',  , 'Dispersion axis')
                #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'',, "")
                pass

                 
def addMDF(self,ad,mdf,fullPrint=False): 
            
        maskname = ad.phuGetKeyValue("MASKNAME")
        ad.extSetKeyValue((maskname,len(ad)),'EXTNAME', 'MDF',"Extension name" )
        pass
        
         
       