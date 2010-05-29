#Author: Kyle Mede, May 2010
#this class file is being used as the workhorse for the prepare primitives.


#!/usr/bin/env python

import os
import pyfits as pf
import numpy as np
import time
from adutils import mefutil, paramutil, geminiLogger
from astrodata.AstroData import AstroData

class PrepareTK():
    
     def __init__(self):
         pass

     def fixHeader(self, ad):  
         ##adding the missing/needed keywords into the headers 
              
         #print 'ptk20: types in ad; ', ad.types
 
         ##keywords that are updated/added for all GMOS images/spectra PHU
         ad.phuSetKeyValue('NSCIEXT', ad.nsciext(), 'Number of science extensions')
         ad.phuSetKeyValue('PIXSCALE', ad.pixscale(), 'Pixel scale in Y in arcsec/pixel')
         ad.phuSetKeyValue('NEXTEND', len(ad) , 'Number of extensions')
         ad.phuSetKeyValue('OBSMODE', ad.obsmode() , 'Observing mode (IMAGE|IFU|MOS|LONGSLIT)')
         ad.phuSetKeyValue('COADDEXP', ad.phuValue("EXPTIME") , 'Exposure time for each coadd frame')
         numcoadds = ad.coadds()
         if not numcoadds:  
             numcoadds = 1      #for if there are no coadds performed, set to 1
         effExpTime = ad.phuValue("EXPTIME")*numcoadds    
         ad.phuSetKeyValue('EXPTIME', effExpTime , 'Effective exposure time')     
         #ad.phuSetKeyValue('GEM-TLM',  , 'UT Last modification with GEMINI')
         #ad.phuSetKeyValue('',  , '')
         print ad.getPHUHeader()
         
         ##a loop to add the missing/needed keywords in the SCI extensions
         for ext in ad["SCI"]:
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'GAIN', ext.gain(asList=True)[0], "Gain (e-/ADU)")
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTVER', int(ext.header['EXTVER']), "Extension version")
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTNAME', 'SCI', "Extension name")
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'PIXSCALE', ext.pixscale(), 'Pixel scale in Y in arcsec/pixel')
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'RDNOISE', ext.rdnoise(asList=True)[0] , "readout noise in e-")
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'BUNIT','adu' , 'Physical units')
            nonlin = ext.nonlinear()
            if not nonlin:
                nonlin = 'None'     #if no nonlinear section provided then set to string 'None'
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'NONLINEA',nonlin , 'Non-linear regime level in ADU')
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'SATLEVEL',ext.satlevel() , 'Saturationlevel in ADU')
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXPTIME', effExpTime , 'Saturationlevel in ADU')
            
         if ad.isType('GMOS_IMAGE'):
             ##do the stuff to the headers that is for the MOS 
            
             ##a loop to add the missing/needed keywords in the SCI extensions specific for IMAGE
             for ext in ad["SCI"]:
                #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'',, "")
                print '-----------------------------------------------------------------'
                print 'ptk41: ', ext.getHeader() 
                  
             
         if ad.isType('GMOS_IFU'):   
            ##do the stuff to the headers that is for the IFU
            
            ##a loop to add the missing/needed keywords in the SCI extensions specific for IMAGE
            for ext in ad["SCI"]:
                #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'DISPAXIS',  , 'Dispersion axis')
                #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'',, "")
                pass
            pass
         if ad.isType('GMOS_MOS'):   
            ##do the stuff to the headers that is for the MOS
            
            ##a loop to add the missing/needed keywords in the SCI extensions specific for IMAGE
            for ext in ad["SCI"]:
                #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'DISPAXIS',  , 'Dispersion axis')
                #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'',, "")
                pass
            pass
         if ad.isType('GMOS_LONGSLIT'):   
            ##do the stuff to the headers that is for the LONGSLIT
            
            #a loop to add the missing/needed keywords in the SCI extensions specific for IMAGE
            pass
            for ext in ad["SCI"]:
                #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'DISPAXIS',  , 'Dispersion axis')
                #ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'',, "")
                pass
                 
                 
            
         
       