#Author: Kyle Mede, May 2010
#this module is being used as the workhorse for the prepare primitives.

#!/usr/bin/env python

import os
import pyfits as pf
import numpy as np
import time
from datetime import datetime
from astrodata.adutils import mefutil, paramutil, geminiLogger
from astrodata.AstroData import AstroData

def stdObsHdrs(ad):
   
    ##keywords that are updated/added for all Gemini PHUs 
    ad.phuSetKeyValue('NSCIEXT', ad.countExts("SCI"), 'Number of science extensions')
    ad.phuSetKeyValue('PIXSCALE', ad.pixel_scale(), 'Pixel scale in Y in arcsec/pixel')
    ad.phuSetKeyValue('NEXTEND', len(ad) , 'Number of extensions')
    ad.phuSetKeyValue('OBSMODE', ad.observation_mode() , 'Observing mode (IMAGE|IFU|MOS|LONGSLIT)')
    ad.phuSetKeyValue('COADDEXP', ad.phuValue("EXPTIME") , 'Exposure time for each coadd frame')
    
    numcoadds = ad.coadds()
    if not numcoadds:  
        numcoadds = 1      #for if there are no coadds performed, set to 1
    effExpTime = ad.phuValue("EXPTIME")*numcoadds    
    ad.phuSetKeyValue('EXPTIME', effExpTime , 'Effective exposure time') 

    ut = datetime.now().isoformat()  #$$$$$$$$$$$ just for saving the sintax, move this to its final destination when determined
    ad.phuSetKeyValue('GEM-TLM', ut , 'UT Last modification with GEMINI')  #$$$$$$$$$$$ just for saving the sintax, move this to its final destination when determined
         
    ##a loop to add the missing/needed keywords in the Gemini SCI extensions
    for ext in ad["SCI"]:
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'GAIN', ext.gain(), "Gain (e-/ADU)")
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'PIXSCALE', ext.pixel_scale(), 'Pixel scale in Y in arcsec/pixel')
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'RDNOISE', ext.read_noise() , "readout noise in e-")
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'BUNIT','adu' , 'Physical units')
        nonlin = ext.non_linear_level()
        if not nonlin:
            nonlin = 'None'     #if no nonlinear section provided then set to string 'None'
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'NONLINEA',nonlin , 'Non-linear regime level in ADU')
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'SATLEVEL',ext.saturation_level() , 'Saturationlevel in ADU')
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXPTIME', effExpTime , 'Saturationlevel in ADU')

def stdObsStruct(ad):
    ##a loop to add the missing/needed keywords in the SCI extensions
    for ext in ad["SCI"]:
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTNAME', 'SCI', "Extension name")        
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTVER', int(ext.header['EXTVER']), "Extension version")        
        
def fileNameUpdater(ad, debugLevel, postpend , strip=False):
    print 'gemini52: entered fileNameUpdater'
    infilename = os.path.basename(ad.filename)
    if strip:
        infilename = stripPostfix(infilename)
    (name,filetype) = os.path.splitext(infilename)
    if debugLevel>=2:  
        print 'gemini56: infilename = ', infilename
    outFileName = name+postpend+filetype
    ad.filename = outFileName
    if debugLevel>=2: 
        print 'gemini60: current output filename  = ',ad.filename
    
 

def stripPostfix(filename):
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    (name, filetype) = os.path.splitext(basename)
    a = name.split("_")
    name = a[0]
    retname = os.path.join(dirname,name+filetype)
    return retname    




   