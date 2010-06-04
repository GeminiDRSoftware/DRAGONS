#Author: Kyle Mede, May 2010
#this class file is being used as the workhorse for the prepare primitives.


#!/usr/bin/env python

import os
import pyfits as pf
import numpy as np
import time
from astrodata.adutils import mefutil, paramutil, geminiLogger
from astrodata.AstroData import AstroData

def stdObsHdrs(ad):
    fullPrint = False ##$$$$ TEMP PRINT OPTION TO PRINT ALL THE HEADERS WHILE RUNNING FOR DEBUGGING
    
    ##keywords that are updated/added for all GMOS images/spectra PHU
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
    ut = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) #FIX THIS WITH PAULS FIX SOON!!!!
    ad.phuSetKeyValue('GEM-TLM', ut , 'UT Last modification with GEMINI')
    #ad.phuSetKeyValue('',  , '')
    if fullPrint:
        print ad.getPHUHeader()
         
    ##a loop to add the missing/needed keywords in the SCI extensions
    for ext in ad["SCI"]:
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'GAIN', ext.gain(), "Gain (e-/ADU)")
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTVER', int(ext.header['EXTVER']), "Extension version")
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTNAME', 'SCI', "Extension name")
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'PIXSCALE', ext.pixel_scale(), 'Pixel scale in Y in arcsec/pixel')
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'RDNOISE', ext.read_noise() , "readout noise in e-")
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'BUNIT','adu' , 'Physical units')
        nonlin = ext.non_linear_level()
        if not nonlin:
            nonlin = 'None'     #if no nonlinear section provided then set to string 'None'
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'NONLINEA',nonlin , 'Non-linear regime level in ADU')
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'SATLEVEL',ext.saturation_level() , 'Saturationlevel in ADU')
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXPTIME', effExpTime , 'Saturationlevel in ADU')
            