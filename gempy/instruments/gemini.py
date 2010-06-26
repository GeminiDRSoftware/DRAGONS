#Author: Kyle Mede, May 2010
#this module is being used as the workhorse for the prepare primitives.

#!/usr/bin/env python

import os
import pyfits as pf
import numpy as np
import time
from datetime import datetime
from astrodata.adutils import mefutil, paramutil
from astrodata.adutils.future import gemLog
from astrodata.AstroData import AstroData

log=gemLog.getGeminiLog() 

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
    ad.phuSetKeyValue("GPREPARE",ut,'fake UT Time stamp for GPREPARE')     ##$$$$ possible fix to gireduce flag to not call gprepare
    ##updating logger with updated/added keywords
    log.fullinfo('****************************************************','header')
    log.fullinfo('file = '+ad.filename,'header')
    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
    log.fullinfo('PHU keywords updated/added:\n', 'header')
    log.fullinfo('NSCIEXT = '+str(ad.countExts("SCI")),'header' )
    log.fullinfo('PIXSCALE = '+str(ad.pixel_scale()),'header' )
    log.fullinfo('NEXTEND = '+str(len(ad)),'header' )
    log.fullinfo('OBSMODE = '+str(ad.observation_mode()),'header' )
    log.fullinfo('COADDEXP = '+str(ad.phuValue("EXPTIME")),'header' )
    log.fullinfo('EXPTIME = '+str(effExpTime),'header' )
    log.fullinfo('GEM-TLM = '+str(ut),'header' )
    log.fullinfo('---------------------------------------------------','header')
         
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
        log.fullinfo('SCI extension number '+str(ext.header['EXTVER'])+' keywords updated/added:\n', 'header')
        log.fullinfo('GAIN = '+str(ext.gain()),'header' )
        log.fullinfo('PIXSCALE = '+str(ext.pixel_scale()),'header' )
        log.fullinfo('RDNOISE = '+str(ext.read_noise()),'header' )
        log.fullinfo('BUNIT = '+'adu','header' )
        log.fullinfo('NONLINEA = '+str(nonlin),'header' )
        log.fullinfo('SATLEVEL = '+str(ext.saturation_level()),'header' )
        log.fullinfo('EXPTIME = '+str(effExpTime),'header' )
        log.fullinfo('---------------------------------------------------','header')

def stdObsStruct(ad):
    log.fullinfo('****************************************************','header') 
    log.fullinfo('file = '+ad.filename, 'header')
    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~','header')
    ##a loop to add the missing/needed keywords in the SCI extensions
    for ext in ad["SCI"]:
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTNAME', 'SCI', "Extension name")        
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTVER', int(ext.header['EXTVER']), "Extension version") 
        log.fullinfo('SCI extension number '+str(ext.header['EXTVER'])+' keywords updated/added:\n', 'header')       
        log.fullinfo('EXTNAME = '+'SCI','header' )
        log.fullinfo('EXTVER = '+str(ext.header['EXTVER']),'header' )
        log.fullinfo('---------------------------------------------------','header')
        
def fileNameUpdater(ad, postpend='',prepend='' , strip=False):
    infilename = os.path.basename(ad.filename)
    if postpend !='':
        if strip:
            infilename = stripPostfix(infilename)
        (name,filetype) = os.path.splitext(infilename)
        outFileName = name+postpend+filetype
    elif prepend !='':
        if strip:
            infilename = stripPostfix(infilename)
        (name,filetype) = os.path.splitext(infilename)
        outFileName = prepend+name+filetype
    ad.filename = outFileName

def stripPostfix(filename):
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    (name, filetype) = os.path.splitext(basename)
    a = name.split("_")
    name = a[0]
    retname = os.path.join(dirname,name+filetype)
    return retname    


   