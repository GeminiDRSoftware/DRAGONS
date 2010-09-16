#Author: Kyle Mede, May 2010
#this module is being used as the workhorse for the prepare primitives.

#!/usr/bin/env python

import os
import pyfits as pf
import numpy as np
#from astrodata.adutils import mefutil, paramutil
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData

log=gemLog.getGeminiLog() 

def stdObsHdrs(ad):
    
    ## keywords that are updated/added for all Gemini PHUs 
    ad.phuSetKeyValue('NSCIEXT', ad.countExts("SCI"), 'Number of science extensions')
    ad.phuSetKeyValue('PIXSCALE', ad.pixel_scale(), 'Pixel scale in Y in arcsec/pixel')
    ad.phuSetKeyValue('NEXTEND', len(ad) , 'Number of extensions')
    ad.phuSetKeyValue('OBSMODE', ad.observation_mode() , 'Observing mode (IMAGE|IFU|MOS|LONGSLIT)')
    ad.phuSetKeyValue('COADDEXP', ad.phuValue("EXPTIME") , 'Exposure time for each coadd frame')
    numcoadds = ad.coadds()
    if not numcoadds:  
        numcoadds = 1      # for if there are no coadds performed, set to 1
    effExpTime = ad.phuValue("EXPTIME")*numcoadds    
    ad.phuSetKeyValue('EXPTIME', effExpTime , 'Effective exposure time') 
    ad.phuSetKeyValue('NCOADD', str(numcoadds) , 'Number of coadds')

    ut = ad.historyMark()
    ad.historyMark(key="GPREPARE",stomp=False) 
       
    ## updating logger with updated/added keywords
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
         
    ## a loop to add the missing/needed keywords in the Gemini SCI extensions
    for ext in ad["SCI"]:
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'GAIN', ext.gain(), "Gain (e-/ADU)")
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'PIXSCALE', ext.pixel_scale(), 'Pixel scale in Y in arcsec/pixel')
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'RDNOISE', ext.read_noise() , "readout noise in e-")
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'BUNIT','adu' , 'Physical units')
        nonlin = ext.non_linear_level()
        #print nonlin
        #print type(nonlin)
        if not nonlin:
            nonlin = 'None'     # if no nonlinear section provided then set to string 'None'
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'NONLINEA',nonlin , 'Non-linear regime level in ADU')
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'SATLEVEL',ext.saturation_level() , 'Saturationlevel in ADU')
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXPTIME', effExpTime , 'Effective exposure time')
        
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
    ## a loop to add the missing/needed keywords in the SCI extensions
    for ext in ad["SCI"]:
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTNAME', 'SCI', "Extension name")        
        ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'EXTVER', int(ext.header['EXTVER']), "Extension version") 
        
        log.fullinfo('SCI extension number '+str(ext.header['EXTVER'])+' keywords updated/added:\n', 'header')       
        log.fullinfo('EXTNAME = '+'SCI','header' )
        log.fullinfo('EXTVER = '+str(ext.header['EXTVER']),'header' )
        log.fullinfo('---------------------------------------------------','header')
        
def fileNameUpdater(origfilename, postpend='',prepend='' , strip=False):
    
    infilename = os.path.basename(origfilename)
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
    return outFileName

def stripPostfix(filename):
    
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    (name,filetype) = os.path.splitext(basename)
    a = name.split("_")
    name = a[0]
    retname = os.path.join(dirname,name+filetype)
    return retname    

def secStrToIntList(string):
    
    ## string must be of the format '[#1:#2,#3:#4]', returns list of ints [#1,#2,#3,#4]
    coords=string.strip('[').strip(']').split(',')
    Ys=coords[0].split(':')
    Xs=coords[1].split(':')
    retl=[]
    retl.append(int(Ys[0]))
    retl.append(int(Ys[1]))
    retl.append(int(Xs[0]))
    retl.append(int(Xs[1]))
    return retl
    
def biassecStrTonbiascontam(biassec,ad):
    ''' 
    This function works with nbiascontam() of the CLManager. 
    It will find the largest horizontal difference between biassec and BIASSEC for each SCI extension in a single input.
    
    @param biassec: biassec parameter of format '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
    @type biassec: string  
    
    @param ad: AstroData instance
    @type ad: AstroData instance
    '''
    try:
        ccdStrList=biassec.split('],[')
        ccdIntList=[]
        for string in ccdStrList:
            ccdIntList.append(secStrToIntList(string))
            
        retvalue=0
        for i in range(0,len(ad['SCI'])):
            BIASSEClist=secStrToIntList(ad.extGetKeyValue(i,'BIASSEC'))
            biasseclist=ccdIntList[i]
            if biasseclist[2]==BIASSEClist[2] and biasseclist[3]==BIASSEClist[3]:
                if biasseclist[0]<50: # ie overscan/bias section is on the left side of chip
                    if biasseclist[0]==BIASSEClist[0]: 
                        nbiascontam=BIASSEClist[1]-biasseclist[1]
                    else:
                        log.error('left horizontal components of biassec and BIASSEC did not match, so using default nbiascontam=4','error')
                        nbiascontam=4
                else: # ie overscan/bias section is on the right side of chip
                    if biasseclist[1]==BIASSEClist[1]: 
                        nbiascontam=BIASSEClist[0]-biasseclist[0]
                    else:
                        log.error('right horizontal components of biassec and BIASSEC did not match, so using default nbiascontam=4','error') 
                        nbiascontam=4
            else:
                log.error('vertical components of biassec and BIASSEC parameters did not match, so using default nbiascontam=4', 'error')
                nbiascontam=4
            if nbiascontam>retvalue: # ie returning the largest nbiascontam value throughout all chips 
                retvalue=nbiascontam
            
        return retvalue
            
    except:
        log.error('An error ocured while trying to calculate the nbiascontam, so using default value = 4','error')
        return 4 # so exiting 'gracefully' and still returning a value
        
        
def pyrafBoolean(pythonBool):
    '''
    A very basic function to reduce code repetition that simply 'casts' any given 
    Python boolean into a pyraf/iraf one for use in the CL scripts.
    '''
    import pyraf
    
    if pythonBool:
        return pyraf.iraf.yes
    elif  not pythonBool:
        return pyraf.iraf.no
    else:
        print "DANGER DANGER Will Robinson, pythonBool passed in was not True or False, and thats just crazy talk :P"

class CLManager(object):
    _preCLcachestorenames = [] # the version of the names for input to the CL script
    _preCLfilenames = [] # the original names of the files at the start of the primitive which called CLManager
    rc = None
    prefix = None
    outpref = None
    listname = None
    
    def __init__(self, rc, outpref = None):
        self.rc  = rc
        if outpref == None:
            outpref = rc["outpref"]
        self.outpref=outpref
        self._preCLcachestorenames=[]
        self._preCLfilenames = []
        self.prefix = self.uniquePrefix()
        self.preCLwrites()
    
    # perform all the finalizing steps after CL script is ran, currently just an alias for postCLloads
    def finishCL(self,combine=False): 
        self.postCLloads(combine)    
    
    def preCLwrites(self):
        for ad in self.rc.getInputs(style="AD"):
            self._preCLfilenames.append(ad.filename)
            name = fileNameUpdater(ad.filename,prepend=self.prefix,strip=True)
            self._preCLcachestorenames.append(name)
            log.fullinfo('Temporary file on disk for input to CL: '+name,'CLprep')
            ad.write(name, rename = False) 
    
    # just a function to return the 'private' member variable _preCLcachestorenames
    def cacheStoreNames(self):
        return self._preCLcachestorenames
       
    # A function to remove the filenames written to disk by setStackable 
    def rmStackFiles(self):
        for file in self._preCLfilenames:
            log.fullinfo('removing file '+file+' from disk', 'postCL')
            os.remove(file)
        
    # just a function to return the 'private' member variable _preCLfilenames
    def preCLNames(self):
        return self._preCLfilenames
    
    def inputsAsStr(self):
        return ",".join(self._preCLcachestorenames)
    
    def inputList(self):
        self.listname='List'+str(os.getpid())+self.rc.ro.curPrimName
        return self.rc.makeInlistFile(self.listname,self._preCLcachestorenames)
        
    def uniquePrefix(self):
        return "tmp"+ str(os.getpid())+self.rc.ro.curPrimName
    
    def combineOutname(self):
        #@@ REFERENCE IMAGE: for output name
        return self.outpref+self._preCLcachestorenames[0]
    
    def postCLloads(self,combine=False):
        if combine==True:
            cloutname=self.outpref+self._preCLcachestorenames[0]
            finalname=fileNameUpdater(self._preCLfilenames[0], postpend= self.outpref, strip=False)
            os.rename(cloutname, finalname )
            self.rc.reportOutput(finalname)
            os.remove(finalname)
            #print 'g209: self.listname = ',self.listname
            os.remove(self.listname)
            log.fullinfo('CL outputs '+cloutname+' was renamed on disk to:\n'+finalname,'postCL')
            log.fullinfo(finalname+' was loaded into memory', 'postCL')
            log.fullinfo(finalname+' was deleted from disk', 'postCL')
            log.fullinfo(self.listname+' was deleted from disk','postCL')
            
            for i in range(0, len(self._preCLcachestorenames)):
                storename = self._preCLcachestorenames[i]  # name of file written to disk for input to CL script
                os.remove(storename) # clearing renamed file ouput by CL
                log.fullinfo(storename+' was deleted from disk', 'postCL')
                
        elif combine==False:
            for i in range(0, len(self._preCLcachestorenames)):
                storename = self._preCLcachestorenames[i]  # name of file written to disk for input to CL script
                cloutname = self.outpref + storename  # name of file CL wrote to disk
                finalname = fileNameUpdater(self._preCLfilenames[i], postpend= self.outpref, strip=False)  # name i want the file to be
                
                os.rename(cloutname, finalname )
                
                # THIS LOADS THE FILE INTO MEMORY
                self.rc.reportOutput(finalname)
                
                os.remove(finalname) # clearing file written for CL input
                os.remove(storename) # clearing renamed file output by CL
                log.fullinfo('CL outputs '+cloutname+' was renamed on disk to:\n '+finalname,'postCL')
                log.fullinfo(finalname+' was loaded into memory', 'postCL')
                log.fullinfo(finalname+' was deleted from disk', 'postCL')
                log.fullinfo(storename+' was deleted from disk','postCL')
        
    def LogCurParams(self):
        log.fullinfo('\ncurrent general parameters:', 'params')
        for key in self.rc:
            val=self.rc[key]
            log.fullinfo(repr(key)+' = '+repr(val),'params')

        log.fullinfo('\ncurrent primitive specific parameters:', 'params')
        for key in self.rc.localparms:
            val=self.rc.localparms[key]
            log.fullinfo(repr(key)+' = '+repr(val),'params')
            
    def nbiascontam(self):
        '''This function will find the largest difference between the horizontal component 
           of every BIASSEC value and those of the biassec parameter and return that difference 
           as an integer to be the value for the nbiascontam parameter used in the gireduce call of the overscanSubtract prim'''
        
        retval=0
        for ad in self.rc.getInputs(style='AD'):
            biassec=self.rc['biassec']
            val = biassecStrTonbiascontam(biassec,ad)
            if val>retval:
                retval=val
        return retval

class IrafStdout():

    def __init__(self):
        pass
    
    def write(self, out):
        if "PANIC" in out or "ERROR" in out:
            log.error(out,'clError')
        elif len(out)>1:
            log.fullinfo(out,'clInfo')
        
    def flush(self):
        pass
 
        