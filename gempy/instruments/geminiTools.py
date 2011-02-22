# Author: Kyle Mede, May 2010
# This module provides many functions used by all primitives 

import os

import pyfits as pf
import numpy as np
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
import tempfile

def biassecStrTonbiascontam(biassec, ad):
    """ 
    This function works with nbiascontam() of the CLManager. 
    It will find the largest horizontal difference between biassec and 
    BIASSEC for each SCI extension in a single input.  This value will 
    be the new bias contamination value for use in IRAF scripts.
    
    @param biassec: biassec parameter of format '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
    @type biassec: string  
    
    @param ad: AstroData instance to calculate the bias contamination for
    @type ad: AstroData instance
    
    """
    log=gemLog.getGeminiLog() 
    try:
        # Split up the input triple list into three separate ones
        ccdStrList = biassec.split('],[')
        # Prepare the to-be lists of lists
        ccdIntList = []
        for string in ccdStrList:
            # Use secStrToIntList function to convert each string version
            # of the list into actual integer lists and load it into the lists
            # of lists
            ccdIntList.append(secStrToIntList(string))
        
        # Setting the return value to be updated in the loop below    
        retvalue=0
        for i in range(0, ad.countExts('SCI')):
            # Retrieving current BIASSEC value
            BIASSEC = ad.extGetKeyValue(('SCI',i+1),'BIASSEC')
            # Converting the retrieved string into a integer list
            BIASSEClist = secStrToIntList(BIASSEC)
            # Setting the lower case biassec list to the appropriate list in the 
            # lists of lists created above the loop
            biasseclist = ccdIntList[i]
            # Ensuring both biassec's have the same vertical coords
            if (biasseclist[2] == BIASSEClist[2]) and \
            (biasseclist[3] == BIASSEClist[3]):
                # If overscan/bias section is on the left side of the chip
                if biasseclist[0]<50: 
                    # Ensuring left X coord of both biassec's are equal
                    if biasseclist[0] == BIASSEClist[0]: 
                        # Set the number of contaminating columns to the 
                        # difference between the biassec's right X coords
                        nbiascontam = BIASSEClist[1]-biasseclist[1]
                    # If left X coords of biassec's don't match, set number of 
                    # contaminating columns to 4 and make a error log message
                    else:
                        log.error('left horizontal components of biassec and'+
                                  ' BIASSEC did not match, so using default'+
                                  ' nbiascontam=4')
                        nbiascontam = 4
                # If overscan/bias section is on the right side of chip
                else: 
                    if biasseclist[1] == BIASSEClist[1]: 
                        nbiascontam = BIASSEClist[0]-biasseclist[0]
                    else:
                        log.error('right horizontal components of biassec'+
                                  ' and BIASSEC did not match, so using '+
                                  'default nbiascontam=4') 
                        nbiascontam = 4
            # Overscan/bias section is not on left or right side of chip, so 
            # set to number of contaminated columns to 4 and log error message
            else:
                log.error('vertical components of biassec and BIASSEC '+
                          'parameters did not match, so using default '+
                          'nbiascontam=4')
                nbiascontam = 4
            # Find the largest nbiascontam value throughout all chips and 
            # set it as the value to be returned  
            if nbiascontam > retvalue:  
                retvalue = nbiascontam
            
        return retvalue
    
    # If all the above checks and attempts to calculate a new nbiascontam fail,
    # make a error log message and return the value 4. so exiting 'gracefully'.        
    except:
        log.error('An error occurred while trying to calculate the '+
                  'nbiascontam, so using default value = 4')
        return 4 

def fileNameUpdater(adIn=None, infilename='', postpend='', prepend='' , strip=False, verbose=1):
    """ This function is for updating the file names of astrodata objects.
        
        It can be used in a few different ways.  For simple post/prepending of
        the infilename string, there is no need to define adIn or strip. The 
        current filename for adIn will be used if infilename is not defined. 
        The examples below should make the main uses clear.
        
    Note: 1.if the input filename has a path, the returned value will have
          path stripped off of it.
          2. if strip is set to True, then adIn must be defined.
          
    @param adIn: input astrodata instance having its filename being updated
    @type adIn: astrodata object
    
    @param infilename: filename to be updated
    @type infilename: string
    
    @param postpend: string to put between end of current filename and the 
                    extension 
    @type postpend: string
    
    @param prepend: string to put at the beginning of a filename
    @type prepend: string
    
    @param strip: Boolean to signal that the original filename of the astrodata
                  object prior to processing should be used. adIn MUST be 
                  defined for this to work.
    @type strip: Boolean
    
    ex. fileNameUpdater(adIn=myAstrodataObject, postpend='_prepared', strip=True)
        result: 'N20020214S022_prepared.fits'
        
        fileNameUpdater(infilename='N20020214S022_prepared.fits', postpend='_biasCorrected')
        result: 'N20020214S022_prepared_biasCorrected.fits'
        
        fileNameUpdater(adIn=myAstrodataObject, prepend='testversion_')
        result: 'testversion_N20020214S022.fits'
    
    """
    log=gemLog.getGeminiLog(verbose=verbose) 

    # Check there is a name to update
    if infilename=='':
        # if both infilename and adIn are not passed in, then log critical msg
        if adIn==None:
            log.critical('A filename or an astrodata object must be passed '+
                         'into fileNameUpdater, so it has a name to update')
        # adIn was passed in, so set infilename to that ad's filename
        else:
            infilename = adIn.filename
            
    # Strip off any path that the input file name might have
    basefilename = os.path.basename(infilename)

    # Split up the filename and the file type ie. the extension
    (name,filetype) = os.path.splitext(basefilename)
    
    if strip:
        # Grabbing the value of PHU key 'ORIGNAME'
        phuOrigFilename = adIn.phuGetKeyValue('ORIGNAME') 
        # If key was 'None', ie. storeOriginalName() wasn't ran yet, then run it now
        if phuOrigFilename is None:
            # Storing the original name of this astrodata object in the PHU
            phuOrigFilename = adIn.storeOriginalName()
            
        # Split up the filename and the file type ie. the extension
        (name,filetype) = os.path.splitext(phuOrigFilename)
        
    # Create output filename
    outFileName = prepend+name+postpend+filetype
    return outFileName
    

def logDictParams(indict, verbose=1):
    """ A function to log the parameters in a provided dictionary.  Main use
    is to log the values in the dictionaries of parameters for function 
    calls using the ** method.
    """
    log=gemLog.getGeminiLog(verbose=verbose)
    for key in indict:
        log.fullinfo(repr(key)+' = '+repr(indict[key]), 
                     category='parameters')
def observationMode(ad):
    """ 
    A basic function to determine if the input is one of 
    (IMAGE|IFU|MOS|LONGSLIT) type.  It returns the type as a string. If input is  
    none of these types, then None is returned.
    """
    types = ad.getTypes()
    if 'IMAGE' in types:
        return 'IMAGE'
    elif 'IFU' in types:
        return 'IFU'
    elif 'MOS' in types:
        return 'MOS'
    elif 'LONGSLIT' in types:
        return 'LONGSLIT'
    else:
        return None
    
def pyrafBoolean(pythonBool):
    """
    A very basic function to reduce code repetition that simply 'casts' any 
    given Python boolean into a pyraf/IRAF one for use in the CL scripts.
    
    """
    log=gemLog.getGeminiLog() 
    import pyraf
    
    # If a boolean was passed in, convert it
    if pythonBool:
        return pyraf.iraf.yes
    elif  not pythonBool:
        return pyraf.iraf.no
    else:
        log.critical('DANGER DANGER Will Robinson, pythonBool passed in was '+
        'not True or False, and thats just crazy talk :P')

def secStrToIntList(string):
    """ A function to convert a string representing a list of integers to 
        an actual list of integers.
        
        @param string: string to be converted
        @type string: string of format '[#1:#2,#3:#4]'
        
        returns list of ints [#1,#2,#3,#4]
    
    """
    # Strip off the brackets and then split up into a string list 
    # using the ',' delimiter
    coords = string.strip('[').strip(']').split(',')
    # Split up strings into X and Y components using ':' delimiter
    Ys = coords[0].split(':')
    Xs = coords[1].split(':')
    # Prepare the list and then fill it with the string coordinates 
    # converted to integers
    retl = []
    retl.append(int(Ys[0]))
    retl.append(int(Ys[1]))
    retl.append(int(Xs[0]))
    retl.append(int(Xs[1]))
    return retl

def stdObsHdrs(ad, verbose=1):
    """ This function is used by standardizeHeaders in primitives_GEMINI.
        
        It will update the PHU header keys NSCIEXT, PIXSCALE
        NEXTEND, OBSMODE, COADDEXP, EXPTIME and NCOADD plus it will add 
        a time stamp for GPREPARE to indicate that the file has be prepared.
        
        In the SCI extensions the header keys GAIN, PIXSCALE, RDNOISE, BUNIT,
        NONLINEA, SATLEVEL and EXPTIME will be updated.
        
        @param ad: astrodata instance to perform header key updates on
        @type ad: an AstroData instance
    
    """
    log=gemLog.getGeminiLog(verbose=verbose) 
    # Keywords that are updated/added for all Gemini PHUs 
    ad.phuSetKeyValue('NSCIEXT', ad.countExts('SCI'), 
                      'Number of science extensions')
    ad.phuSetKeyValue('PIXSCALE', ad.pixel_scale(), 
                      'Pixel scale in Y in arcsec/pixel')
    ad.phuSetKeyValue('NEXTEND', len(ad) , 'Number of extensions')
    ad.phuSetKeyValue('OBSMODE', observationMode(ad) , 
                      'Observing mode (IMAGE|IFU|MOS|LONGSLIT)')
    ad.phuSetKeyValue('COADDEXP', ad.phuValue('EXPTIME') , 
                      'Exposure time for each coadd frame')
    # Retrieving the number of coadds using the coadds descriptor 
    numcoadds = ad.coadds()
    # If the value the coadds descriptor returned was None (or zero) set to 1
    if not numcoadds:  
        numcoadds = 1      
    # Calculate the effective exposure time  
    # = (current EXPTIME value) X (# of coadds)
    effExpTime = ad.phuValue('EXPTIME')*numcoadds  
    # Set the effective exposure time and number of coadds in the header  
    ad.phuSetKeyValue('EXPTIME', effExpTime , 'Effective exposure time') 
    ad.phuSetKeyValue('NCOADD', str(numcoadds) , 'Number of coadds')
    
    # Adding the current filename (without directory info) to ORIGNAME in PHU
    origName = ad.storeOriginalName()
    
    # Adding/updating the GEM-TLM (automatic) and GPREPARE time stamps
    ut = ad.historyMark(key='GPREPARE',stomp=False) 
       
    # Updating logger with updated/added keywords
    log.fullinfo('****************************************************', 
                 category='header')
    log.fullinfo('file = '+ad.filename, category='header')
    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', 
                 category='header')
    log.fullinfo('PHU keywords updated/added:\n', category='header')
    log.fullinfo('NSCIEXT = '+str(ad.phuGetKeyValue('NSCIEXT')), category='header' )
    log.fullinfo('PIXSCALE = '+str(ad.phuGetKeyValue('PIXSCALE')), category='header' )
    log.fullinfo('NEXTEND = '+str(ad.phuGetKeyValue('NEXTEND')), category='header' )
    log.fullinfo('OBSMODE = '+ad.phuGetKeyValue('OBSMODE'), category='header' )
    log.fullinfo('COADDEXP = '+str(ad.phuGetKeyValue('COADDEXP')), category='header' )
    log.fullinfo('EXPTIME = '+str(ad.phuGetKeyValue('EXPTIME')), category='header' )
    log.fullinfo('ORIGNAME = '+ad.phuGetKeyValue('ORIGNAME'), category='header')
    log.fullinfo('GEM-TLM = '+str(ad.phuGetKeyValue('GEM-TLM')), category='header' )
    log.fullinfo('---------------------------------------------------', 
                 category='header')
         
    # A loop to add the missing/needed keywords in the SCI extensions
    for ext in ad['SCI']:
        ext.header.update('GAIN', ext.gain(), 'Gain (e-/ADU)')
        ext.header.update('PIXSCALE', ext.pixel_scale(), 
                           'Pixel scale in Y in arcsec/pixel')
        ext.header.update('RDNOISE', ext.read_noise(), 'readout noise in e-')
        ext.header.update('BUNIT','adu', 'Physical units')
        
        # Retrieving the value for the non-linear value of the pixels using the
        # non_linear_level descriptor, if it returns nothing, 
        # set it to the string None.
        nonlin = ext.non_linear_level()
        if not nonlin:
            nonlin = 'None'     
        ext.header.update( 'NONLINEA', nonlin, 'Non-linear regime level in ADU')
        ext.header.update( 'SATLEVEL', 
                           ext.saturation_level(), 'Saturation level in ADU')
        ext.header.update( 'EXPTIME', effExpTime, 'Effective exposure time')
        
        log.fullinfo('SCI extension number '+str(ext.extver())+
                     ' keywords updated/added:\n', category='header')
        log.fullinfo('GAIN = '+str(ext.gain()), category='header' )
        log.fullinfo('PIXSCALE = '+str(ext.pixel_scale()), category='header')
        log.fullinfo('RDNOISE = '+str(ext.read_noise()), category='header')
        log.fullinfo('BUNIT = '+'adu', category='header' )
        log.fullinfo('NONLINEA = '+str(nonlin), category='header' )
        log.fullinfo('SATLEVEL = '+str(ext.saturation_level()),
                     category='header')
        log.fullinfo('EXPTIME = '+str(effExpTime), category='header' )
        log.fullinfo('---------------------------------------------------', 
                     category='header')

def stdObsStruct(ad, verbose=1):
    """ This function is used by standardizeStructure in primitives_GEMINI.
    
        It currently checks that the SCI extensions header key EXTNAME = 'SCI' 
        and EXTVER matches that of descriptor values 
        
        @param ad: astrodata instance to perform header key updates on
        @type ad: an AstroData instance
    
    """
    log=gemLog.getGeminiLog(verbose=verbose)    
    # Formatting so logger looks organized for these messages
    log.fullinfo('****************************************************', 
                 category='header') 
    log.fullinfo('file = '+ad.filename, category='header')
    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', 
                 category='header')
    # A loop to add the missing/needed keywords in the SCI extensions
    for ext in ad['SCI']:
        # Setting EXTNAME = 'SCI' and EXTVER = descriptor value
        ext.header.update( 'EXTNAME', 'SCI', 'Extension name')        
        ext.header.update( 'EXTVER', ext.extver(), 'Extension version') 
        # Updating logger with new header key values
        log.fullinfo('SCI extension number '+str(ext.header['EXTVER'])+
                     ' keywords updated/added:\n', category='header')       
        log.fullinfo('EXTNAME = '+'SCI', category='header' )
        log.fullinfo('EXTVER = '+str(ext.header['EXTVER']), category='header' )
        log.fullinfo('---------------------------------------------------', 
                     category='header') 

class CLManager(object):
    """This is a class that will take care of all the preparation and wrap-up 
        tasks needed when writing a primitive that wraps a IRAF CL routine.
        
    """
    # The version of the names for input to the CL script
    _preCLcachestorenames = [] 
    # The original names of the files at the start of the 
    # primitive which called CLManager
    _preCLfilenames = [] 
    # Preparing other 'global' objects to be accessed throughout this class
    prefix = None
    postpend = None
    listname = None
    templog = None
    outNames = None
    funcName = None
    adOuts = None
    status = None
    log=None
    verbose=1
     
    def __init__(self, adIns=None, outNames=None, postpend=None, funcName=None,
                 logName=None, verbose=1, noLogFile=None):
        """This instantiates all the globally accessible variables and prepares
           the inputs for use in CL scripts by temporarily writing them to 
           disk with temp names.
        """
        if isinstance(adIns,list):
            self.inputs = adIns
        else:
            self.inputs = [adIns]
        # Check that the inputs have been prepared, if not then CL scripts might
        # not work correctly.
        self.status = True
        for ad in self.inputs:
            if (ad.phuGetKeyValue('GPREPARE')==None) and (ad.phuGetKeyValue('PREPARED')==None):
                self.status = False
        # All inputs prepared, then continue, else the False status will trigger
        # the caller to not proceed further.
        if self.status:
            # Get the REAL log file object
            self.log = gemLog.getGeminiLog(logName=logName, verbose=verbose, 
                                           noLogFile=noLogFile)
            self.verbose=verbose
            self.postpend = postpend
            self._preCLcachestorenames = []
            self._preCLfilenames = []
            self.outNames = outNames
            self.funcName = funcName
            self.adOuts = []
            self.prefix = self.uniquePrefix()
            self.preCLwrites()
            # Create a temporary log file object
            self.templog = tempfile.NamedTemporaryFile() 
            
    def outNamesMaker(self):
        """ A function to apply the provided postpend value to the end of 
            each input filename and add to a list for use to name the output
            files with.
        """
        if isinstance(self.inputs,list):
            outNames=[]
            for ad in self.inputs:
                outNames.append(fileNameUpdater(adIn=ad, postpend=self.postpend, 
                                                                strip=False, verbose= self.verbose))
        else:
            outNames = [fileNameUpdater(adIn=ad, postpend=self.postpend, strip=False, verbose= self.verbose)]
        return outNames
    
    def cacheStoreNames(self):
        """ Just a function to return the 'private' member variable 
         _preCLcachestorenames.
        """
        return self._preCLcachestorenames
        
    def combineOutname(self):
        """ This creates the output name for combine type IRAF tasks to write 
            the combined output file to.  Uses the postpend value and
        
        """
        #@@ REFERENCE IMAGE: for output name
        return self.postpend+self._preCLcachestorenames[0]
         
    def finishCL(self, combine=False): 
        """ Performs all the finalizing steps after CL script is ran. 
         This is currently just an alias for postCLloads and might have 
         more functionality in the future.
         
         """    
        self.postCLloads(combine)
        return self.adOuts  
          
    def inputsAsStr(self):
        """ This returns the list of temporary file names written to disk for 
            the input files in the form of a list joined by commas for passing
            into IRAF.
        
        """
        return ','.join(self._preCLcachestorenames)
    
    def inputList(self):
        """ This creates a list file of the inputs for use in combine type
        primitives.
        """
        try:
            filename = 'List'+str(os.getpid())+self.funcName
            if os.path.exists(filename):
                return filename
            else:
                fh = open(filename, 'w')
                for item in self._preCLcachestorenames:
                    fh.writelines(item + '\n')                    
                fh.close()
                self.listname = filename
                return "@" + self.listname
        except:
            raise "Could not write inlist file for stacking." 
           
    def logfile(self):
        """ A function to return the name of the unique temporary log file to 
            be used by IRAF.
        
        """
        return self.templog.name
        
    def nbiascontam(self, biassec=None):
        """This function will find the largest difference between the horizontal 
        component of every BIASSEC value and those of the biassec parameter. 
        The returned value will be that difference as an integer and it will be
        used as the value for the nbiascontam parameter used in the gireduce 
        call of the overscanSubtract primitive.
        
        """
        
        # Prepare a stored value to be compared between the inputs
        retval=0
        # Loop through the inputs
        for ad in self.inputs:
            # Pass the retrieved value to biassecStrToBiasContam function
            # to do the work in finding the difference of the biassec's
            val = biassecStrTonbiascontam(biassec, ad)
            # Check if value returned for this input is larger. Keep the largest
            if val > retval:
                retval = val
        return retval
    
    def preCLNames(self):
        """Just a function to return the 'private' member 
            variable _preCLfilenames.
        
        """
        return self._preCLfilenames
   
    def preCLwrites(self):
        """ The function that writes the files in memory to disk with temporary 
            names and saves the original names in a list.
        
        """
        if self.outNames is None:
            # load up a outNames list for use in postCLloads
            self.outNames = self.outNamesMaker()
        
        for ad in self.inputs:            
            # Load up the preCLfilenames list with the input's filename
            self._preCLfilenames.append(ad.filename)
            # Strip off all postfixes and prepend filename with a unique prefix
            name = fileNameUpdater(adIn=ad, prepend=self.prefix, strip=True, verbose= self.verbose)
            # store the unique name in preCLcachestorenames for later reference
            self._preCLcachestorenames.append(name)
            # Log the name of this temporary file being written to disk
            self.log.fullinfo('Temporary file on disk for input to CL: '+name)
            # Write this file to disk with its unique filename 
            ad.write(name, rename=False)
                     
    def postCLloads(self,combine=False):
        """  This function takes care of loading the output files the IRAF
            routine wrote to disk back into memory with the appropriate name.  
            Then it will delete all the temporary files created by the 
            CLManager.
        
        """
        # Do the appropriate wrapping up for combine type primitives
        if combine is True:
            # If a combine type CL function is called, the output will be a 
            # single file and thus no looping is required here.
            
            # The name that IRAF wrote the output to
            cloutname = self.postpend+self._preCLcachestorenames[0]
            # The name we want the file to be
            outName = self.outNames[0] 
            
            # Renaming the IRAF written file to the name we want
            os.rename(cloutname, outName )
            # Reporting the renamed file to the reduction context and thus
            # bringing it into memory
            self.adOuts.append(AstroData(outName))
            
            # Deleting the renamed file from disk
            os.remove(outName)
            # Removing the list file of the inputs 
            os.remove(self.listname)
            # Close, and thus delete, the temporary log object needed by IRAF
            self.templog.close() 
            # Logging files that were affected during wrap-up
            self.log.fullinfo('CL outputs '+cloutname+' was renamed on disk to:\n'+
                         outName)
            self.log.fullinfo(outName+' was loaded into memory')
            self.log.fullinfo(outName+' was deleted from disk')
            self.log.fullinfo('Temporary list '+self.listname+' was deleted from disk')
            self.log.fullinfo('Temporary log '+self.templog.name+' was deleted from disk')
            # Removing the temporary files on disk that were inputs to IRAF
            for i in range(0, len(self._preCLcachestorenames)):
                # Name of file written to disk for input to CL script
                storename = self._preCLcachestorenames[i]  
                # Clearing renamed file ouput by CL
                os.remove(storename) 
                self.log.fullinfo(storename+' was deleted from disk')
                
        # Do the appropriate wrapping up for non-combine type primitives        
        elif combine is False:
            for i in range(0, len(self._preCLcachestorenames)):
                # Name of file written to disk for input to CL script
                storename = self._preCLcachestorenames[i]  
                # Name of file CL wrote to disk
                cloutname = self.postpend + storename 
                
                # Name I want the file to be dictated by outNames
                outName = self.outNames[i]
                
                # Renaming the IRAF written file to the name we want
                os.rename(cloutname, self.outNames[i] )
                
                # Reporting the renamed file to the reduction context and thus
                # bringing it into memory
                self.adOuts.append(AstroData(outName))
                # Clearing file written for CL input
                os.remove(outName) 
                # clearing renamed file output by CL
                os.remove(storename) 
                # Close, and thus delete, the temporary log needed by IRAF
                self.templog.close()
                # Logging files that were affected during wrap-up
                self.log.fullinfo('CL outputs '+cloutname+
                             ' was renamed on disk to:\n '+outName)
                self.log.fullinfo(outName+' was loaded into memory')
                self.log.fullinfo(outName+' was deleted from disk')
                self.log.fullinfo(storename+' was deleted from disk')
         
    def rmStackFiles(self):
        """A function to remove the filenames written to disk by 
            setStackable.
        
        """
        for file in self._preCLfilenames:
            self.log.fullinfo('Removing file '+file+' from disk')
            os.remove(file)
            
    def uniquePrefix(self):
        """ uses the primitive name and the process ID to create a unique
            prefix for the files being temporarily written to disk.
        
        """
        return 'tmp'+ str(os.getpid())+self.funcName
    
class IrafStdout():
    """  This is a class to act as the standard output for the IRAF 
        routines that instead of printing its messages to the screen,
        it will print them to the gemlog.py logger that the primitives use
        
    """
    log=None
    
    def __init__(self, verbose=1):
        """ A function that is needed IRAF but not used in our wrapping its
        scripts"""
        self.log = gemLog.getGeminiLog(verbose=verbose)
    
    def write(self, out):
        """ This function converts the IRAF console prints to logger calls.
            If the print has 'PANIC' in it, then it becomes a error log message,
            else it becomes a fullinfo message.
            
        """
        if 'PANIC' in out or 'ERROR' in out:
            self.log.error(out, category='clError')
        elif len(out) > 1:
            self.log.fullinfo(out, category='clInfo')
        
    def flush(self):
        """ A function that is needed IRAF but not used in our wrapping its
        scripts"""
        pass
 
