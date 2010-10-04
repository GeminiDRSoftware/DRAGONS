# Author: Kyle Mede, May 2010
# This module provides many functions used by all primitives 

import os

import pyfits as pf
import numpy as np
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
import tempfile

log=gemLog.getGeminiLog() 

def stdObsHdrs(ad):
    """ This function is used by standardizeHeaders in primitives_GEMINI.
        
        It will update the PHU header keys NSCIEXT, PIXSCALE
        NEXTEND, OBSMODE, COADDEXP, EXPTIME and NCOADD plus it will add 
        a time stamp for GPREPARE to indicate that the file has be prepared.
        
        In the SCI extensions the header keys GAIN, PIXSCALE, RDNOISE, BUNIT,
        NONLINEA, SATLEVEL and EXPTIME will be updated.
        
        @param ad: astrodata instance to perform header key updates on
        @type ad: an AstroData instance
        """
    # Keywords that are updated/added for all Gemini PHUs 
    ad.phuSetKeyValue('NSCIEXT', ad.countExts('SCI'), \
                      'Number of science extensions')
    ad.phuSetKeyValue('PIXSCALE', ad.pixel_scale(), \
                      'Pixel scale in Y in arcsec/pixel')
    ad.phuSetKeyValue('NEXTEND', len(ad) , 'Number of extensions')
    ad.phuSetKeyValue('OBSMODE', ad.observation_mode() , \
                      'Observing mode (IMAGE|IFU|MOS|LONGSLIT)')
    ad.phuSetKeyValue('COADDEXP', ad.phuValue('EXPTIME') , \
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
    
    # Adding/updating the GEM-TLM (automatic) and GPREPARE time stamps
    ad.historyMark(key='GPREPARE',stomp=False) 
       
    # Updating logger with updated/added keywords
    log.fullinfo('****************************************************', \
                 'header')
    log.fullinfo('file = '+ad.filename,'header')
    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', \
                 'header')
    log.fullinfo('PHU keywords updated/added:\n', 'header')
    log.fullinfo('NSCIEXT = '+str(ad.countExts('SCI')),'header' )
    log.fullinfo('PIXSCALE = '+str(ad.pixel_scale()),'header' )
    log.fullinfo('NEXTEND = '+str(len(ad)),'header' )
    log.fullinfo('OBSMODE = '+str(ad.observation_mode()),'header' )
    log.fullinfo('COADDEXP = '+str(ad.phuValue('EXPTIME')),'header' )
    log.fullinfo('EXPTIME = '+str(effExpTime),'header' )
    log.fullinfo('GEM-TLM = '+str(ut),'header' )
    log.fullinfo('---------------------------------------------------','header')
         
    # A loop to add the missing/needed keywords in the SCI extensions
    for ext in ad['SCI']:
        ext.SetKeyValue('GAIN', ext.gain(), \
                           'Gain (e-/ADU)')
        ext.SetKeyValue('PIXSCALE', ext.pixel_scale(), \
                           'Pixel scale in Y in arcsec/pixel')
        ext.SetKeyValue('RDNOISE', ext.read_noise() , \
                           'readout noise in e-')
        ext.SetKeyValue('BUNIT','adu' , \
                           'Physical units')
        
        # Retrieving the value for the non-linear value of the pixels using the
        # non_linear_level descriptor, if it returns nothing, 
        # set it to the string None.
        nonlin = ext.non_linear_level()
        if not nonlin:
            nonlin = 'None'     
        ext.SetKeyValue( 'NONLINEA', nonlin , 'Non-linear regime level in ADU')
        ext.SetKeyValue( 'SATLEVEL', \
                           ext.saturation_level(), 'Saturation level in ADU')
        ext.SetKeyValue( 'EXPTIME', effExpTime , 'Effective exposure time')
        
        log.fullinfo('SCI extension number '+str(ext.extver())+\
                     ' keywords updated/added:\n', 'header')
        log.fullinfo('GAIN = '+str(ext.gain()), 'header' )
        log.fullinfo('PIXSCALE = '+str(ext.pixel_scale()), 'header' )
        log.fullinfo('RDNOISE = '+str(ext.read_noise()), 'header' )
        log.fullinfo('BUNIT = '+'adu', 'header' )
        log.fullinfo('NONLINEA = '+str(nonlin), 'header' )
        log.fullinfo('SATLEVEL = '+str(ext.saturation_level()),'header' )
        log.fullinfo('EXPTIME = '+str(effExpTime), 'header' )
        log.fullinfo('---------------------------------------------------', \
                     'header')

def stdObsStruct(ad):
    """ This function is used by standardizeStructure in primitives_GEMINI.
    
        It currently checks that the SCI extensions header key EXTNAME = 'SCI' 
        and EXTVER matches that of descriptor values 
        
        @param ad: astrodata instance to perform header key updates on
        @type ad: an AstroData instance
        """
        
    # Formatting so logger looks organized for these messages
    log.fullinfo('****************************************************', \
                 'header') 
    log.fullinfo('file = '+ad.filename, 'header')
    log.fullinfo('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', \
                 'header')
    # A loop to add the missing/needed keywords in the SCI extensions
    for ext in ad['SCI']:
        # Setting EXTNAME = 'SCI' and EXTVER = descriptor value
        ext.SetKeyValue( 'EXTNAME', 'SCI', 'Extension name')        
        ext.SetKeyValue( 'EXTVER', ext.extver(), 'Extension version') 
        # Updating logger with new header key values
        log.fullinfo('SCI extension number '+str(ext.header['EXTVER'])+\
                     ' keywords updated/added:\n', 'header')       
        log.fullinfo('EXTNAME = '+'SCI', 'header' )
        log.fullinfo('EXTVER = '+str(ext.header['EXTVER']), 'header' )
        log.fullinfo('---------------------------------------------------', \
                     'header')
        
def fileNameUpdater(origfilename, postpend='', prepend='' , strip=False):
    """ This function is for updating the file names.
    
    Note: if the input filename has a path, the returned value will have
          path stripped off of it.
    
    @param postpend: string to put between end of current filename and the 
                    extension 
    @type postpend: string
    
    @param prepend: string to put at the beginning of a filename
    @type prepend: string
    
    @param strip: Boolean to signal if the original postpends should be 
                  removed prior to adding the new one (if one exists).
    @type strip: Boolean
    
    ex. fileNameUpdater('N20020214S022_prepared_vardq_oversubed_overtrimd.fits',
                        postpend='_prepared', strip=True)
        result: 'N20020214S022_prepared.fits'
                    """
    # Strip off any path that the input file name might have
    infilename = os.path.basename(origfilename)
    
    # If a value for postpend was passed in
    if postpend != '':
        # if stripping was requested, then do so using stripPostfix function
        if strip:
            infilename = stripPostfix(infilename)
        # Split up the filename and the file type ie. the extension
        (name,filetype) = os.path.splitext(infilename)
        # Create output filename
        outFileName = name+postpend+filetype
        
    elif prepend !='':
        if strip:
            infilename = stripPostfix(infilename)
        (name,filetype) = os.path.splitext(infilename)
        outFileName = prepend+name+filetype
        
    return outFileName

def stripPostfix(filename):
    """ This function is used by fileNameUpdater to strip all the original
        postfixes of a input string, separated from the base filename by
        '_'. """
    # Saving the path of the input file
    dirname = os.path.dirname(filename)
    # Saving the filename without its path
    basename = os.path.basename(filename)
    # Split up the filename and the file type ie. the extension 
    (name, filetype) = os.path.splitext(basename)
    # Splitting up file name into a list by the '_' delimiter 
    a = name.split('_')
    # The file name without the postfixes is the first element of the list
    name = a[0]
    # Re-attaching the path and file type to the cleaned file name 
    retname = os.path.join(dirname, name+filetype)
    return retname    

def secStrToIntList(string):
    """ A function to convert a string representing a list of integers to 
        an actual list of integers.
        
        @param string: string to be converted
        @type string: string of format '[#1:#2,#3:#4]'
        
        returns list of ints [#1,#2,#3,#4]"""
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
                        log.error('left horizontal components of biassec and'+\
                                  ' BIASSEC did not match, so using default'+\
                                  ' nbiascontam=4')
                        nbiascontam = 4
                # If overscan/bias section is on the right side of chip
                else: 
                    if biasseclist[1] == BIASSEClist[1]: 
                        nbiascontam = BIASSEClist[0]-biasseclist[0]
                    else:
                        log.error('right horizontal components of biassec'+\
                                  ' and BIASSEC did not match, so using '+\
                                  'default nbiascontam=4') 
                        nbiascontam = 4
            # Overscan/bias section is not on left or right side of chip, so 
            # set to number of contaminated columns to 4 and log error message
            else:
                log.error('vertical components of biassec and BIASSEC '+\
                          'parameters did not match, so using default '+\
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
        log.error('An error occurred while trying to calculate the '+\
                  'nbiascontam, so using default value = 4')
        return 4 
        
        
def pyrafBoolean(pythonBool):
    """
    A very basic function to reduce code repetition that simply 'casts' any 
    given Python boolean into a pyraf/IRAF one for use in the CL scripts.
    """
    import pyraf
    
    # If a boolean was passed in, convert it
    if pythonBool:
        return pyraf.iraf.yes
    elif  not pythonBool:
        return pyraf.iraf.no
    else:
        log.critical('DANGER DANGER Will Robinson, pythonBool passed in was '+\
        'not True or False, and thats just crazy talk :P')

class CLManager(object):
    """This is a class that will take care of all the preparation and wrap-up 
        tasks needed when writing a primitive that wraps a IRAF CL routine.
    """
    # The version of the names for input to the CL script
    _preCLcachestorenames = [] 
    # The original names of the files at the start of the 
    # primitive which called CLManager
    _preCLfilenames = [] 
    # Preparing 'global' objects to be accessed throughout this class
    rc = None
    prefix = None
    outpref = None
    listname = None
    templog = None
    
     
    def __init__(self, rc, outpref = None):
        """This instantiates all the globally accessible variables"""
        self.rc  = rc
        if outpref is None:
            # If no outpref passed in retrieve the outpref value from the 
            # local parameters in the reduction context
            outpref = rc['outpref']
        self.outpref = outpref
        self._preCLcachestorenames = []
        self._preCLfilenames = []
        self.prefix = self.uniquePrefix()
        self.preCLwrites()
        # Create a temporary log file object
        self.templog = tempfile.NamedTemporaryFile() 
    
    def finishCL(self, combine=False): 
        """ Performs all the finalizing steps after CL script is ran. 
         This is currently just an alias for postCLloads and might have 
         more functionality in the future."""
        self.postCLloads(combine)    
    
    def preCLwrites(self):
        """ The function that writes the files in memory to disk with temporary 
            names and saves the original names in a list."""
        for ad in self.rc.getInputs(style='AD'):
            # Load up the preCLfilenames list with the input's filename
            self._preCLfilenames.append(ad.filename)
            # Strip off all postfixes and prepend filename with a unique prefix
            name = fileNameUpdater(ad.filename, prepend=self.prefix, strip=True)
            # store the unique name in preCLcachestorenames for later reference
            self._preCLcachestorenames.append(name)
            # Log the name of this temporary file being written to disk
            log.fullinfo('Temporary file on disk for input to CL: '+name)
            # Write this file to disk with its unique filename 
            ad.write(name, rename=False) 
    
    def cacheStoreNames(self):
        """ Just a function to return the 'private' member variable 
         _preCLcachestorenames."""
        return self._preCLcachestorenames
        
    def rmStackFiles(self):
        """A function to remove the filenames written to disk by 
            setStackable."""
        for file in self._preCLfilenames:
            log.fullinfo('Removing file '+file+' from disk')
            os.remove(file)
        
    def preCLNames(self):
        """Just a function to return the 'private' member 
            variable _preCLfilenames."""
        return self._preCLfilenames
    
    def logfile(self):
        """ A function to return the name of the unique temporary log file to 
            be used by IRAF."""
        return self.templog.name
    
    def inputsAsStr(self):
        """ This returns the list of temporary file names written to disk for 
            the input files in the form of a list joined by commas for passing
            into IRAF."""
        return ','.join(self._preCLcachestorenames)
    
    def inputList(self):
        """ This creates a list file of the inputs for use in combine type
        primitives."""
        # Create a unique name for the list file
        self.listname = 'List'+str(os.getpid())+self.rc.ro.curPrimName
        return self.rc.makeInlistFile(self.listname, self._preCLcachestorenames)
        
    def uniquePrefix(self):
        """ uses the primitive name and the process ID to create a unique
            prefix for the files being temporarily written to disk."""
        return 'tmp'+ str(os.getpid())+self.rc.ro.curPrimName
    
    def combineOutname(self):
        """ This creates the output name for combine type IRAF tasks to write 
            the combined output file to"""
        #@@ REFERENCE IMAGE: for output name
        return self.outpref+self._preCLcachestorenames[0]
    
    def postCLloads(self,combine=False):
        """  This function takes care of loading the output files the IRAF
            routine wrote to disk back into memory with the appropriate name.  
            Then it will delete all the temporary files created by the 
            CLManager."""
        # Do the appropriate wrapping up for combine type primitives
        if combine is True:
            # The name that IRAF wrote the output to
            cloutname = self.outpref+self._preCLcachestorenames[0]
            # The name we want the file to be
            finalname = fileNameUpdater(self._preCLfilenames[0], \
                                      postpend= self.outpref, strip=False)
            # Renaming the IRAF written file to the name we want
            os.rename(cloutname, finalname )
            # Reporting the renamed file to the reduction context and thus
            # bringing it into memory
            self.rc.reportOutput(finalname)
            # Deleting the renamed file from disk
            os.remove(finalname)
            # Removing the list file of the inputs 
            os.remove(self.listname)
            # Close, and thus delete, the temporary log object needed by IRAF
            self.templog.close() 
            # Logging files that were affected during wrap-up
            log.fullinfo('CL outputs '+cloutname+' was renamed on disk to:\n'+\
                         finalname)
            log.fullinfo(finalname+' was loaded into memory')
            log.fullinfo(finalname+' was deleted from disk')
            log.fullinfo(self.listname+' was deleted from disk')
            log.fullinfo(self.templog.name+' was deleted from disk')
            # Removing the temporary files on disk that were inputs to IRAF
            for i in range(0, len(self._preCLcachestorenames)):
                # Name of file written to disk for input to CL script
                storename = self._preCLcachestorenames[i]  
                # Clearing renamed file ouput by CL
                os.remove(storename) 
                log.fullinfo(storename+' was deleted from disk')
                
        # Do the appropriate wrapping up for non-combine type primitives        
        elif combine is False:
            for i in range(0, len(self._preCLcachestorenames)):
                # Name of file written to disk for input to CL script
                storename = self._preCLcachestorenames[i]  
                # Name of file CL wrote to disk
                cloutname = self.outpref + storename  
                # Name I want the file to be
                finalname = fileNameUpdater(self._preCLfilenames[i], \
                                            postpend= self.outpref, strip=False)  
                # Renaming the IRAF written file to the name we want
                os.rename(cloutname, finalname )
                
                # Reporting the renamed file to the reduction context and thus
                # bringing it into memory
                self.rc.reportOutput(finalname)
                # Clearing file written for CL input
                os.remove(finalname) 
                # clearing renamed file output by CL
                os.remove(storename) 
                # Close, and thus delete, the temporary log needed by IRAF
                self.templog.close()
                # Logging files that were affected during wrap-up
                log.fullinfo('CL outputs '+cloutname+\
                             ' was renamed on disk to:\n '+finalname)
                log.fullinfo(finalname+' was loaded into memory')
                log.fullinfo(finalname+' was deleted from disk')
                log.fullinfo(storename+' was deleted from disk')
        
    def LogCurParams(self):
        """ A function to log the parameters in the local parameters file 
            and then global ones in the reduction context"""
        log.fullinfo('\ncurrent general parameters:', 'parameters')
        # Loop through the parameters in the general dictionary
        # of the reduction context and log them
        for key in self.rc:
            val = self.rc[key]
            log.fullinfo(repr(key)+' = '+repr(val), 'parameters')

        log.fullinfo('\ncurrent primitive specific parameters:', 'parameters')
        # Loop through the parameters in the local dictionary for the primitive
        # the CLManager was called from of the reduction context and log them
        for key in self.rc.localparms:
            val = self.rc.localparms[key]
            log.fullinfo(repr(key)+' = '+repr(val), 'parameters')
            
    def nbiascontam(self):
        """This function will find the largest difference between the horizontal 
        component of every BIASSEC value and those of the biassec parameter. 
        The returned value will be that difference as an integer and it will be
        used as the value for the nbiascontam parameter used in the gireduce 
        call of the overscanSubtract primitive."""
        
        # Prepare a stored value to be compared between the inputs
        retval=0
        # Loop through the inputs
        for ad in self.rc.getInputs(style='AD'):
            # Retrieve the biassec value in the parameters file
            biassec = self.rc['biassec']
            # Pass the retrieved value to biassecStrToBiasContam function
            # to do the work in finding the difference of the biassec's
            val = biassecStrTonbiascontam(biassec, ad)
            # Check if value returned for this input is larger. Keep the largest
            if val > retval:
                retval = val
        return retval

class IrafStdout():
    """  This is a class to act as the standard output for the IRAF 
        routines that instead of printing its messages to the screen,
        it will print them to the gemlog.py logger that the primitives use"""
    def __init__(self):
        """ A function that is needed IRAF but not used in our wrapping its
        scripts"""
        pass
    
    def write(self, out):
        """ This function converts the IRAF console prints to logger calls.
            If the print has 'PANIC' in it, then it becomes a error log message,
            else it becomes a fullinfo message."""
        if 'PANIC' in out or 'ERROR' in out:
            log.error(out, 'clError')
        elif len(out) > 1:
            log.fullinfo(out, 'clInfo')
        
    def flush(self):
        """ A function that is needed IRAF but not used in our wrapping its
        scripts"""
        pass
 
        