# Author: Kyle Mede, May 2010
# This module provides many functions used by all primitives 

import os

import pyfits as pf
import numpy as np
import tempfile
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
from astrodata.Errors import ToolboxError

def biassecStrTonbiascontam(ad, biassec):
    """ 
    This function works with nbiascontam() of the CLManager. 
    It will find the largest horizontal difference between biassec and 
    BIASSEC for each SCI extension in a single input.  This value will 
    be the new bias contamination value for use in IRAF scripts.
    
    :param ad: AstroData instance to calculate the bias contamination for
    :type ad: AstroData instance
    
    :param biassec: biassec parameter of format '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
    :type biassec: string  
    
    """
    log = gemLog.getGeminiLog() 
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

def fileNameUpdater(adIn=None, infilename='', suffix='', prefix='', strip=False):
    """
    This function is for updating the file names of astrodata objects.
    It can be used in a few different ways.  For simple post/pre pending of
    the infilename string, there is no need to define adIn or strip. The 
    current filename for adIn will be used if infilename is not defined. 
    The examples below should make the main uses clear.
        
    Note: 
    1.if the input filename has a path, the returned value will have
    path stripped off of it.
    2. if strip is set to True, then adIn must be defined.
          
    :param adIn: input astrodata instance having its filename being updated
    :type adIn: astrodata object
    
    :param infilename: filename to be updated
    :type infilename: string
    
    :param suffix: string to put between end of current filename and the 
                   extension 
    :type suffix: string
    
    :param prefix: string to put at the beginning of a filename
    :type prefix: string
    
    :param strip: Boolean to signal that the original filename of the astrodata
                  object prior to processing should be used. adIn MUST be 
                  defined for this to work.
    :type strip: Boolean
    
    ::
    
     fileNameUpdater(adIn=myAstrodataObject, suffix='_prepared', strip=True)
     result: 'N20020214S022_prepared.fits'
        
     fileNameUpdater(infilename='N20020214S022_prepared.fits', suffix='_biasCorrected')
     result: 'N20020214S022_prepared_biasCorrected.fits'
        
     fileNameUpdater(adIn=myAstrodataObject, prefix='testversion_')
     result: "testversion_N20020214S022.fits"
    
    """
    log = gemLog.getGeminiLog() 

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
    outFileName = prefix+name+suffix+filetype
    return outFileName
    
def listFileMaker(list=None, listName=None):
        """ 
        This function creates a list file of the input to IRAF.
        If the list requested all ready exists on disk, then it's filename
        is returned.
        This function is utilized by the CLManager. 
        NOTE: '@' must be post pended onto this listName if not done all ready 
        for use with IRAF.
        
        :param list: list of filenames to be written to a list file.
        :type list: list of strings
        
        :param listName: Name of file list is to be written to.
        :type listName: string
        """
        try:
            if listName==None:
                raise ToolboxError("listName can not be None, please provide a string")
            elif os.path.exists(listName):
                return listName
            else:
                fh = open(listName, 'w')
                for item in list:
                    fh.writelines(item + '\n')                    
                fh.close()
                return listName
        except:
            raise ToolboxError("Could not write inlist file for stacking.") 
        
def logDictParams(indict):
    """ A function to log the parameters in a provided dictionary.  Main use
    is to log the values in the dictionaries of parameters for function 
    calls using the ** method.
    
    :param indict: Dictionary full of parameters/settings to be recorded as 
                   fullinfo log messages.
    :type indict: dictionary. 
                  ex. {'param1':param1_value, 'param2':param2_value,...}
    
    """
    log = gemLog.getGeminiLog()
    for key in indict:
        log.fullinfo(repr(key)+' = '+repr(indict[key]), 
                     category='parameters')
        
def nbiascontam(adInputs, biassec=None):
    """
    This function will find the largest difference between the horizontal 
    component of every BIASSEC value and those of the biassec parameter. 
    The returned value will be that difference as an integer and it will be
    used as the value for the nbiascontam parameter used in the gireduce 
    call of the overscanSubtract primitive.
    
    :param adInputs: AstroData instance(s) to calculate the bias contamination 
    :type adInputs: AstroData instance in a list
    
    :param biassec: biassec parameter of format '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
    :type biassec: string 
    
    """
        
    # Prepare a stored value to be compared between the inputs
    retval=0
    # Loop through the inputs
    for ad in adInputs:
        # Pass the retrieved value to biassecStrToBiasContam function
        # to do the work in finding the difference of the biassec's
        val = biassecStrTonbiascontam(ad, biassec)
        # Check if value returned for this input is larger. Keep the largest
        if val > retval:
            retval = val
    return retval
    
def observationMode(ad):
    """ 
    A basic function to determine if the input is one of 
    (IMAGE|IFU|MOS|LONGSLIT) type.  It returns the type as a string. If input is  
    none of these types, then None is returned.
    """
    types = ad.getTypes()
    try:
        if 'IMAGE' in types:
            return 'IMAGE'
        elif 'IFU' in types:
            return 'IFU'
        elif 'MOS' in types:
            return 'MOS'
        elif 'LONGSLIT' in types:
            return 'LONGSLIT'
    except:
        raise ToolboxError('Input '+ad.filename+
                            ' is not of type IMAGE or IFU or MOS or LONGSLIT.')
    
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
        raise ToolBoxError('DANGER DANGER Will Robinson, pythonBool passed '+
        'in was not True or False, and thats just crazy talk :P')

def secStrToIntList(string):
    """ 
    A function to convert a string representing a list of integers to 
    an actual list of integers.
    
    :param string: string to be converted
    :type string: string of format '[#1:#2,#3:#4]'
    
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

def update_key_value(ad, valueFuncStr, phu=True):
    """
    This is a function to update header keys, perform logging of the changes
    and write history of change to the PHU.
    
    :param ad: astrodata instance to perform header key updates on
    :type ad: an AstroData instance
    
    :param valueFuncStr: string for an astrodata function or descriptor to 
                         perform on the input ad.
                         ie. for ad.countExts('SCI'), 
                         valueFuncStr="countExts('SCI')"
    :type valueFuncStr: string 
    
    :param phu: Is this update to be performed on the phu?
    :type phu: Python boolean (True/False)
               If False, the ad input must ONLY have ONE extension.
    
    """
    log = gemLog.getGeminiLog()
    keyAndCommentDict = {    'pixel_scale()':['PIXSCALE',
                             'Pixel scale in Y in [arcsec/pixel]'],
                             'gain()':['GAIN', 'Gain [e-/ADU]'],
                             'dispersion_axis()':['DISPAXIS','Dispersion axis'],
                             'countExts("SCI")':['NSCIEXT',
                             'Number of science extensions'],
                             #storeOriginalName() actually all ready writes to the PHU, but doubling it doesn't hurt.
                             'storeOriginalName()':['ORIGNAME',
                             'Original filename prior to processing'],
                             'read_noise()':['RDNOISE',
                             'readout noise in [e-]'],
                             'non_linear_level()':['NONLINEA',
                             'Non-linear regime level in [ADU]'],
                             'saturation_level()':['SATLEVEL',
                             'Saturation level in [ADU]'],                                    
                         }
    # Extract key and comment for input valueFuncStr from above dict
    key = keyAndCommentDict[valueFuncStr][0]
    comment = keyAndCommentDict[valueFuncStr][1]
    
    if phu:
        # Try to get orig value of key for history purposes, if not there
        # then no history is needed
        try:
            original_value = ad.phuGetKeyValue(key)
            comment = '(UPDATED) '+comment
            historyComment = 'Raw keyword '+key+'='+str(original_value)+\
                                      ' was overwritten in the PHU.'
            # Add and log history of orig key val to PHU before changing it 
            ad.getPHUHeader().add_history(historyComment)
            log.fullinfo('History comment added: '+historyComment)
        except:
            original_value = None
            comment = '(NEW) '+comment
            
        # handling storeOriginalName issues caused by deepcopy
        if valueFuncStr=='storeOriginalName()': 
            origname = ad.phuGetKeyValue('ORIGNAME')
            if origname==None:
                try:
                    origname= ad.storeOriginalName() 
                except:
                    log.critical('Unable to store original filename as does \
                                    not exist in this astrodata instance \
                                    anymore.  If using deepcopy on objects, \
                                    please ensure to use storeOriginalName()\
                                    on them prior to performing a deepcopy.')
        else:
            # using exec to perform the requested valueFuncStr on input ad    
            exec('try:\n'+
                 '    output_value = ad.%s\n' % valueFuncStr+
                 'except:\n'+
                 '    output_value = "An exception was thrown"')
        
            # Perform key update
            ad.phuSetKeyValue(key, output_value, comment)
        # log key update
        log.fullinfo(key+' = '+str(ad.phuGetKeyValue(key)), category='header')
        
    else:
        # double check there is only one extension being passed in
        if len(ad)!=1:
            log.warning('If phu=False, only a single extension AstroData \
                        instance can be passed in to update_key_value.')
        else:
            # Try to get orig value of key for history purposes, if not there
            # then no history is needed
            try:
                original_value = ad.getKeyValue(key)
                comment = '(UPDATED) '+comment
                historyComment = 'Raw keyword '+key+'='+str(original_value)+\
                                           ' was overwritten in extension '+\
                                          ad.extname()+','+str(ad.extver())
                # Add and log history of orig key val to PHU before changing it
                ad.getPHUHeader().add_history(historyComment)
                log.fullinfo('History comment added: '+historyComment)
            except:
                original_value = None
                comment = '(NEW) '+comment
            # using exec to perform the requested valueFuncStr on input ad
            exec('try:\n'+
                 '    output_value = ad.%s\n' % valueFuncStr+
                 'except:\n'+
                 '    output_value = "An exception was thrown"')
            #print 'GT399: key='+str(key)+', output_value='+str(output_value)+', comment='+str(comment)
            # Perform key update
            ad.setKeyValue(key, str(output_value), comment)
            # log key update
            log.fullinfo(key+' = '+str(ad.getKeyValue(key)), category='header')

def stdObsStruct(ad):
    """ 
    This function is used by standardizeStructure in primitives_GEMINI.
    
    It currently checks that the SCI extensions header key EXTNAME = 'SCI' 
    and EXTVER matches that of descriptor values 
        
    :param ad: astrodata instance to perform header key updates on
    :type ad: an AstroData instance
    
    """
    log = gemLog.getGeminiLog()    
    # Formatting so logger looks organized for these messages
    log.fullinfo('*'*50, 
                 category='header') 
    log.fullinfo('file = '+ad.filename, category='header')
    log.fullinfo('~'*50, 
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
        log.fullinfo('-'*50, 
                     category='header') 
        
#---------------------------------OBJECTS/CLASES--------------------------------

class CLManager(object):
    """
    This is a class that will take care of all the preparation and wrap-up 
    tasks needed when writing a primitive that wraps a IRAF CL routine.
        
    """
    # The original names of the files at the start of the 
    # primitive which called CLManager
    _preCLimageNames = None
    _preCLrefnames = None
    # The version of the names for input to the CL script
    imageInsCLdiskNames = None
    refInsCLdiskNames = None
    arrayInsCLdiskNames = None
    # Preparing other 'global' variables to be accessed throughout this class
    # Ins
    imageIns = None
    refIns = None
    arrayIns = None
    imageInsListName = None
    refInsListName = None
    arrayInsListName = None
    # Outs
    imageOuts = None
    refOuts = None
    arrayOuts = None
    numArrayOuts = None
    imageOutsNames = None
    refOutsNames = None
    arrayOutsNames = None
    imageOutsListName = None
    refOutsListName = None
    arrayOutsListName = None
    # Others
    suffix = None
    funcName = None
    status = None
    combinedImages = None
    templog = None
    log = None
     
    def __init__(self, imageIns=None, refIns=None, arrayIns=None, suffix=None,  
                  imageOutsNames=None, refOutsNames=None, numArrayOuts=None,
                 combinedImages=False, funcName=None, log=None):
        """
        This instantiates all the globally accessible variables (within the 
        CLManager class) and prepares the inputs for use in CL scripts by 
        temporarily writing them to disk with temporary names.  
        
        By using temporary filenames for the on disk copies 
        of the inputs, we avoid name collisions and by only temporarily writing
        them to disk, the 'user level functions' that utilize the CLManager
        will appear as if their processing is all done in memory.
       
        NOTE: all input images must have been prepared.
        
        :param imageIns: Input image(s). 
                         Use the imageInsFiles function to return the file names
                         for the temporary disk file versions of these inputs
                         in any desired form for input to IRAF.
        :type imageIns: astrodata object(s); Either as single instance, a list 
                        of them, or None.
        
        :param refIns: Input reference image(s). This may be used for any 
                       second set of input images.
                       Use the refInsFiles function to return the file names
                       for the temporary disk file versions of these inputs
                       in any desired form for input to IRAF.
        :type adIns: astrodata object(s); Either as single instance, a list of 
                     them, or None.
        
        :param arrayIns: Input array(s) of object locations in the images or 
                         any other arrays needed for input to IRAF.
                         Use the arrayInsFiles function to return the file names
                         for the temporary disk file versions of these inputs
                         in any desired form for input to IRAF.
        :type arrayIns: Python list-of-lists with each element of an array being 
                        an entire line to be written to an input file for IRAF; 
                        Either list of input arrays or None.
                        Format: 
                        [[list1-line1,list1-line2,...],[list2-line2,list2-line2,
                        ...],...]
                        another way of looking at it if lists are objects:
                        [LIST1, LIST2,...]
                        Even if only a single list is to be passed in, it MUST  
                        be within another set of [].
        
        :param suffix: Desired suffix to be added to input filenames to create 
                       the output names.
                       Use this option if not using the imageOutsNames/
                       refOutsNames parameters for the output names.
        :type suffix: String
        
        :param imageOutsNames: Desired final name(s) of output image(s) from 
                               IRAF.
                               Use the imageOutsFiles function to return these 
                               file names in any desired form for input to IRAF.
        :type imageOutsNames: String(s); Either a single string, a list of them 
                              of length matching the expected number of output 
                              images from IRAF, or None. If None, the list will 
                              be populated automatically by use of the 
                              'combinedImages' flag and post pending the 
                              'suffix' parameter onto the input image names.
                             
                        
        :param refOutsNames: Desired final name(s) of output reference image(s) 
                             from IRAF. 
                             These could be used to name any second set of 
                             output images.
                             Use the refOutsFiles function to return these 
                             file names in any desired form for input to IRAF.
        :type refOutsNames: String(s); Either a single string, a list of them of 
                            length matching the expected number of output 
                            reference images from IRAF, or None.
                            If None, no reference image outputs from IRAF will 
                            be handled by the CLManager.         
        
        :param numArrayOuts: The number of expected arrays to be output by IRAF.
                             The output array names will be automatically 
                             created.
                             Use the arrayOutsFiles function to return these
                             file names in any desired form for input to IRAF.
        :type numArrayOuts: int or None.
                            If 0 or None, no array outputs from IRAF will be 
                            handled by the CLManager.
        
        :param combinedImages: A flag to indicated that the input images of 
                               imageIns will be combined to form one single 
                               image output from IRAF.
                               The use of this parameter is optional and is  
                               overridden by providing imageOutsNames. 
                               No parallel version of this argument exists for
                               refIns.
        :type combinedImages: Python boolean (True/False)
        
        :param funcName: Name of the Python function using the CLManager. This 
                         is used to name the temporary files on disk for input 
                         to IRAF; so using the function name makes it easier to 
                         track down any errors that might occur.
        :type funcName: String
        
        :param log: logger object to send log messges to
        :type log: A gemLog object from astrodata/adutils/gemLog.py .
                   It is an upgraded version of the Python logger for use 
                   with all new scripts in gemini_python/ .
                   If None, then a null logger object will be created and used
                   (ie, no log file, no messages to screen).  
        
        """
        # Casting the two types of input images to lists for internal use, 
        # if not None
        if imageIns!=None:
            if isinstance(imageIns,list):
                self.imageIns = imageIns
            else:
                self.imageIns = [imageIns]
        if refIns!=None:  
            if isinstance(refIns,list):
                self.refIns = refIns
            else:
                self.refIns = [refIns]
        # Check that the inputs have been prepared, else then CL scripts might
        # not work correctly.
        self.status = True
        if imageIns!=None:
            for ad in self.imageIns:
                if (ad.phuGetKeyValue('GPREPARE')==None) and \
                   (ad.phuGetKeyValue('PREPARE')==None):
                    self.status = False
        if refIns!=None:
            for ad in self.refIns:
                if (ad.phuGetKeyValue('GPREPARE')==None) and \
                   (ad.phuGetKeyValue('PREPARE')==None):
                    self.status = False
        # All inputs prepared, then continue, else the False status will trigger
        # the caller to not proceed further.
        if self.status:
            # Create a temporary log file object
            self.templog = tempfile.NamedTemporaryFile() 
            if not log:
                # Get the REAL log file object
                self.log = gemLog.getGeminiLog()
            else:
                self.log = log
            # load these to global early as they are needed below
            self.suffix = suffix
            # start up global lists
            if imageIns!=None:
                self._preCLimageNames = []
                self.imageInsCLdiskNames = []
            if refIns!=None:
                self._preCLrefnames = []
                self.refInsCLdiskNames = []
            if arrayIns!=None:
                self.arrayInsCLdiskNames = []
            # load up the rest of the inputs to being global
            self.imageOutsNames = imageOutsNames
            self.refOutsNames = refOutsNames
            self.combinedImages = combinedImages
            self.funcName = funcName
            self.arrayIns = arrayIns
            self.numArrayOuts = numArrayOuts
            # now that everything is loaded to global make the uniquePrefix
            self.prefix = 'tmp'+ str(os.getpid())+self.funcName
            # now the preCLwrites can load up the input lists and write 
            # the temp files to disk
            self.preCLwrites()
     
    def arrayInsFiles(self, type=''):
        """
        The function to get the file names for the temporary files written to 
        disk for the arrayIns as either a string (or comma-separated string if 
        input was a list), a list of the filenames, or a list file. 
        These files are required to be on disk by IRAF and the file names are 
        automatically created when the CLManager is instantiated based on the 
        funcName parameter and the array location in the 'arrayIns' list.
        
        :param type: Desired form of the temp filenames on disk for arrayIns.
        :type type: 'string' for filenames as a string (comma-separated if input was a list),
                    'list' for filenames as strings in a python list, or
                    'listFile' for a IRAF type list file.
        """
        if type!='':
            if type=='string':
                return ','.join(self.arrayInsCLdiskNames)
            if type=='list':
                return self.arrayInsCLdiskNames
            if type=='listFile':
                arrayInsListName = listFileMaker(self.arrayInsCLdiskNames,
                                                    listName='arrayList'+str(os.getpid())+self.funcName)
                self.arrayInsListName = arrayInsListName
                return '@'+arrayInsListName
        else:
            raise ToolboxError('Parameter "type" must not be an empty string'+
                           '; choose either "string","list" or "listFile"')    
    
    def arrayOutsFiles(self, type=''):
        """
        This function is used to return the names of the array files to be written 
        to disk by IRAF in the form desired to pass into the IRAF routine call.
        The names of these files is automatically produced simply based
        on the funcName parameter, the string '_arrayOut_' and the integer value
        of the arrays location in the arrayOuts list.
        
        This function is simply for 'convenience' and can be ignored as long
        as the filenames in the CLManger.arrayOutsNames are passed into 
        IRAF properly.
        
        :param type: Desired form of the filenames on disk for refOutsNames.
        :type type: 'string' for filenames as a string (comma-separated if input was a list),
                    'list' for filenames as strings in a python list, or
                    'listFile' for a IRAF type list file.
        
        """
        # returning the arrayOutsNames contents in the form requested, else 
        # error log messsage
        if type!='':
            if type=='string':
                return ','.join(self.arrayOutsNames)
            if type=='list':
                return self.arrayOutsNames
            if type=='listFile':
                arrayOutsListName = listFileMaker(list=self.arrayOutsNames,
                                    listName='arrayOutsList'+str(os.getpid())+self.funcName)
                self.arrayOutsListName = arrayOutsListName
                return '@'+arrayOutsListName
        else:
            raise ToolboxError('Parameter "type" must not be an empty string'+
                           '; choose either "string","list" or "listFile"')
         
    def finishCL(self): 
        """ 
         Performs all the finalizing steps after CL script is ran. 
         This function is just a wrapper for postCLloads but might 
         contain more later.
         
        """    
        imageOuts, refOuts, arrayOuts = self.postCLloads()
        return (imageOuts, refOuts, arrayOuts) 
          
    def imageInsFiles(self, type=''):
        """
        The function to get the temporary files written to disk for the imageIns
        as either a string (or comma-separated string if input was a list), a 
        list of the file names, or a list file.  These files are required to 
        be on disk by IRAF and the file names are automatically created when 
        the CLManager is instantiated based on the 'funcName' parameter and 
        the original file names of the 'imageIns' astrodata objects.
        
        :param type: Desired form of the temp filenames on disk for imageIns.
        :type type: 'string' for filenames as a string (comma-separated if input was a list),
                    'list' for filenames as strings in a python list, or
                    'listFile' for a IRAF type list file.
        """
        if type!='':
            if type=='string':
                return ','.join(self.imageInsCLdiskNames)
            if type=='list':
                return self.imageInsCLdiskNames
            if type=='listFile':
                imageInsListName = listFileMaker(list=self.imageInsCLdiskNames,
                                    listName='imageList'+str(os.getpid())+self.funcName)
                self.imageInsListName = imageInsListName
                return '@'+imageInsListName
        else:
            raise ToolboxError('Parameter "type" must not be an empty string'+
                           '; choose either "string","list" or "listFile"')
            
    def imageOutsFiles(self, type=''):
        """
        This function is used to return the names of the images that will be  
        written to disk by IRAF in the form desired to pass into the IRAF 
        routine call.
        The names of these files can either be defined using
        the imageOutsNames parameter set during the CLManager initial call, or
        automatically created in one of two ways:
        1. Combine case: triggered by,
        imageOutsNames=None, combinedImages=True, suffix=<any string>.
        Then imageOutsNames will be a list with only the filename from the first 
        input of imageIns post pended by the value of suffix.
        2. Non-combine case: triggered by,
        imageOutsNames=None, combinedImages=False, suffix=<any string>.
        Then imageOutsNames will be a list with each file name of the imageIns  
        post pended by the value of suffix.
        
        This function is simply for 'convenience' and can be ignored as long
        as the imageOutsNames is set properly and its filenames are passed into 
        IRAF properly.
        
        :param type: Desired form of the filenames on disk for imageOutsNames.
        :type type: 'string' for filenames as a string (comma-separated if input was a list),
                    'list' for filenames as strings in a python list, or
                    'listFile' for a IRAF type list file.
        
        """
        # Loading up the imageOutsNames list if not done yet and params are set 
        # correctly, else error log message.
        if self.imageOutsNames==None:
            self.imageOutsNames = []
            if self.combinedImages and (self.suffix!=None):
                name = fileNameUpdater(adIn=self.imageIns[0], 
                                       suffix=self.suffix)
                self.imageOutsNames.append(name)
            elif (not self.combinedImages) and (self.suffix!=None):
                for ad in self.imageIns:
                    name = fileNameUpdater(adIn=ad, suffix=self.suffix)
                    self.imageOutsNames.append(name) 
            else:
                self.log.error('The "automatic" setting of imageOutsNames can '+
                        'only work if at least the suffix parameter is set')
        # The parameter was set, ie not None
        else:
            # Cast it to a list for use below
            if isinstance(self.imageOutsNames,str):
                self.imageOutsNames = [self.imageOutsNames]   
        # returning the imageOutsNames contents in the form requested, else error
        # log messsage
        if type!='':
            if type=='string':
                return ','.join(self.imageOutsNames)
            if type=='list':
                return self.imageOutsNames
            if type=='listFile':
                imageOutsListName = listFileMaker(list=self.imageOutsNames,
                                    listName='imageOutsList'+str(os.getpid())+self.funcName)
                self.imageOutsListName = imageOutsListName
                return '@'+imageOutsListName
        else:
            raise ToolboxError('Parameter "type" must not be an empty string'+
                           '; choose either "string","list" or "listFile"')
             
    def obsmodeAdd(self, ad):
        """This is an internally used function to add the 'OBSMODE' key to the 
           inputs for use by IRAF routines in the GMOS package.
        """
        if 'GMOS' in ad.getTypes():
            ad.phuSetKeyValue('OBSMODE', observationMode(ad) , 
                      'Observing mode (IMAGE|IFU|MOS|LONGSLIT)')
        return ad    
    
    def obsmodeDel(self, ad):
        """This is an internally used function to delete the 'OBSMODE' key from
           the outputs from IRAF routines in the GMOS package.
        """
        if 'GMOS' in ad.getTypes():
            del ad.getPHU().header['OBSMODE']
        return ad
    
    def postCLloads(self):
        """ This function takes care of loading the image, reference and/or 
            array files output by IRAF back into memory, in the form of the 
            imageOuts, refOuts and arrayOuts variables, and deleting those disk 
            files. 
            Then it will delete ALL the temporary files created by the 
            CLManager.  If the 'OBSMODE' phu key was added during preCLwrites
            to the imageIns and/or refIns, then it will be deleted here.
            
            NOTE: all fits files loaded back into memory in the form of 
            astrodata objects (adOuts, refOuts), will be in 'update' mode.
        
        """
        # Loading any output images into imageOuts and 
        # killing off any disk files caused by them
        if self.imageOutsNames!=None:
            self.imageOuts = []
            self.log.fullinfo('Loading output images into imageOuts and'+\
                              ' removing temporary files from disk.')
            for name in self.imageOutsNames:
                # Loading the file into an astrodata object
                ad = AstroData(name, mode='update')
                # Removing the 'OBSMODE' phu key if it is in there
                ad = self.obsmodeDel(ad)
                # appending the astrodata object to the imageOuts list to be
                # returned
                self.imageOuts.append(ad)
                # Deleting the file from disk
                os.remove(name)
                self.log.fullinfo(name+' was loaded into memory')
                self.log.fullinfo(name+' was deleted from disk')
            if self.imageOutsListName!=None:
                os.remove(self.imageOutsListName)
                self.log.fullinfo('Temporary list '+self.imageOutsListName+
                                  ' was deleted from disk')
        # Loading any output ref images into refOuts and 
        # killing off any disk files caused by them
        if self.refOutsNames!=None:
            self.refOuts = []
            self.log.fullinfo('Loading output reference images into refOuts'+\
                              ' and removing temporary files from disk.')
            for name in self.refOutsNames:
                # Loading the file into an astrodata object
                ad = AstroData(name, mode='update')
                # Removing the 'OBSMODE' phu key if it is in there
                ad = self.obsmodeDel(ad)
                # appending the astrodata object to the refOuts list to be
                # returned
                self.refOuts.append(ad)
                # Deleting the file from disk
                os.remove(name)
                self.log.fullinfo(name+' was loaded into memory')
                self.log.fullinfo(name+' was deleted from disk')
            if self.refOutsListName!=None:
                os.remove(self.refOutsListName)
                self.log.fullinfo('Temporary list '+self.refOutsListName+
                                  ' was deleted from disk') 
        # Loading any output arrays into arrayOuts and 
        # killing off any disk files caused by them 
        if self.arrayOutsNames!=None:
            self.arrayOuts = []
            self.log.fullinfo('Loading output reference array into arrayOuts'+\
                              ' and removing temporary files from disk.')
            for name in self.arrayOutsNames:
                # read in input array txt file to an array with each line
                # of the file is an element of the array, ie a list of lines.
                fin = open(name,'r')
                lineList = fin.readlines()
                fin.close()                
                # appending the array to the arrayOuts list to be returned
                self.arrayOuts.append(lineList)
                # Deleting the file from disk
                os.remove(name)
                self.log.fullinfo(name+' was loaded into memory')
                self.log.fullinfo(name+' was deleted from disk')
            if self.arrayOutsListName!=None:
                os.remove(self.arrayOutsListName)
                self.log.fullinfo('Temporary list '+self.arrayOutsListName+
                                  ' was deleted from disk') 
        # Killing off any disk files associated with imageIns
        if self.imageIns!=None:
            self.log.fullinfo('Removing temporary files associated with '+\
                              'imageIns from disk')
            for name in self.imageInsCLdiskNames:
                # Deleting the file from disk
                os.remove(name)
                self.log.fullinfo(name+' was deleted from disk')
            if self.imageInsListName!=None:
                os.remove(self.imageInsListName)
                self.log.fullinfo('Temporary list '+self.imageInsListName+
                                  ' was deleted from disk') 
        # Killing off any disk files associated with refIns
        if self.refIns!=None:
            self.log.fullinfo('Removing temporary files associated with '+\
                              'refIns from disk')
            for name in self.refInsCLdiskNames:
                # Deleting the file from disk
                os.remove(name)
                self.log.fullinfo(name+' was deleted from disk')
            if self.refInsListName!=None:
                os.remove(self.refInsListName)
                self.log.fullinfo('Temporary list '+self.refInsListName+
                                  ' was deleted from disk') 
        # Killing off any disk files associated with arrayIns
        if self.arrayIns!=None:
            self.log.fullinfo('Removing temporary files associated with '+\
                              'arrayIns from disk')
            for name in self.arrayInsCLdiskNames:
                # Deleting the file from disk
                os.remove(name)
                self.log.fullinfo(name+' was deleted from disk')
            if self.arrayInsListName!=None:
                os.remove(self.arrayInsListName)
                self.log.fullinfo('Temporary list '+self.arrayInsListName+
                                  ' was deleted from disk') 
                
        return (self.imageOuts, self.refOuts, self.arrayOuts)            

    def preCLimageNames(self):
        """Just a simple function to return the value of the private member
           variable _preCLimageNames that is a list of the filenames of imageIns.
        """
        return self._preCLimageNames
    
    def preCLwrites(self):
        """ The function that writes the files in memory to disk with temporary 
            names and saves the original and temporary names in lists and 
            fills out the output file name lists for any output arrays if needed.  
            The 'OBSMODE' PHU key will also be added to all input GMOS images
            of imageIns and refIns.
        
        """
        # preparing the input filenames for temporary input image files to 
        # IRAF if needed along with saving the original astrodata filenames    
        if self.imageIns!=None:
            for ad in self.imageIns:            
                # Adding the 'OBSMODE' phu key if needed
                ad = self.obsmodeAdd(ad)
                # Load up the _preCLimageNames list with the input's filename
                self._preCLimageNames.append(ad.filename)
                # Strip off all postfixes and prefix filename with a unique prefix
                name = fileNameUpdater(adIn=ad, prefix=self.prefix, strip=True)
                # store the unique name in imageInsCLdiskNames for later reference
                self.imageInsCLdiskNames.append(name)
                # Log the name of this temporary file being written to disk
                self.log.fullinfo('Temporary image file on disk for input to CL: '
                                  +name)
                # Write this file to disk with its unique filename 
                ad.write(name, rename=False)
        # preparing the input filenames for temperary input ref image files to 
        # IRAF if needed along with saving the original astrodata filenames
        if self.refIns!=None:
            for ad in self.refIns:            
                # Adding the 'OBSMODE' phu key if needed
                ad = self.obsmodeAdd(ad)
                # Load up the _preCLrefnames list with the input's filename
                self._preCLrefnames.append(ad.filename)
                # Strip off all suffixs and prefix filename with a unique prefix
                name = fileNameUpdater(adIn=ad, prefix=self.prefix, strip=True)
                # store the unique name in refInsCLdiskNames for later reference
                self.refInsCLdiskNames.append(name)
                # Log the name of this temporary file being written to disk
                self.log.fullinfo('Temporary ref file on disk for input to CL: '
                                  +name)
                # Write this file to disk with its unique filename 
                ad.write(name, rename=False)
        # preparing the input filenames for temperary input array files to 
        # IRAF if needed and writing them to disk.   
        if self.arrayIns!=None:
            count=1
            for array in self.arrayIns:
                # creating temp name for array
                name = self.prefix+'_arrayIn_'+str(count)+'.txt'
                # store the unique name in arrayInsCLdiskNames for later reference
                self.arrayInsCLdiskNames.append(name)
                # Log the name of this temporary file being written to disk
                self.log.fullinfo('Temporary ref file on disk for input to CL: '
                                  +name)
                # Write this array to text file on disk with its unique filename 
                fout = open(name,'w')
                for line in array:
                    fout.write(line)
                fout.close()
                
                count=count+1
        # preparing the output filenames for temperary output array files from 
        # IRAF if needed, no writing here that is done by IRAF.
        if (self.numArrayOuts!=None) and (self.numArrayOuts!=0):
            # create empty list of array file names to be loaded in loop below
            self.arrayOutsNames = []
            for count in range(1,self.numArrayOuts+1):
                # Create name of output array file
                name = self.prefix+'_arrayOut_'+str(count)+'.txt'
                # store the unique name in arrayOutsNames for later reference
                self.arrayOutsNames.append(name)
                # Log the name of this temporary file being written to disk
                self.log.fullinfo('Temporary ref file on disk for input to CL: '
                                  +name)
    
    def refInsFiles(self, type=''):
        """
        The function to get the temporary files written to disk for the refIns
        as either a string (or comma-separated string if input was a list), a 
        list of the filenames, or a list file. These files are required to 
        be on disk by IRAF and the file names are automatically created when 
        the CLManager is instantiated based on the 'funcName' parameter and 
        the original file names of the 'refIns' astrodata objects.
        
        :param type: Desired form of the temp filenames on disk for refIns.
        :type type: 'string' for filenames as a string (comma-separated if input was a list),
                    'list' for filenames as strings in a python list, or
                    'listFile' for a IRAF type list file.
        """
        if type!='':
            if type=='string':
                return ','.join(self.refInsCLdiskNames)
            if type=='list':
                return self.refInsCLdiskNames
            if type=='listFile':
                refInsListName = listFileMaker(list=self.refInsCLdiskNames,
                                    listName='refList'+str(os.getpid())+self.funcName)
                self.refInsListName = refInsListName
                return '@'+refInsListName
        else:
            raise ToolboxError('Parameter "type" must not be an empty string'+
                           '; choose either "string","list" or "listFile"')  
                
    def refOutsFiles(self, type=''):
        """
        This function is used to return the names of the reference images that
        will be written to disk by IRAF in the form desired to pass into the 
        IRAF routine call.
        The names of these files can either be defined using
        the refOutsNames parameter set during the CLManager initial call, or
        automatically created in one way:
        Triggered by, refOutsNames=None and suffix=<any string>.
        Then refOutsNames will be a list with each file name of the refIns  
        post pended by the value of suffix.
        
        This function is simply for 'convenience' and can be ignored as long
        as the refOutsNames is set properly and its filenames are passed into 
        IRAF properly.
        
        :param type: Desired form of the filenames on disk for refOutsNames.
        :type type: 'string' for filenames as a string (comma-separated if input was a list),
                    'list' for filenames as strings in a python list, or
                    'listFile' for a IRAF type list file.
        
        """
        # Loading up the refOutsNames list if not done yet and params are set 
        # correctly, else error log message
        if self.refOutsNames==None:
            self.refOutsNames = []
            if (self.suffix!=None):
                for ad in self.refIns:
                    name = fileNameUpdater(adIn=ad, suffix=self.suffix)
                    self.refOutsNames.append(name) 
            else:
                self.log.error('The "automatic" setting of refOutsNames can '+
                        'only work if at least the suffix parameter is set')
        # The parameter was set, ie not None
        else:
            # Cast it to a list for use below
            if isinstance(self.refOutsNames,str):
                self.refOutsNames = [self.refOutsNames] 
        # returning the refOutsNames contents in the form requested, else error
        # log messsage
        if type!='':
            if type=='string':
                return ','.join(self.refOutsNames)
            if type=='list':
                return self.refOutsNames
            if type=='listFile':
                refOutsListName = listFileMaker(list=self.refOutsNames,
                                    listName='refOutsList'+str(os.getpid())+self.funcName)
                self.refOutsListName = refOutsListName
                return '@'+refOutsListName
        else:
            raise ToolboxError('Parameter "type" must not be an empty string'+
                           '; choose either "string","list" or "listFile"')             

class IrafStdout():
    """ This is a class to act as the standard output for the IRAF 
        routines that instead of printing its messages to the screen,
        it will print them to the gemlog.py logger that the primitives use
        
    """
    log=None
    
    def __init__(self):
        """ 
        A function that is needed IRAF but not used in our wrapping its
        scripts.
            
        """
        self.log = gemLog.getGeminiLog()
    
    def flush(self):
        """ A function that is needed IRAF but not used in our wrapping its
            scripts
        """
        pass
    
    def write(self, out):
        """ This function converts the IRAF console prints to logger calls.
            If the print has 'PANIC' in it, then it becomes a error log message,
            else it becomes a fullinfo message.
            
        """
        if 'PANIC' in out or 'ERROR' in out:
            self.log.error(out, category='clError')
        elif len(out) > 1:
            self.log.fullinfo(out, category='clInfo')

class ScienceFunctionManager():
    """
    A manager class to hold functions for performing input checks, naming,
    log instantiation... code that is repeated throughout all the 'user level
    functions' in the gempy libraries (currently those functions in 
    science/geminiScience.py and gmosScience.py).
    """
    # Set up global variables 
    adInputs = None
    outNames = None
    suffix = None
    funcName = None
    combinedInputs = False
    log = None  
    
    def __init__(self, adInputs=None, outNames=None, suffix=None, funcName=None,
                 combinedInputs=False):
        """
        This will load up the global variables to use throughout the manager
        functions and instantiate the logger object for use in here and 
        back in the 'user level function' that is utilizing this manager.
        
        Either a 'main' type logger object, if it exists, or a null logger will
        be returned from startUp() along with the checked adInputs and outNames.
        
        :param adInputs: Astrodata inputs to have DQ extensions added to
        :type adInputs: Astrodata objects, either a single or a list of objects.
                        At least one object MUST be passed in for adInputs.
        
        :param outNames: filenames of output(s)
        :type outNames: String, either a single or a list of strings of same 
                        length as adInputs.
        
        :param suffix: String to add on the end of the input filenames 
                       (or outNames if not None) for the output filenames.
        :type suffix: string
        
        :param funcName: Name of the Python function using this manager.
        :type funcName: String
        
        :param combinedInputs: A flag to indicated that the input images of 
                               adInputs will be combined to form one single 
                               image output.
                               The use of this parameter is optional and is  
                               overridden by providing outNames. 
        :type combinedInputs: Python boolean (True/False)
        
        """ 
        self.adInputs = adInputs
        self.outNames = outNames
        self.suffix = suffix
        self.funcName = funcName
        self.combinedInputs = combinedInputs
        # loading of the logger 
        self.log = gemLog.getGeminiLog()
    
    def autoVardq(self, fl_vardq):
        """
        This is a function to perform either the 'AUTO' fl_vardq determination
        or just to check convert the value from True->iraf.yes, False->iraf.no .
        
        NOTE: 'AUTO' uses the first input to determine if VAR and  
        DQ frames exist, so, if the first does, then the rest MUST 
        also have them as well.
        
        :param fl_vardq: The value of the fl_vardq parameter at the start of the
                         Python user level function.
        :type fl_vardq: either: Python bool (True/False) or the string 'AUTO'
        """
        from astrodata.adutils.gemutil import pyrafLoader
        # loading and/or bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
        
        if fl_vardq=='AUTO':
            # if there are matching numbers of VAR, DQ and SCI extensions
            # then set to yes to ensure the outputs have VAR and DQ's as well.
            if self.adInputs[0].countExts('VAR')==\
                        self.adInputs[0].countExts('DQ')\
                                            ==self.adInputs[0].countExts('SCI'):
                fl_vardq=yes
            else:
                fl_vardq=no
        else:
            # 'AUTO' wasn't selected, so just convert the python bools to iraf
            # yes or no.
            if fl_vardq:
                fl_vardq=yes
            elif fl_vardq==False:
                fl_vardq=no
       
        return fl_vardq
    
    def markHistory(self, adOutputs=None, historyMarkKey=None):
        """
        The function to use near the end of a python user level function to 
        add a historyMark timestamp to all the outputs indicating when and what
        function was just performed on them, then logging the new historyMarkKey
        PHU key and updated 'GEM-TLM' key values due to historyMark.
        
        Note: The GEM-TLM key will be updated, or added if not in the PHU yet, 
        automatically everytime wrapUp is called.
        
        :param adOutputs: List of astrodata instance(s) to perform historyMark on.
        :type adOutputs: Either a single or multiple astrodata instances in a list.
        
        :param historyMarkKey: The PHU header key to write the current UT time 
        :type historyMarkKey: Under 8 character, all caps, string.
                              If None, then only 'GEM-TLM' is added/updated.
        """
        # casting inputs to a list if not one all ready to make loop work right
        if not isinstance(adOutputs,list):
            adOutputs = [adOutputs]
        
        # looping though inputs to perform historyMark on each of them
        for ad in adOutputs:
            # Adding 'GEM-TLM' (automatic) and historyMarkKey (if not None)
            # time stamps to the PHU
            ad.historyMark(key=historyMarkKey, stomp=False)
            
            # Updating log with new GEM-TLM and GIFLAT time stamps
            self.log.fullinfo('*'*50, category='header')
            self.log.fullinfo('File = '+ad.filename, category='header')
            self.log.fullinfo('~'*50, category='header')
            self.log.fullinfo('PHU keywords updated/added:\n', 'header')
            self.log.fullinfo('GEM-TLM = '+ad.phuGetKeyValue('GEM-TLM'), 
                              category='header')
            if historyMarkKey!=None:
                self.log.fullinfo(historyMarkKey+' = '+
                                  ad.phuGetKeyValue(historyMarkKey), 
                                  category='header')
            self.log.fullinfo('-'*50, category='header')
        
    def startUp(self):
        """
        This function is to perform the input checks and fill out the outNames
        parameter if needed.
        """
        try:
            # Ensuring there are inputs ad's to work on, else raise
            if self.adInputs==None:
                raise ToolboxError('The parameter "adInputs" must not be None')
            elif isinstance(self.adInputs,list):
                if len(self.adInputs)==0:
                    raise ToolboxError('The parameter "adInputs" must not be\
                                        an empty list')
                
            # raise if funcName=None as it needs to be a valid string
            if self.funcName==None:
                raise ToolboxError('funcName was None, must be a string.')
            
            self.log.status('**STARTING** the '+self.funcName+' function')
    
            # casting the input ad's and outNames into lists 
            # for internal use and simplified returning
            if not isinstance(self.adInputs,list):
                self.adInputs=[self.adInputs]
            if self.outNames==None:
                self.outNames = []
            if (not isinstance(self.outNames,list)):
                if isinstance(self.outNames,str):
                    self.outNames = [self.outNames]
                elif self.suffix==None:
                    raise ToolboxError('outNames must be either a string or a \
                                     list of them.')
            
            # Checking combinations of outNames, combinedInputs, suffix and 
            # len(adInputs) are valid
            if len(self.outNames)>0:
                if not self.combinedInputs:
                    if len(self.adInputs)!= len(self.outNames):
                        raise ToolboxError('outNames was not None or the same \
                                    length as number of adInputs')
            elif len(self.adInputs)>1:
                    if self.suffix==None:
                        raise ToolboxError('Both outNames and suffix were None')
                
            # Checking the current outNames and loading it up if needed
            if len(self.outNames)!=len(self.adInputs):
                if self.combinedInputs:
                    ad = self.adInputs[0]
                    self.log.debug('Calling gemt.fileNameUpdater on '+
                                   ad.filename)
                    outName = fileNameUpdater(infilename=ad.filename,
                                              suffix=self.suffix, strip=False)
                    self.outNames.append(outName)
                else:
                    for ad in self.adInputs:
                        self.log.debug('Calling gemt.fileNameUpdater on '+
                                       ad.filename)
                        outName = fileNameUpdater(infilename=ad.filename,
                                                  suffix=self.suffix, 
                                                  strip=False)
                        self.outNames.append(outName)
                
            # return the now checked and loaded up (if needed) adInputs, 
            # outNames and log object.
            return (self.adInputs, self.outNames, self.log)
            
        except:
            # logging the exact message from the actual exception that was 
            # raised in the try block. Then raising a general ToolboxError 
            # with message.
            self.log.critical(repr(sys.exc_info()[1]))
            raise ToolboxError('An Error occurred during\
                                ScienceFunctionManager.startUp')

 
