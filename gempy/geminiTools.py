# Author: Kyle Mede, May 2010
# This module provides many functions used by all primitives 

import os, sys

import pyfits as pf
import numpy as np
import tempfile
import astrodata
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
from astrodata import Errors

def checkInputsMatch(adInsA=None, adInsB=None):
    """
    This function will check if the inputs match.  It will check the filter,
    binning and shape/size of the every SCI frames in the inputs.
    
    There must be a matching number of inputs for A and B.
    
    :param adInsA: input astrodata instance(s) to be check against adInsB
    :type adInsA: AstroData objects, either a single or a list of objects
                Note: inputs A and B must be matching length lists or single 
                objects
    
    :param adInsB: input astrodata instance(s) to be check against adInsA
    :type adInsB: AstroData objects, either a single or a list of objects
                  Note: inputs A and B must be matching length lists or single 
                  objects
    """
    log = gemLog.getGeminiLog() 
    
    # Check inputs are both matching length lists or single objects
    if (adInsA is None) or (adInsB is None):
        log.error('Neither A or B inputs can be None')
        raise Errors.ToolboxError('Either A or B inputs were None')
    if isinstance(adInsA,list):
        if isinstance(adInsB,list):
            if len(adInsA)!=len(adInsB):
                log.error('Both the A and B inputs must be lists of MATCHING'+
                          ' lengths.')
                raise Errors.ToolboxError('There were miss-matched numbers ' \
                                          'of A and B inputs.')
    if isinstance(adInsA,AstroData):
        if isinstance(adInsB,AstroData):
            # casting both A and B inputs to lists for looping later
            adInsA = [adInsA]
            adInsB = [adInsB]
        else:
            log.error('Both the A and B inputs must be lists of MATCHING'+
                      ' lengths.')
            raise Errors.ToolboxError('There were miss-matched numbers of '+
                               'A and B inputs.')
    
    for count in range(0,len(adInsA)):
        A = adInsA[count]
        B = adInsB[count]
        log.status('Checking inputs '+A.filename+' and '+B.filename)
        
        if A.countExts('SCI')!=B.countExts('SCI'):
            log.error('Inputs have different numbers of SCI extensions.')
            raise Errors.ToolboxError('Miss-matching number of SCI ' \
                                      'extensions in inputs')
        for extCount in range(1,A.countExts('SCI')+1):
            # grab matching SCI extensions from A's and B's
            sciA = A[('SCI',extCount)]
            sciB = B[('SCI',extCount)]
            
            log.status('Checking SCI extension '+str(extCount))
            
            # Check shape/size
            if sciA.data.shape!=sciB.data.shape:
                log.error('Extensions have different shapes')
                raise Errors.ToolboxError('Extensions have different shape')
            
            # Check binning
            aX = sciA.detector_x_bin()
            aY = sciA.detector_y_bin()
            bX = sciB.detector_x_bin()
            bY = sciB.detector_y_bin()
            if (aX!=bX) or (aY!=bY):
                log.error('Extensions have different binning')
                raise Errors.ToolboxError('Extensions have different binning')
        
            # Check filter
            if sciA.filter_name().asPytype()!=sciB.filter_name().asPytype():
                log.error('Extensions have different filters')
                raise Errors.ToolboxError('Extensions have different filters')
        
        log.status('Inputs match')    

def fileNameUpdater(adIn=None, infilename='', suffix='', prefix='',
                    strip=False):
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
        
     fileNameUpdater(infilename='N20020214S022_prepared.fits',
         suffix='_biasCorrected')
     result: 'N20020214S022_prepared_biasCorrected.fits'
        
     fileNameUpdater(adIn=myAstrodataObject, prefix='testversion_')
     result: 'testversion_N20020214S022.fits'
    
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
        # If key was 'None', ie. storeOriginalName() wasn't ran yet, then run
        # it now
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
                raise Errors.ToolboxError("listName can not be None, please " \
                                          "provide a string")
            elif os.path.exists(listName):
                return listName
            else:
                fh = open(listName, 'w')
                for item in list:
                    fh.writelines(item + '\n')                    
                fh.close()
                return listName
        except:
            raise Errors.ToolboxError("Could not write inlist file for " \
                                      "stacking.") 
        
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

def log_message(function, name, message_type):
    if function == 'ulf':
        full_function_name = 'user level function'
    else:
        full_function_name = function
    if message_type == 'calling':
        message = 'Calling the %s %s' % (full_function_name, name)
    if message_type == 'starting':
        message = 'Starting the %s %s' % (full_function_name, name)
    if message_type == 'finishing':
        message = 'Finishing the %s %s' % (full_function_name, name)
    if message_type == 'completed':
        message = 'The %s %s completed successfully' % (name,
                                                        full_function_name)
    if message:
        return message
    else:
        return None

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
        raise Errors.ToolBoxError('DANGER DANGER Will Robinson, pythonBool ' \
                                  ' passed in was not True or False, and ' \
                                  ' thats just crazy talk :P')

def updateKeyValue(adinput=None, function=None, value=None, extname=None):
    """
    This function updates keywords in the headers of the input dataset,
    performs logging of the changes and writes history keyword related to the
    changes to the PHU.
    
    :param adinput: astrodata instance to perform header key updates on
    :type adinput: an AstroData instance
    
    :param function: string for an astrodata function or descriptor to 
                         perform on the input ad.
                         ie. for ad.countExts('SCI'), 
                         function='countExts('SCI')'
    :type function: string 
    
    :param extname: Set to 'PHU', 'SCI', 'VAR' or 'DQ' to update the given
                    keyword in the PHU, SCI, VAR or DQ extension, respectively.
                    
    :type extname: string
    """
    log = gemLog.getGeminiLog()
    historyComment = None
    update = False
    keyAndCommentDict = {
        'pixel_scale()':['PIXSCALE', 'Pixel scale in Y in [arcsec/pixel]'],
        'gain()':['GAIN', 'Gain [e-/ADU]'],
        'dispersion_axis()':['DISPAXIS','Dispersion axis'],
        'countExts("SCI")':['NSCIEXT', 'Number of science extensions'],
        # storeOriginalName() actually all ready writes to the PHU, but
        # doubling it doesn't hurt.
        'storeOriginalName()':
            ['ORIGNAME', 'Original filename prior to processing'],
        'read_noise()':['RDNOISE', 'readout noise in [e-]'],
        'non_linear_level()':['NONLINEA', 'Non-linear regime level in [ADU]'],
        'saturation_level()':['SATLEVEL', 'Saturation level in [ADU]'],
        'bunit':['BUNIT', 'Physical units'],
        'len(output)':['NEXTEND', 'Number of extensions'],
                        }
    # Extract key and comment for input function from above dict
    if function not in keyAndCommentDict:
        raise Errors.Error("Unknown value for the 'function' parameter")
    else:
        key = keyAndCommentDict[function][0]
        comment = keyAndCommentDict[function][1]
    
    if extname == "PHU":
        # Check to see whether the keyword is already in the PHU
        original_value = adinput.phuGetKeyValue(key)
        if original_value is not None:
            # The keyword exists, so store a history comment for later use
            log.debug("Keyword %s=%s already exists in the PHU" \
                  % (key, original_value))
            comment = '(UPDATED) %s' % comment
            msg = "updated"
            historyComment = "Raw keyword %s=%s was overwritten in the PHU " \
                             "by AstroData" % (key, original_value)
        else:
            comment = '(NEW) %s' % comment
            msg = "added"

        # Use exec to perform the requested function on input
        try:
            exec('output_value = adinput.%s' % function)
        except:
            output_value = value
        # Only update the keyword value in the PHU if it is different from the
        # value already in the PHU
        log.debug ("Original value=%s, Output value=%s" \
                  % (original_value, output_value))
        if output_value is not None:
            if output_value != original_value:
                # Update the header and write a history comment
                adinput.phuSetKeyValue(key, output_value, comment)
                log.fullinfo("PHU keyword %s=%s %s" \
                             % (key, adinput.phuGetKeyValue(key), msg),
                             category='header')
                # Only need to write a history comment if the value in the
                # header is actually overwritten
                if original_value is None:
                    historyComment = "New keyword %s=%s was written to the " \
                                     "PHU by AstroData" % (key, output_value)
                adinput.getPHUHeader().add_history(historyComment)
                log.fullinfo('History comment added: %s' % historyComment)
    else:
        if extname is None:
            extname = "SCI"
        for ext in adinput[extname]:
            # Check to see whether the keyword is already in the pixel data
            # extension 
            original_value = ext.getKeyValue(key)
            if original_value is not None:
                log.debug("Keyword %s=%s already in extension %s,%s" \
                          % (key, original_value, extname, ext.extver()))
                comment = '(UPDATED) %s' % comment
                msg = "updated"
                historyComment = "Raw keyword %s=%s was overwritten in " \
                                 "extension %s,%s" % (key, original_value,
                                                      extname, ext.extver())
            else:
                comment = '(NEW) %s' % comment
                msg = "added"
            # Use exec to perform the requested function on input
            try:
                exec('output_value = ext.%s' % function)
            except:
                output_value = value
            # Only update the keyword value in the pixel data extension if it
            # is different from the value already in the pixel data extension
            if original_value is not None and output_value is not None:
                if output_value != original_value:
                    # Update the header and write a history comment
                    ext.setKeyValue(key, output_value, comment)
                    log.fullinfo("%s,%s keyword %s=%s %s" \
                                 % (extname, ext.extver(), key,
                                    ext.getKeyValue(key), msg),
                                 category="header")
                    # Only need to write a history comment if the value in the
                    # header is actually overwritten
                    if output_value != original_value:
                        adinput.getPHUHeader().add_history(historyComment)
                        log.fullinfo('History comment added: %s' \
                                     % historyComment)
