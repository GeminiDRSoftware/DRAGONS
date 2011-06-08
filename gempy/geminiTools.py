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

def checkInputsMatch(adInsA=None, adInsB=None, check_filter=True):
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
        log.error('Neither A nor B inputs can be None')
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
        
        if A.count_exts('SCI')!=B.count_exts('SCI'):
            log.error('Inputs have different numbers of SCI extensions.')
            raise Errors.ToolboxError('Miss-matching number of SCI ' \
                                      'extensions in inputs')
        for extCount in range(1,A.count_exts('SCI')+1):
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
        
            # Check filter if desired
            if check_filter:
                if (sciA.filter_name().as_pytype() != 
                    sciB.filter_name().as_pytype()):
                    log.error('Extensions have different filters')
                    raise Errors.ToolboxError('Extensions have different ' +
                                              'filters')
        
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
        phuOrigFilename = adIn.phu_get_key_value('ORIGNAME') 
        # If key was 'None', ie. store_original_name() wasn't ran yet, then run
        # it now
        if phuOrigFilename is None:
            # Storing the original name of this astrodata object in the PHU
            phuOrigFilename = adIn.store_original_name()
            
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
        message = 'Calling the %s %s' \
                  % (full_function_name, name)
    if message_type == 'starting':
        message = 'Starting the %s %s' \
                  % (full_function_name, name)
    if message_type == 'finishing':
        message = 'Finishing the %s %s' \
                  % (full_function_name, name)
    if message_type == 'completed':
        message = 'The %s %s completed successfully' \
                  % (name, full_function_name)
    if message:
        return message
    else:
        return None

def make_dict(key_list=None, value_list=None):
    """
    The make_dict function creates a dictionary with the elements in 'key_list'
    as the key and the elements in 'value_list' as the value to create an
    association between the input science dataset (the 'key_list') and a, for
    example, dark that is needed to be subtracted from the input science
    dataset. This function also does some basic checks to ensure that the
    filters, exposure time etc are the same.

    :param key: List containing one or more AstroData objects
    :type key: AstroData

    :param value: List containing one or more AstroData objects
    :type value: AstroData
    """
    # Check the inputs have matching filters, binning and SCI shapes.
    #checkInputsMatch(adInsA=darks, adInsB=adInputs)
    ret_dict = {}
    if len(key_list) == 1 and len(value_list) == 1:
        # There is only one key and one value - create a single entry in the
        # dictionary
        ret_dict[key_list[0]] = value_list[0]
    elif len(key_list) > 1 and len(value_list) == 1:
        # There is only one value for the list of keys
        while i in range (0, len(key_list)):
            ret_dict[key_list[i]] = value_list[0]
    elif len(key_list) > 1 and len(value_list) > 1:
        # There is one value for each key. Check that the lists are the same
        # length
        if len(key_list) != len(value_list):
            msg = """Number of AstroData objects in key_list does not match
            with the number of AstroData objects in value_list. Please provide
            lists containing the same number of AstroData objects. Please
            supply either a single AstroData object in value_list to be applied
            to all AstroData objects in key_list OR the same number of
            AstroData objects in value_list as there are in key_list"""
            raise Errors.InputError(msg)
        while i in range (0, len(key_list)):
            ret_dict[key_list[i]] = value_list[i]
    return ret_dict

def mark_history(adinput=None, keyword=None):
    """
    The function to use near the end of a python user level function to 
    add a history_mark timestamp to all the outputs indicating when and what
    function was just performed on them, then logging the new historyMarkKey
    PHU key and updated 'GEM-TLM' key values due to history_mark.
    
    Note: The GEM-TLM key will be updated, or added if not in the PHU yet, 
    automatically everytime wrapUp is called.
    
    :param adinput: List of astrodata instance(s) to perform history_mark 
                      on.
    :type adinput: Either a single or multiple astrodata instances in a 
                     list.
    
    :param keyword: The PHU header key to write the current UT time 
    :type keyword: Under 8 character, all caps, string.
                          If None, then only 'GEM-TLM' is added/updated.
    """
    # Instantiate the log
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Loop over each input AstroData object in the input list
    for ad in adinput:
        # Add the 'GEM-TLM' keyword (automatic) and the keyword specified by
        # the 'keyword' parameter to the PHU. If 'keyword' is None,
        # history_mark will still add the 'GEM-TLM' keyword
        ad.history_mark(key=keyword, stomp=True)
        
        log.fullinfo('*'*50, category='header')
        log.fullinfo('File = %s' % ad.filename, category='header')
        log.fullinfo('~'*50, category='header')
        log.fullinfo("PHU keyword GEM-TLM = %s updated" \
                     % (ad.phu_get_key_value('GEM-TLM')),
                     category='header')
        # Only log the following message if the 'keyword' parameter was defined
        if keyword is not None:
            log.fullinfo("PHU keyword %s = %s updated" \
                         % (keyword, ad.phu_get_key_value(keyword)),
                         category='header')
        log.fullinfo('-'*50, category='header')
        
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

def update_key_value(adinput=None, function=None, value=None, extname=None):
    """
    This function updates keywords in the headers of the input dataset,
    performs logging of the changes and writes history keyword related to the
    changes to the PHU.
    
    :param adinput: astrodata instance to perform header key updates on
    :type adinput: an AstroData instance
    
    :param function: string for an astrodata function or descriptor to 
                         perform on the input ad.
                         ie. for ad.count_exts('SCI'), 
                         function='count_exts('SCI')'
    :type function: string 
    
    :param extname: Set to 'PHU', 'SCI', 'VAR' or 'DQ' to update the given
                    keyword in the PHU, SCI, VAR or DQ extension, respectively.
                    
    :type extname: string
    """
    log = gemLog.getGeminiLog()
    historyComment = None
    keyAndCommentDict = {
        'bunit':['BUNIT', 'Physical units of the array values'],
        'count_exts("SCI")':['NSCIEXT', 'Number of science extensions'],
        'dispersion_axis()':['DISPAXIS','Dispersion axis'],
        'filter_name(stripID=True, pretty=True)':
            ['FILTER', 'Combined filter name'],
        'gain()':['GAIN', 'Gain [electrons/ADU]'],
        'gain_setting()':['GAINSET', 'Gain setting (low / high)'],
        'non_linear_level()':['NONLINEA', 'Non-linear regime [ADU]'],
        'numext':['NEXTEND', 'Number of extensions'],
        'pixel_scale()':['PIXSCALE', 'Pixel scale [arcsec/pixel]'],
        'read_noise()':['RDNOISE', 'Estimated read noise [electrons]'],
        'saturation_level()':['SATLEVEL', 'Saturation level [ADU]'],
        # store_original_name() actually all ready writes to the PHU, but
        # doubling it doesn't hurt.
        'store_original_name()':
            ['ORIGNAME', 'Original filename prior to processing'],
                        }
    # Extract key and comment for input function from above dict
    if function not in keyAndCommentDict:
        raise Errors.Error("Unknown value for the 'function' parameter")
    else:
        key = keyAndCommentDict[function][0]
        comment = keyAndCommentDict[function][1]
    
    if extname == "PHU":
        # Check to see whether the keyword is already in the PHU
        original_value = adinput.phu_get_key_value(key)
        if original_value is not None:
            # The keyword exists, so store a history comment for later use
            log.debug("Keyword %s = %s already exists in the PHU" \
                  % (key, original_value))
            comment = '(UPDATED) %s' % comment
            msg = "updated"
            historyComment = "The keyword %s = %s was overwritten in the " \
                             "PHU by AstroData" % (key, original_value)
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
        log.debug ("Original value = %s, Output value = %s" \
                  % (original_value, output_value))
        if output_value is not None:
            if output_value != original_value:
                # Update the header and write a history comment
                adinput.phu_set_key_value(key, str(output_value), comment)
                log.info("PHU keyword %s = %s %s" \
                         % (key, adinput.phu_get_key_value(key), msg),
                         category='header')
                if original_value is None:
                    # A new keyword was written to the PHU. Update the
                    # historyComment accordingly.
                    historyComment = "New keyword %s = %s was written to " \
                                     "the PHU by AstroData" \
                                     % (key, adinput.phu_get_key_value(key))
                else:
                    historyComment = "The keyword %s = %s was overwritten " \
                                     "with a new value of %s in the PHU by " \
                                     "AstroData" \
                                     % (key, original_value,
                                        adinput.phu_get_key_value(key))
                adinput.get_phuheader().add_history(historyComment)
                log.fullinfo(historyComment, category="history")
            else:
                # The keyword value in the pixel data extension is the same
                # as the new value just determined.
                log.info("PHU keyword %s = %s already exists" \
                         % (key, adinput.phu_get_key_value(key)),
                         category="header")
        else:
            log.info("No value found for keyword %s" % (key))
    else:
        if extname is None:
            extname = "SCI"
        # Get the PHU here so that we can write history to the PHU in the loop
        # below
        phu = adinput.get_phuheader()
        for ext in adinput[extname]:
            # Check to see whether the keyword is already in the pixel data
            # extension 
            original_value = ext.get_key_value(key)
            if original_value is not None:
                # The keyword exists, so store a history comment for later use
                log.debug("Keyword %s = %s already in extension %s,%s" \
                          % (key, original_value, extname, ext.extver()))
                comment = '(UPDATED) %s' % comment
                msg = "updated"
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
            log.debug ("Original value = %s, Output value = %s" \
                       % (original_value, output_value))
            if output_value is not None:
                if output_value != original_value:
                    # Update the header and write a history comment
                    ext.set_key_value(key, str(output_value), comment)
                    log.info("%s,%s keyword %s = %s %s" \
                             % (extname, ext.extver(), key,
                                ext.get_key_value(key), msg),
                             category="header")
                    if original_value is None:
                        # A new keyword was written to the pixel data
                        # extension. Update the historyComment accordingly.
                        historyComment = "New keyword %s = %s was written " \
                                         "to extension %s,%s by AstroData" \
                                         % (key, ext.get_key_value(key),
                                            extname, ext.extver())
                    else:
                        historyComment = "The keyword %s = %s was " \
                                         "overwritten with a new value of " \
                                         "%s in extension %s,%s by AstroData" \
                                         % (key, original_value,
                                         ext.get_key_value(key), extname,
                                         ext.extver())

                    phu.add_history(historyComment)
                    log.fullinfo(historyComment, category="history")
                else:
                    # The keyword value in the pixel data extension is the same
                    # as the new value just determined.
                    log.info("%s,%s keyword %s = %s already exists" \
                             % (extname, ext.extver(), key,
                                ext.get_key_value(key)), category="header")
            else:
                log.info("No value found for keyword %s" % (key))

def validate_input(adinput=None):
    """
    The validate_input helper function is used to validate the inputs given to
    the user level functions.
    """
    # If the adinput is None, raise an exception
    if adinput is None:
        raise Errors.InputError("The adinput cannot be None")
    # If the adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # If the adinput is an empty list, raise an exception
    if len(adinput) == 0:
        raise Errors.InputError("The adinput cannot be an empty list")
    # Now, adinput is a list that contains one or more AstroData objects
    return adinput
