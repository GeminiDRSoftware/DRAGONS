# This module contains user level functions related to the preprocessing of
# the input dataset with a flat frame

import sys
import numpy as np
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict

def divide_by_flat(adinput=None, flat=None):
    """
    The divide_by_flat user level function will divide the science extension of
    the input science frames by the science extension of the input flat frames.
    The variance and data quality extension will be updated, if they exist.
    
    This is all conducted in pure Python through the arith 'toolbox' of 
    astrodata. 
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adinput: Astrodata inputs to have DQ extensions added to
    :type adinput: Astrodata
    
    :param flat: The flat(s) to divide the input(s) by. The flats can be a list
                 of AstroData objects or a single AstroData object. Note: If
                 there is multiple inputs and one flat provided, then the same
                 flat will be applied to all inputs; else the flats list must
                 match the length of the inputs.
    :type flat: AstroData
    """
    
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    flat = gt.validate_input(adinput=flat)
    
    # Create a dictionary that has the AstroData objects specified by adinput
    # as the key and the AstroData objects specified by flat as the value
    flat_dict = gt.make_dict(key_list=adinput, value_list=flat)
    
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "DIVFLAT"
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the divide_by_flat user level function has been
            # run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "divide_by_flat" % (ad.filename))
            
            # Divide the adinput by the flat
            log.info("Dividing the input AstroData object (%s) "\
                     "by this flat:\n%s" % (ad.filename, 
                                               flat_dict[ad].filename))
            ad = ad.div(flat_dict[ad])

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=keyword)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Return the list of output AstroData objects
        return adoutput_list
    
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def normalize_image(adinput=None):
    """
    The normalize_image user level function will normalize each science
    extension of the input AstroData object(s) and automatically update the
    variance and data quality extensions, if they exist.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adinput: Astrodata input flat(s) to be combined and normalized
    :type adinput: Astrodata
    """
    
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "NORMLIZE"
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the normalize_image user level function has
            # been run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "normalize_image" % (ad.filename))
            
            # Loop over each science extension in each input AstroData object
            for ext in ad["SCI"]:

                # Normalise the input AstroData object. Calculate the mean
                # value of the science extension
                mean = np.mean(ext.data)
                # Divide the science extension by the mean value of the science
                # extension
                log.info("Normalizing %s[%s,%d] by dividing by the mean = %f" \
                         % (ad.filename, ext.extname(), ext.extver(), mean))
                ext = ext.div(mean)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=keyword)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Return the list of output AstroData objects
        return adoutput_list
    
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def normalize_flat_image_gmos(adinput=None, saturation="45000"):
    """
    This function will combine the input flats (adinput) and then normalize
    them using the CL script giflat.
    
    WARNING: The giflat script used here replaces the previously 
    calculated DQ frames with its own versions. This may be corrected 
    in the future by replacing the use of the giflat
    with a Python routine to do the flat normalizing.
    
    NOTE: The inputs to this function MUST be prepared.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adinput: Astrodata input flat(s) to be combined and normalized
    :type adinput: Astrodata

    :param saturation: Defines saturation level for the raw frame, in ADU
    :type saturation: string, can be 'default', or a number (default
                      value for this function is '45000')

    
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "NORMFLAT"

    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Load PyRAF
        pyraf, gemini, yes, no = pyrafLoader()

        # Use the CL manager to get the input parameters
        clm = mgr.CLManager(imageIns=adinput, funcName="normalizeFlat",
                            suffix="_out", combinedImages=True, log=log)
        if not clm.status:
            raise Errors.InputError("Please provide prepared inputs")

        # Get the input parameters for IRAF as specified by the stackFrames
        # primitive 
        clPrimParams = {
            # Retrieving the inputs as a list from the CLManager
            "inflats" : clm.imageInsFiles(type="listFile"),
            # Maybe allow the user to override this in the future
            "outflat" : clm.imageOutsFiles(type="string"),
            # This returns a unique/temp log file for IRAF
            "logfile" : clm.templog.name,
            "reject"  : "none",
            "fl_over" : no,
            "fl_trim" : no,
            }

        # Get the input parameters for IRAF as specified by the user
        fl_vardq = no
        fl_dqprop = no
        for ad in adinput:
            if ad["DQ"]:
                fl_dqprop = yes
                if ad["VAR"]:
                    fl_vardq = yes

        # check units of file -- if electrons, convert the saturation
        # parameter from ADU to electrons
        ele_saturation = None
        if saturation == "default":
            saturation = 65000.0
        else:
            saturation = float(saturation)
        for sciext in ad['SCI']:
            bunit = sciext.get_key_value('BUNIT')
            if bunit=='electron':
                gain = sciext.gain().as_pytype()
                conv_sat = saturation * gain
                if ele_saturation is None:
                    ele_saturation = conv_sat
                elif conv_sat < ele_saturation:
                    ele_saturation = conv_sat

        if ele_saturation is not None:
            saturation = ele_saturation
            log.fullinfo("Saturation parameter converted to %.2f electrons" %
                         saturation)

        clSoftcodedParams = {
            "fl_vardq"  : fl_vardq,
            "sat"       : saturation,
            }
        # Get the default parameters for IRAF and update them using the above
        # dictionaries
        clParamsDict = CLDefaultParamsDict("giflat")
        clParamsDict.update(clPrimParams)
        clParamsDict.update(clSoftcodedParams)
        # Log the parameters
        gt.logDictParams(clParamsDict)
        # Call giflat
        gemini.giflat(**clParamsDict)
        if gemini.giflat.status:
            raise Errors.OutputError("The IRAF task giflat failed")
        else:
            log.fullinfo("The IRAF task giflat completed sucessfully")
        # Create the output AstroData object by loading the output file from
        # gemcombine into AstroData, remove intermediate temporary files from
        # disk 
        adoutput, junk, junk = clm.finishCL()
        adoutput[0].filename = ad.filename
        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=adoutput[0], keyword=keyword)
        # Return the output AstroData object
        return adoutput
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
