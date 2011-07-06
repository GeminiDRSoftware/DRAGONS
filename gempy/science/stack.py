# This module contains user level functions related to the stacking of the
# input dataset

import sys
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict

def stack_frames(adinput=None, suffix=None, operation="average", 
                 reject_method="none", mask_type="none"):
    """
    This user level function will stack the input AstroData objects. New
    variance extensions are created from the stacked science extensions and the
    data quality extensions are propagated to the output AstroData object.
    
    NOTE: The inputs to this function MUST be prepared.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adinput: Astrodata inputs to be combined
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param operation: type of combining operation to use.
    :type operation: string, options: 'average', 'median'.
    """
    
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    
    # The stack_frames user level function cannot stack one AstroData object.
    # If the adinput list contains a single AstroData object, raise an
    # exception
    if len(adinput) == 1:
        raise Errors.InputError("Cannot stack a single AstroData object")
    
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "STACK"
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:
        # Load PyRAF
        pyraf, gemini, yes, no = pyrafLoader()
        # Use the CL manager to get the input parameters
        clm = mgr.CLManager(imageIns=adinput, funcName="combine",
                            suffix=suffix, combinedImages=True, log=log)
        if not clm.status:
            raise Errors.InputError("Please provide prepared inputs")
        
        # Get the input parameters for IRAF as specified by the stackFrames
        # primitive 
        clPrimParams = {
            # Retrieving the inputs as a list from the CLManager
            "input"   : clm.imageInsFiles(type="listFile"),
            # Maybe allow the user to override this in the future
            "output"  : clm.imageOutsFiles(type="string"),
            # This returns a unique/temp log file for IRAF
            "logfile" : clm.templog.name,
            }
        
        # Get the input parameters for IRAF as specified by the user
        fl_vardq = no
        fl_dqprop = no
        for ad in adinput:
            if ad["DQ"]:
                fl_dqprop = yes
                if ad["VAR"]:
                    fl_vardq = yes
        clSoftcodedParams = {
            "fl_vardq"  : fl_vardq,
            "fl_dqprop" : fl_dqprop,
            "combine"   : operation,
            "reject"    : reject_method,
            "masktype"  : mask_type,
            }
        
        # Get the default parameters for IRAF and update them using the above
        # dictionaries
        clParamsDict = CLDefaultParamsDict("gemcombine")
        clParamsDict.update(clPrimParams)
        clParamsDict.update(clSoftcodedParams)
        # Log the parameters
        gt.logDictParams(clParamsDict)
        # Call gemcombine
        gemini.gemcombine(**clParamsDict)
        if gemini.gemcombine.status:
            raise Errors.OutputError("The IRAF task gemcombine failed")
        else:
            log.info("The IRAF task gemcombine completed sucessfully")
        
        # Create the output AstroData object by loading the output file from
        # gemcombine into AstroData, remove intermediate temporary files from
        # disk 
        adstack, junk, junk = clm.finishCL()
        
        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, keyword=keyword)
        
        # Return the output AstroData object
        return adstack
    
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
