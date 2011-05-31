# This module contains user level functions related to the preprocessing of
# the input dataset with a dark frame

import sys
from astrodata import Errors
from astrodata.adutils import gemLog
from gempy import geminiTools as gt

def subtract_dark(adinput=None, dark=None):
    """
    This function will subtract the science extension of the input dark frames
    from the science extension of the input science frames. The variance and
    data quality extension will be updated, if they exist.
    
    This is all conducted in pure Python through the arith 'toolbox' of 
    astrodata. 
       
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adinput: Astrodata input science data
    :type adinput: Astrodata
    
    :param dark: The dark(s) to be added to the input(s). The darks can be a
                 list of AstroData objects or a single AstroData object.
                 Note: If there are multiple inputs and one dark provided, 
                 then the same dark will be applied to all inputs; else the 
                 darks list must match the length of the inputs.
    :type dark: AstroData
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(input=adinput)
    dark = gt.validate_input(input=dark)
    # Create a dictionary that has the AstroData objects specified by adinput
    # as the key and the AstroData objects specified by dark as the value
    dark_dict = gt.make_dict(key_list=adinput, value_list=dark)
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "SUBDARK"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the subtract_dark user level function has been
            # run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "subtract_dark" % (ad.filename))
            # Subtract the dark from the adinput
            dark = dark_dict[ad]
            log.info("Subtracting the dark (%s) from the input AstroData " \
                     "object %s" % (dark.filename, ad.filename))
            ad = ad.sub(dark)
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
