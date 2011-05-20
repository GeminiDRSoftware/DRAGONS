# This module contains user level functions related to validating the input
# dataset

import sys
from copy import deepcopy
from astrodata import Errors
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy import managers as mgr

def validate_data_f2(adinput=None, repair=False):
    """
    This user level function is used to validate FLAMINGOS-2 data.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "VALDATA"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the validate_data_f2 user level function has been
            # run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "validate_data_f2" % (ad.filename))
            # Validate the input AstroData object. ACTUALLY DO SOMETHING HERE?
            log.info("No validation required for FLAMINGOS-2")
            # Add the appropriate time stamps to the PHU
            gt.markHistory(adinput=ad, keyword=keyword)
            gt.markHistory(adinput=ad, keyword="PREPARE")
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def validate_data_gmos(adinput=None, repair=False):
    """
    This user level function is used to validate GMOS data. It will ensure the
    data is not corrupted or in an odd format that will affect later steps in
    the reduction process. It currently just checks if there are 1, 3, 6 or 12
    SCI extensions in the input. If there are issues with the data, the flag
    'repair' can be used to turn on the feature to repair it or not.
    
    This user level function is called by validateData.
    
    Either a 'main' type logger object, if it exists, or a null logger (i.e.,
    no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adinput: A list containing one or more AstroData instances to be
                  validated
    :type adinput: Astrodata
    
    :param repair: Set to True (the default) to repair the data 
                   Note: this feature does not work yet.
    :type repair: Python boolean
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "VALDATA"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the validate_data_gmos user level function has been
            # run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "validate_data_gmos" % (ad.filename))
            # Validate the input AstroData object. Ensure that the input have
            # 1, 3, 6 or 12 extensions
            numext = ad.count_exts("SCI")
            if (numext != 1 and numext != 3 and numext != 6 and numext != 12):
                if repair:
                    # This would be where we would attempt to repair the data 
                    log.warning("Currently, the 'repair' parameter does " +
                                "not work. Please come back later.")
                else:
                    raise Errors.Error("The number of extensions in %s do " +
                                       "match with the number of extensions " +
                                       "expected in raw GMOS data." \
                                       % ad.filename)
            else:
                log.info("The GMOS input file has been validated: %s " \
                         "contains %d extensions" % (ad.filename, numext))
            # Add the appropriate time stamps to the PHU
            gt.markHistory(adinput=ad, keyword=keyword)
            gt.markHistory(adinput=ad, keyword="PREPARE")
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
