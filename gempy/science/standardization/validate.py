# This module contains user level functions related to validating the input
# dataset

import sys
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from gempy import geminiTools as gt

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

def validate_data_f2(adinput=None, repair=False):
    """
    This user level function is used to validate FLAMINGOS-2 data.
    """
    
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    
    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["validate_data_f2"]
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the validate_data_f2 user level function has been
            # run previously
            if ad.phu_get_key_value(timestamp_key):
                raise Errors.InputError("%s has already been processed by " \
                                        "validate_data_f2" % (ad.filename))
            
            # Validate the input AstroData object. ACTUALLY DO SOMETHING HERE?
            log.stdinfo("No validation required for FLAMINGOS-2")
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            gt.mark_history(adinput=ad, keyword=timestamp_keys["prepare"])
            
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
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    
    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["validate_data_gmos"]
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the validate_data_gmos user level function has been
            # run previously
            if ad.phu_get_key_value(timestamp_key):
                raise Errors.InputError("%s has already been processed by " \
                                        "validate_data_gmos" % (ad.filename))
            
            # Validate the input AstroData object by ensuring that it has
            # 1, 2, 3, 4, 6 or 12 extensions
            valid_num_ext = [1, 2, 3, 4, 6, 12]
            num_ext = ad.count_exts("SCI")
            if num_ext not in valid_num_ext:
                if repair:
                    # This would be where we would attempt to repair the data 
                    log.warning("Currently, the 'repair' parameter does " +
                                "not work. Please come back later.")
                else:
                    raise Errors.Error("The number of extensions in %s do " +
                                       "match with the number of extensions " +
                                       "expected in raw GMOS data." \
                                       % (ad.filename))
            else:
                log.fullinfo("The GMOS input file has been validated: %s " \
                             "contains %d extensions" % (ad.filename, num_ext))
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            gt.mark_history(adinput=ad, keyword=timestamp_keys["prepare"])
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Return the list of output AstroData objects
        return adoutput_list
    
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
