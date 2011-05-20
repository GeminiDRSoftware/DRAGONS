# This module contains user level functions related to standardizing the input
# dataset

import sys
from astrodata import Errors
from astrodata.adutils import gemLog
from gempy import geminiTools as gt

def standardize_headers_f2(adinput=None):
    """
    This user level function is used to update headers of FLAMINGOS-2 data.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "STDHDRSI"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the standardize_headers_f2 user level function has
            # been run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "standardize_headers_f2" \
                                        % (ad.filename))
            # Standardize the headers of the input AstroData object. First,
            # update the keywords in the headers that are common to all Gemini
            # data. The standardize_headers_gemini user level function returns
            # a list, so select the first element.
            ad = standardize_headers_gemini(adinput=ad)[0]
            # Now, update the keywords in the headers that are specific to
            # FLAMINGOS-2
            log.status("Updating keywords that are specific to FLAMINGOS-2")
            # Formatting so logger looks organized for these messages
            log.fullinfo("*"*50, category="header") 
            log.fullinfo("File = %s" % ad.filename, category="header")
            log.fullinfo("~"*50, category="header")
            # Filter name (required for IRAF?)
            gt.update_key_value(adinput=ad,
                                function=
                                "filter_name(stripID=True, pretty=True)",
                                extname="PHU")
            # Pixel scale
            gt.update_key_value(adinput=ad, function="pixel_scale()",
                                extname="PHU")
            # Read noise (new keyword, should it be written?)
            gt.update_key_value(adinput=ad, function="read_noise()",
                                extname="SCI")
            # Gain (new keyword, should it be written?)
            gt.update_key_value(adinput=ad, function="gain()",
                                extname="SCI")
            # Dispersion axis (new keyword, should it be written?)
            if "IMAGE" not in ad.types:
                gt.update_key_value(adinput=ad, function="dispersion_axis()",
                                    extname="SCI")
            log.fullinfo("-"*50, category="header")
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

def standardize_headers_gemini(adinput=None):
    """ 
    This user level function is used by the instrument specific
    standardize_headers user level functions. This function adds the keywords
    NSCIEXT, NEXTEND and ORIGNAME to the PHU of the input dataset and the
    keywords BUNIT, NONLINEA and SATLEVEL to the pixel data extensions of the
    input dataset.
    
    Either a 'main' type logger object, if it exists, or a null logger (i.e.,
    no log file, no messages to screen) will be retrieved/created in the
    ScienceFunctionManager and used within this function.
          
    :param adinput: Astrodata inputs to have their headers standardized
    :type adinput: Astrodata objects, either a single or a list of objects
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "STDHDRSG"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the standardize_headers_gemini user level function
            # has been run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "standardize_headers_gemini" \
                                        % (ad.filename))
            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are common to all Gemini data
            # Formatting so logger looks organized for these messages
            log.fullinfo("*"*50, category="header")
            log.fullinfo("File = %s" % ad.filename, category="header")
            log.fullinfo("~"*50, category="header")
            # Number of science extensions
            gt.update_key_value(adinput=ad, function="count_exts(\"SCI\")",
                                extname="PHU")
            # Original name
            gt.update_key_value(adinput=ad, function="store_original_name()",
                                extname="PHU")
            # Number of extensions
            gt.update_key_value(adinput=ad, function="numext", value=len(ad),
                                extname="PHU")
            log.fullinfo("-"*50, category="header")
            # Non linear level
            gt.update_key_value(adinput=ad, function="non_linear_level()",
                                extname="SCI")
            # Saturation level
            gt.update_key_value(adinput=ad, function="saturation_level()",
                                extname="SCI")
            # Physical units (assuming raw data has units of ADU)
            gt.update_key_value(adinput=ad, function="bunit", value="adu",
                                extname="SCI")
            log.fullinfo("-"*50, category="header") 
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

def standardize_headers_gmos(adinput=None):
    """
    This user level function is used to update and add keywords to the headers
    of the input dataset. First, it calls the standardize_headers_gemini user
    level function to update Gemini specific keywords and then updates GMOS
    specific keywords.
    
    Either a 'main' type logger object, if it exists, or a null logger (i.e.,
    no log file, no messages to screen) will be retrieved/created in the
    ScienceFunctionManager and used within this function.
          
    :param adinput: Astrodata inputs to have their headers standardized
    :type adinput: Astrodata objects, either a single or a list of objects
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "STDHDRSI"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the my_science_step user level function has been
            # run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "my_science_step" % (ad.filename))
            # Standardize the headers of the input AstroData object. First,
            # update the keywords in the headers that are common to all Gemini
            # data. The standardize_headers_gemini user level function returns
            # a list, so select the first element.
            ad = standardize_headers_gemini(adinput=ad)[0]
            # Now, update the keywords in the headers that are specific to GMOS
            log.status("Updating keywords that are specific to GMOS")
            # Formatting so logger looks organized for these messages
            log.fullinfo("*"*50, category="header") 
            log.fullinfo("File = %s" % ad.filename, category="header")
            log.fullinfo("~"*50, category="header")
            # Pixel scale
            gt.update_key_value(adinput=ad, function="pixel_scale()",
                                extname="SCI")
            # Read noise
            gt.update_key_value(adinput=ad, function="read_noise()",
                                extname="SCI")
            # Gain
            gt.update_key_value(adinput=ad, function="gain()",
                                extname="SCI")
            # Dispersion axis
            if "IMAGE" not in ad.types:
                gt.update_key_value(adinput=ad, function="dispersion_axis()",
                                    extname="SCI")
            log.fullinfo("-"*50, category="header")
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

def standardize_structure_f2(adinput=None, add_mdf=False, mdf=None):
    """
    This user level function is used to standardize the structure of
    FLAMINGOS-2 data.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "SDZSTRUC"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the standardize_structure_f2 user level function
            # has been run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "standardize_structure_f2" \
                                        % (ad.filename))
            # Standardize the structure of the input AstroData object.
            # ACTUALLY DO SOMETHING HERE?
            if add_mdf:
                # Check whether the input AstroData object has an AstroData 
                # Type of IMAGE, since MDFs should only be added to
                # spectroscopic data.
                if "IMAGE" in ad.types:
                    raise Errors.InputError("Cannot add an MDF to %s, since " \
                                            "it has an AstroData Type of " \
                                            "'IMAGE'" % (ad.filename))
                # Call the add_mdf_f2 user level function
                ad = add_mdf_f2(adinput=ad, mdf=mdf)
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

def standardize_structure_gmos(adinput=None, add_mdf=False, mdf=None):
    """
    This function ensures the MEF structure of GMOS data is ready for further 
    processing, through adding an MDF if necessary. Appropriately all SPECT
    type data should have an MDF added, while that of IMAGE should not. If 
    input contains mixed types of GMOS data (ie. some IMAGE and some SPECT), 
    then only those of type SPECT will have MDFs attached. The MDF to add can 
    be indicated by providing its filename in the MASKNAME PHU key, or the 
    mdfFiles parameter.
    This function is called by standardizeInstrumentStructure in both the GMOS 
    and GMOS_IMAGE primitives sets to perform their work.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adinput: Astrodata inputs to have their headers standardized
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param add_mdf: A flag to turn on/off appending the appropriate MDF 
                   file to the inputs.
    :type add_mdf: Python boolean (True/False)
                  default: True
                  
    :param mdf: A file name (with path) of the MDF file to append onto the
                     input(s).
                     Note: If there are multiple inputs and one mdf
                     provided, then the same MDF will be applied to all inputs;
                     else the mdf must be in a list of match the length of
                     the inputs and the inputs must ALL be of type SPECT.
    :type mdf: String, or list of strings.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "SDZSTRUC"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the standardize_structure_gmos user level function
            # has been run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "standardize_structure_gmos" \
                                        % (ad.filename))
            # Standardize the structure of the input AstroData object.
            # ACTUALLY DO SOMETHING HERE?
            if add_mdf:
                # Check whether the input AstroData object has an AstroData 
                # Type of IMAGE, since MDFs should only be added to
                # spectroscopic data.
                if "IMAGE" in ad.types:
                    raise Errors.InputError("Cannot add an MDF to %s, since " \
                                            "it has an AstroData Type of " \
                                            "'IMAGE'" % (ad.filename))
                # Call the add_mdf_gmos user level function
                ad = add_mdf_gmos(adinput=ad, mdf=mdf)
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
