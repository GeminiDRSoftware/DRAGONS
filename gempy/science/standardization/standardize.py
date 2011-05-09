# This module contains user level functions related to standardizing the input
# dataset

import sys
from copy import deepcopy
from astrodata import Errors
from gempy import geminiTools as gt
from gempy import managers as mgr

def standardize_headers_gemini(adinput=None, output_names=None, suffix=None):
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
    
    :param output_names: filenames of output(s)
    :type output_names: String, either a single or a list of strings of same 
                    length as input.
    
    :param suffix: string to add on the end of the input filenames 
                   (or output_names if not None) for the output filenames.
    :type suffix: string
    
    """
    # Perform checks on the input AstroData instances specified by the
    # "adinput" parameter, determine the name of the output AstroData
    # instances using the "output_names" and "suffix" parameters, and
    # instantiate the log using the ScienceFunctionManager
    sfm = mgr.ScienceFunctionManager(adinput=adinput,
                                     output_names=output_names,
                                     suffix=suffix)
    adinput_list, output_names_list, log = sfm.startUp()
    try:
        # Define the keyword to be used for the time stamp for this user level
        # function
        timestampkey = "STDHDRSG"
        # Initialize output_names_list counter and output object list
        count = 0
        output_list = []
        # Loop over each input object in the input list
        for ad in adinput_list:
            # Create the output object by making a "deep copy" of the input
            # object.
            output = deepcopy(ad)
            # Check whether validate_data_f2 has been run on the data
            # before
            if output.phu_get_key_value(timestampkey):
                log.warning("%s has already had its headers standardized " +
                            "with Gemini specific keywords" \
                            % (output.filename))
            else:
                # Update keywords in the headers of each input that are common
                # to all Gemini data
                # Formatting so logger looks organized for these messages
                log.fullinfo("*"*50, category="header")
                log.fullinfo("file = %s" % output.filename, category="header")
                log.fullinfo("~"*50, category="header")
                # Number of science extensions
                gt.updateKeyValue(adinput=output,
                                  function="count_exts(\"SCI\")",
                                  extname="PHU")
                # Original name
                gt.updateKeyValue(adinput=output,
                                  function="store_original_name()",
                                  extname="PHU")
                # Number of extensions
                gt.updateKeyValue(adinput=output,
                                  function="len(output)",
                                  extname="PHU")
                log.fullinfo("-"*50, category="header")
                # Non linear level
                gt.updateKeyValue(adinput=output,
                                  function="non_linear_level()",
                                  extname="SCI")
                # Saturation level
                gt.updateKeyValue(adinput=output,
                                  function="saturation_level()",
                                  extname="SCI")
                # Physical units
                gt.updateKeyValue(adinput=output,
                                  function="bunit",
                                  value="adu",
                                  extname="SCI")
                log.fullinfo("-"*50, category="header") 
            # Set the output file name of the output object
            output.filename = output_names_list[count]
            count += 1
            log.info("Setting the output filename to %s" % output.filename)
            # Append the output object to the output list
            output_list.append(output)
            # Add the appropriate time stamps to the PHU
            sfm.markHistory(adOutputs=output, historyMarkKey=timestampkey)
            sfm.markHistory(adOutputs=output, historyMarkKey="PREPARE")
        # Return the output list
        return output_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def standardize_headers_gmos(adinput=None, output_names=None, suffix=None):
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
    
    :param output_names: filenames of output(s)
    :type output_names: String, either a single or a list of strings of same 
                    length as input.
    
    :param suffix: string to add on the end of the input filenames 
                   (or output_names if not None) for the output filenames.
    :type suffix: string
    """
    # Perform checks on the input AstroData instances specified by the
    # "adinput" parameter, determine the name of the output AstroData
    # instances using the "output_names" and "suffix" parameters, and
    # instantiate the log using the ScienceFunctionManager
    sfm = mgr.ScienceFunctionManager(adinput=adinput,
                                     output_names=output_names,
                                     suffix=suffix)
    adinput_list, output_names_list, log = sfm.startUp()
    try:
        # Define the keyword to be used for the time stamp for this user level
        # function
        timestampkey = "STDHDRSI"
        # Initialize output_names_list counter and output object list
        count = 0
        output_list = []
        # Update keywords in the headers of each input that are common to all
        # Gemini data
        intermediate_list = standardize_headers_gemini(
            adinput=adinput,
            output_names=output_names,
            suffix=suffix)
        # Loop over each input object in the input list
        for ad in intermediate_list:
            # Create the output object by making a "deep copy" of the input
            # object.
            output = deepcopy(ad)
            # Check whether validate_data_f2 has been run on the data
            # before
            if output.phu_get_key_value(timestampkey):
                log.warning("%s has already had its headers standardized " +
                            "with GMOS specific keywords" \
                            % (output.filename))
            else:
                # Update keywords in the headers of each input that are
                # specific common to GMOS data
                log.status("Updating GMOS specific headers")
                # Formatting so logger looks organized for these messages
                log.fullinfo("*"*50, category="header") 
                log.fullinfo("file = "+output.filename, category="header")
                log.fullinfo("~"*50, category="header")
                # Pixel scale
                gt.updateKeyValue(adinput=output,
                                  function="pixel_scale()",
                                  extname="SCI")
                # Read noise
                gt.updateKeyValue(adinput=output,
                                  function="read_noise()",
                                  extname="SCI")
                # Gain
                gt.updateKeyValue(adinput=output,
                                  function="gain()",
                                  extname="SCI")
                if "GMOS_IMAGE" not in output.get_types():
                    gt.updateKeyValue(adinput=output,
                                      function="dispersion_axis()",
                                      extname="SCI")
                log.fullinfo("-"*50, category="header")
            # Set the output file name of the output object
            output.filename = output_names_list[count]
            count += 1
            log.info("Setting the output filename to %s" % output.filename)
            # Append the output object to the output list
            output_list.append(output)
            # Add the appropriate time stamps to the PHU
            sfm.markHistory(adOutputs=output, historyMarkKey=timestampkey)
            sfm.markHistory(adOutputs=output, historyMarkKey="PREPARE")
        # Return the output list
        return output_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def standardize_headers_f2(adinput=None, output_names=None, suffix=None):
    """
    This user level function is used to update headers of FLAMINGOS-2 data.
    """
    try:
        # Perform checks on the input AstroData instances specified by the
        # "adinput" parameter, determine the name of the output AstroData
        # instances using the "output_names" and "suffix" parameters, and
        # instantiate the log using the ScienceFunctionManager
        sfm = mgr.ScienceFunctionManager(adinput=adinput,
                                         output_names=output_names,
                                         suffix=suffix)
        adinput_list, output_names_list, log = sfm.startUp()
        # Define the keyword to be used for the time stamp for this user level
        # function
        timestampkey = "STDHDRSI"
        # Initialize output_names_list counter and output object list
        count = 0
        output_list = []
        # Update keywords in the headers of each input that are common to all
        # Gemini data
        intermediate_list = standardize_headers_gemini(
            adinput=adinput,
            output_names=output_names,
            suffix=suffix)
        # Loop over each input object in the input list
        for ad in intermediate_list:
            # Create the output object by making a "deep copy" of the input
            # object.
            output = deepcopy(ad)
            # Check whether validate_data_f2 has been run on the data
            # before
            if output.phu_get_key_value(timestampkey):
                log.warning("%s has already had its headers standardized " +
                            "with FLAMINGOS-2 specific keywords" \
                            % (output.filename))
            else:
                # Update keywords in the headers of each input that are
                # specific common to FLAMINGOS-2 data
                log.info("No header updates required for FLAMINGOS-2")
            # Set the output file name of the output object
            output.filename = output_names_list[count]
            count += 1
            log.info("Setting the output filename to %s" % output.filename)
            # Append the output object to the output list
            output_list.append(output)
            # Add the appropriate time stamps to the PHU
            sfm.markHistory(adOutputs=output, historyMarkKey=timestampkey)
            sfm.markHistory(adOutputs=output, historyMarkKey="PREPARE")
        # Return the output list
        return output_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def standardize_structure_gmos(adinput=None, output_names=None, suffix=None,
                               addMDF=False, mdfFiles=None):
    """
    This function ensures the MEF structure of GMOS data is ready for further 
    processing, through adding an MDF if necessary.  Appropriately all SPECT
    type data should have an MDF added, while that of IMAGE should not.  If 
    input contains mixed types of GMOS data (ie. some IMAGE and some SPECT), 
    then only those of type SPECT will have MDFs attached.  The MDF to add can 
    be indicated by providing its filename in the MASKNAME PHU key, or the 
    mdfFiles parameter.
    This function is called by standardizeInstrumentStructure in both the GMOS 
    and GMOS_IMAGE primitives sets to perform their work.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adinput: Astrodata inputs to have their headers standardized
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param addMDF: A flag to turn on/off appending the appropriate MDF 
                   file to the inputs.
    :type addMDF: Python boolean (True/False)
                  default: True
                  
    :param mdfFiles: A file name (with path) of the MDF file to append onto the
                     input(s).
                     Note: If there are multiple inputs and one mdfFiles  
                     provided, then the same MDF will be applied to all inputs;
                     else the mdfFiles must be in a list of match the length of
                     the inputs and the inputs must ALL be of type SPECT.
    :type mdfFiles: String, or list of strings.
    
    :param output_names: Filenames of output(s)
    :type output_names: String, either a single or a list of strings of same 
                    length as input.
    
    :param suffix: String to add on the end of the input filenames 
                   (or output_names if not None) for the output filenames.
    :type suffix: string
    """
    # Instantiate ScienceFunctionManager object
    sfm = mgr.ScienceFunctionManager(adinput, output_names, suffix, 
                                      funcName="standardize_headers_gmos")
    # Perform start up checks of the inputs, prep/check of output_names, and get log
    adinput, output_names, log = sfm.startUp()
    try:
        # Set up counter for looping through output_names lists during renaming
        count=0
        
        # Creating empty list of ad"s to be returned that will be filled below
        adOutputs=[]
        
        for ad in adinput:
            if addMDF:
                # Ensuring data is not of type IMAGE, as they do not require an MDF
                if "IMAGE" in ad.get_types():
                    # if of type IMAGE then log critical message and pass input 
                    # to the outputs without looking up and appending an MDF
                    log.critical("Input "+ad.filename+" is an IMAGE and "+
                                    "should not have an MDF attached. Thus, "+
                                    "standardize_structure_gmos will pass "+
                                    "this input to the outputs unchanged.")
                    # Passing the input to be an output without appending any MDF
                    adOut = deepcopy(ad)
                else:
                    log.status("Starting to hunt for matching MDF file")
                    # Input is not of type IMAGE so look up and append the right MDF
                    phuMDFkey = ad.phu_get_key_value("MASKNAME")
                    # check if this key exists in the PHU or if
                    # the input is of type IFU.  If not there, only use the 
                    # provided MDF through the mdfFiles param, if = "IFU-*" then 
                    # convert that to a filename.
                    if (phuMDFkey is None):
                        # it isn"t in the PHU, so only use specified one, 
                        # ensuring to take get the right one from mdfFiles
                        if isinstance(mdfFiles,list):
                            if len(mdfFiles)>1:
                                MDFfilename = mdfFiles[count]
                            elif len(mdfFiles)==1:
                                MDFfilename = mdfFiles[0]
                            else:
                                # mdfFiles is an empty list so msg and raise
                                log.critical("Input "+ad.filename+" has no "+
                                "MASKNAME key in its PHU and no mdfFiles is "+
                                "an empty list.")
                                raise ScienceError("mdfFiles was an empty "+
                                "list so no suitible MDF could be found for "+
                                "input "+ad.filename)
                        elif isinstance(mdfFiles, str):
                            MDFfilename = mdfFiles
                        else:
                            # Provided mdfFiles is not a string or list of them
                            # so make critical msg and raise ScienceError
                            log.critical("The MASKNAME key did not exist in "+
                                "the PHU of "+ad.filename+" and the mdfFiles "+
                                "provided was of type "+repr(type(mdfFiles))+
                                " and it MUST be a string or a list of them.")
                            raise ScienceError("Input "+ad.filename+" had no "+
                                            "MASKNAME key in the PHU and the "+
                                            "mdfFiles provided was invalid.")
                            
                    if (phuMDFkey is not None) and ("IFU" in ad.get_types()):
                        # The input is of type IFU, so the value for the 
                        # MASKNAME PHU key needs to be used to find the 
                        # appropriate MDF filename
                        if "GMOS-S" in ad.get_types():
                            mdfPrefix = "gsifu_"
                        if "GMOS-N" in ad.get_types():
                            mdfPrefix = "gnifu_"
                        if phuMDFkey=="IFU-2":
                            MDFfilename = mdfPrefix+"slits_mdf.fits"
                        if phuMDFkey=="IFU-B":
                            MDFfilename = mdfPrefix+"slitb_mdf.fits"
                        if phuMDFkey=="IFU-R":
                            MDFfilename = mdfPrefix+"slitr_mdf.fits"
                        if phuMDFkey=="IFU-NS-2":
                            MDFfilename = mdfPrefix+"ns_slits_mdf.fits"
                        if phuMDFkey=="IFU-NS-B":
                            MDFfilename = mdfPrefix+"ns_slitb_mdf.fits"
                        if phuMDFkey=="IFU-NS-R":
                            MDFfilename = mdfPrefix+"ns_slitr_mdf.fits"    
                            
                    else:
                        # There was a value for MASKNAME in the PHU and the 
                        # input is not of IFU type, so ensure it has a .fits at
                        # the end and then use it
                        if isinstance(phuMDFkey,str):
                            if phuMDFkey[-5:]==".fits":
                                MDFfilename = phuMDFkey
                            else:
                                MDFfilename = phuMDFkey+".fits"
                    
                    # First check if file is in the current working directory
                    if os.path.exists(MDFfilename):
                        MDF = AstroData(MDFfilename)
                    # If not there, see if it is in lookups/GMOS/MDF dir
                    elif os.path.exists(lookupPath("Gemini/GMOS/MDF/"+
                                                   MDFfilename)):
                        MDF = AstroData(lookupPath("Gemini/GMOS/MDF/"+
                                                   MDFfilename))
                    else:
                        log.critical("MDF file "+MDFfilename+" was not found "+
                                        "on disk.")
                        raise ScienceError("MDF file "+MDFfilename+" was not "+
                                            "found on disk.")
                        
                    # log MDF file being used for current input ad    
                    log.status("MDF, "+MDF.filename+", was found for input, "+
                                                                    ad.filename)
        
                    # passing the found single MDF for the current input to add_mdf
                    # NOTE: This is another science function, so it performs the normal
                    #       deepcopy and filename handling that would normally go here.
                    log.debug("Calling add_mdf to append the MDF")
                    adOuts = add_mdf(adinput=ad, MDFs=MDF, 
                                     output_names=output_names[count])
                    # grab the single output in the list as only one went in
                    adOut = adOuts[0]
                    log.status("Input ,"+adOut.filename+", successfully had "+
                                                        "its MDF appended on.")
            else:
                # addMDF=False, so just pass the inputs through without 
                # bothering with looking up or attaching MDFs no matter what 
                # type the inputs are.
                log.status("addMDF was set to False so Input "+ad.filename+
                           " was just passed through to the outputs.")
                adOut = ad
                
            # Updating GEM-TLM (automatic), STDSTRUC and PREPARE time stamps to 
            # the PHU and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey="STDSTRUC")
            sfm.markHistory(adOutputs=adOut, historyMarkKey="PREPARE")
            # This one shouldn"t be needed, but just adding it just in case 
            sfm.markHistory(adOutputs=adOut, historyMarkKey="GPREPARE")
    
            # renaming the output ad filename
            adOut.filename = output_names[count]
            
            log.status("File name updated to "+adOut.filename+"\n")
                
            # Appending to output list
            adOutputs.append(adOut)
    
            count=count+1
        
        log.status("**FINISHED** the standardize_structure_gmos function")
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 

def standardize_structure_f2(adinput=None, output_names=None, suffix=None):
    try:
        # Perform checks on the input AstroData instances specified by the
        # "adinput" parameter, determine the name of the output AstroData
        # instances using the "output" and "suffix" parameters, and
        # instantiate the log using the ScienceFunctionManager
        sfm = mgr.ScienceFunctionManager(adinput=adinput,
                                         output_names=output,
                                         suffix=suffix,
                                         funcName="standardize_headers_f2")
        adinput, output, log = sfm.startUp()
        print "HELLO, WORLD!"
    except:
        print "boo"
