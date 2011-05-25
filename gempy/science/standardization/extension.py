# This module contains user level functions related to adding extensions to
# and removing extensions from the input dataset

import sys
import numpy as np
from astrodata import AstroData
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils import varutil
from astrodata.ConfigSpace import lookup_path
from gempy import geminiTools as gt

def add_dq(adinput=None, bpm=None):
    """
    This user level function (ulf) is used to add a DQ extension to the input
    AstroData object. The value of a pixel in the DQ extension will be the sum
    of the following: (0=good, 1=bad pixel (found in bad pixel mask), 2=pixel
    is in the non-linear regime, 4=pixel is saturated). This ulf will trim the
    BPM to match the input AstroData object(s). If only one BPM is provided,
    that BPM will be used to determine the DQ extension for all input AstroData
    object(s). If more than one BPM is provided, the number of BPM AstroData
    objects must match the number of input AstroData objects.

    :param adinput: Astrodata inputs to have a DQ extension added
    :type adinput: Astrodata
    
    :param bpm: The BPM(s) to be used to flag bad pixels in the DQ extension
    :type bpm: AstroData
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "ADDDQ"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the add_dq user level function has been
            # run previously
            if ad.phu_get_key_value(keyword) or ad["DQ"]:
                raise Errors.InputError("%s has already been processed by " \
                                        "add_dq" % (ad.filename))
            # Call the _select_bpm helper function to get the appropriate BPM
            # for the input AstroData object
            bpmfile = _select_bpm(adinput=ad, bpm=bpm)
            # Loop over each science extension in each input AstroData object
            for ext in ad["SCI"]:
                # Get the non-linear level and the saturation level as integers
                # using the appropriate descriptors
                non_linear_level = ext.non_linear_level().as_pytype()
                saturation_level = ext.saturation_level().as_pytype()
                # Create an array that contains pixels that have a value of 2
                # when that pixel is in the non-linear regime in the input
                # science extension
                if non_linear_level is not None:
                    non_linear_array = np.where((ext.data > non_linear_level) \
                        & (ext.data < saturation_level), 2, 0)
                    # Set the data type of the array to be int16
                    non_linear_array = non_linear_array.astype(np.int16)
                # Create an array that contains pixels that have a value of 4
                # when that pixel is saturated in the input science extension
                if saturation_level is not None:
                    saturation_array = np.where(
                        ext.data > saturation_level, 4, 0)
                    # Set the data type of the array to be int16
                    saturation_array = saturation_array.astype(np.int16)
                # Create a single DQ extension from the three arrays (BPM,
                # non-linear and saturated)
                dq_array = np.add(bpmfile.data, non_linear_array,
                                  saturation_array)
                # Create a DQ AstroData object
                dq = AstroData(header=bpmfile.header, data=dq_array)
                # Name the extension appropriately
                dq.rename_ext("DQ", ver=ext.extver())
            # Append the DQ AstroData object to the input AstroData object
            ad.append(moredata=dq)
            log.status("Adding the DQ extension to the input AstroData " \
                       "object %s" % (ad.filename))
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

def add_mdf(adinput=None, mdf=None):
    """
    This function is to attach the MDFs to the inputs as an extension. 
    It is assumed that the MDFs are single extensions fits files and will
    thus be appended as ('MDF',1) onto the inputs.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adinput: Astrodata inputs to have their headers standardized
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param mdf: The MDF(s) to be added to the input(s).
    :type mdf: AstroData objects in a list, or a single instance.
               Note: If there are multiple inputs and one MDF provided, 
               then the same MDF will be applied to all inputs; else the 
               MDFs list must match the length of the inputs.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "ADDMDF"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the add_mdf user level function has been
            # run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "add_mdf" % (ad.filename))
            # Call the _select_mdf helper function to get the appropriate MDF
            # for the input AstroData object
            mdffile = _select_mdf(adinput=ad, mdf=mdf)
            # Append the MDF to the input AstroData object
            ad.append(mdffile)
            log.status("Adding the MDF %s to the input AstroData object %s" \
                       % (mdffile.filename, ad.filename))
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

def add_var(adinput=None):
    """
    This function uses numpy to calculate the variance of each SCI frame
    in the input files and appends it as a VAR frame using AstroData.
    
    The calculation will follow the formula:
    variance = (read noise/gain)2 + max(data,0.0)/gain
    
    NOTE:
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adInputs: Astrodata inputs to have DQ extensions added to
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length
                    as adInputs.
    
    :param suffix: 
        string to add on the end of the input filenames 
        (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "ADDVAR"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the add_var user level function has been
            # run previously
            if ad.phu_get_key_value(keyword) or ad["VAR"]:
                raise Errors.InputError("%s has already been processed by " \
                                        "add_var" % (ad.filename))
            # Loop over each science extension in each input AstroData object
            for ext in ad["SCI"]:
                # The variance extension is determined using
                # (read noise/gain)**2 + max(data,0.0)/gain
                var_array = varutil.calculateInitialVarianceArray(ext)
                # Create the header for the variance extension
                var_header = varutil.createInitialVarianceHeader(
                    extver=ext.extver(), shape=var_array.shape)
                # Create a DQ AstroData object
                var = AstroData(header=var_header, data=var_array)
                # Name the extension appropriately
                var.rename_ext("VAR", ver=ext.extver())
            # Append the DQ AstroData object to the input AstroData object
            ad.append(moredata=var)
            log.status("Adding the DQ extension to the input AstroData " \
                       "object %s" % (ad.filename))
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

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

def _select_bpm(adinput=None, bpm=None):
    """
    This help functions is used to select the appropriate BPM depending on the
    input AstroData object. The returned BPM will have the same dimensions as
    the input AstroData object.
    """
    ## Matching size of BPM array to that of the SCI data array
    # Getting the data section as a int list of form:
    # [y1, y2, x1, x2] 0-based and non-inclusive
    ### This section might need to be upgraded in the future
    ### for more general use instead of just 1x1 and 2x2 imaging
    #if ad[("SCI",1)].get_key_value("CCDSUM")=="1 1":
    #    bpm = BPM_11
    #elif ad[("SCI",1)].get_key_value("CCDSUM")=="2 2":
    #    bpm = BPM_22
    #datsecList = sciExt.data_section().as_pytype()
    #dsl = datsecList
    #datasecShape = (dsl[1]-dsl[0], dsl[3]-dsl[2])
    # Creating a zeros array the same size as SCI array
    # for this extension
    #BPMArrayOut = np.zeros(sciExt.data.shape, 
    #                       dtype=np.int16)
    # Loading up zeros array with data from BPM array
    # if the sizes match then there is no change, else
    # output BPM array will be "padded with zeros" or 
    # "not bad pixels" to match SCI"s size.
    #if BPMArrayIn.shape==datasecShape:
    #    log.fullinfo("BPM data was found to be of a different size "
    #                 "than the SCI, so padding the BPM"s data to "
    #                 "match the SCI.")
    #    BPMArrayOut[dsl[0]:dsl[1], dsl[2]:dsl[3]] = BPMArrayIn
    #elif BPMArrayIn.shape==BPMArrayOut.shape:
    #    BPMArrayOut = BPMArrayIn
    # Extracting the BPM data array for this extension
    #BPMArrayIn = bpmAD.data
    # Extracting the BPM header for this extension to be 
    # later converted to a DQ header
    #dqheader = bpmAD.header
    # Preparing the non linear and saturated pixel arrays
    # Append the BPM file to the input AstroData instance
    # Getting the filename for the BPM and removing any paths
    #bpm_name = os.path.basename(bpm.filename)
    # Extracting the matching DQ extension from the BPM 
    #bpm_data = bpm[("DQ",sciExt.extver())].data
    #if matchSize:
        # Getting the data section as a int list of form:
        # [y1, y2, x1, x2] 0-based and non-inclusive
        #datsecList = sciExt.data_section().as_pytype()
        #dsl = datsecList
        #datasecShape = (dsl[1]-dsl[0], dsl[3]-dsl[2])
        
        # Creating a zeros array the same size as SCI array
        # for this extension
        #BPMArrayOut = np.zeros(sciExt.data.shape, 
        #                       dtype=np.int16)
        # Loading up zeros array with data from BPM array
        # if the sizes match then there is no change, else
        # output BPM array will be "padded with zeros" or 
        # "not bad pixels" to match SCI"s size.
        #if bpm_data.shape == datasecShape:
        #    log.fullinfo("BPM data was found to be of a different "
        #                 "size than the SCI, so padding the BPM's "
        #                 "data to match the SCI.")
        #    BPMArrayOut[dsl[0]:dsl[1], dsl[2]:dsl[3]] = bpm_data
        #elif bpm_data.shape == BPMArrayOut.shape:
        #    BPMArrayOut = bpm_data
    # Don't match size
    #else:
    #    BPMArrayOut = bpm_data
    # Append the BPM file to the input AstroData instance
    # Getting the filename for the BPM and removing any paths
    #bpm_name = os.path.basename(bpm.filename)
    # Extracting the matching DQ extension from the BPM 
    #bpm_data = bpm[("DQ",sciExt.extver())].data
    # Matching size of BPM array to that of the SCI data array
    # Creating a header for the BPM array and updating
    # further updating to this header will take place in 
    # addDQ primitive
    #BPMheader = pf.Header() 
    #BPMheader.update("BITPIX", 16, 
    #                "number of bits per data pixel")
    #BPMheader.update("NAXIS", 2)
    #BPMheader.update("PCOUNT", 0, 
    #                "required keyword; must = 0")
    #BPMheader.update("GCOUNT", 1, 
    #                "required keyword; must = 1")
    #BPMheader.update("BUNIT", "bit", "Physical units")
    #BPMheader.update("BPMFILE", BPMfilename, 
    #                    "Bad Pixel Mask file name")
    #BPMheader.update("EXTVER", sciExt.extver(), 
    #                    "Extension Version")
    # This extension will be renamed DQ in addDQ
    #BPMheader.update("EXTNAME", "BPM", "Extension Name")

    bpm = AstroData(lookup_path(bpm_dict["F2"]))
    bpm.data = np.squeeze(bpm.data)
    bpm.data = bpm.data.astype(np.int16)
    
    return bpm

bpm_dict = {
    "GMOS11": "Gemini/GMOS/BPM/GMOS_BPM_11.fits",
    "GMOS22": "Gemini/GMOS/BPM/GMOS_BPM_22.fits",
    "F2": "Gemini/F2/BPM/F2_bpm.fits",
    }

#def _select_mdf(adinput=None):
    # Renaming the extension"s extname="MDF" and extver=1, even if 
    # they all ready these values just to be sure.
    #mdffile.rename_ext("MDF",1)
    #mdffile.set_key_value("EXTNAME","MDF", "Extension name")
    #mdffile.set_key_value("EXTVER",1,"Extension version")
    # If no MDF is supplied, try to find an appropriate one. First, check
    # the "MASKNAME" keyword
    #maskname = ad.phu_get_key_value("MASKNAME")
    #if maskname is not None:
    #    if "IFU" in ad.types:
    #        # The input AstroData object has an AstroData Type of "IFU".
    #        # Use the value of the MASKNAME keyword to determine the
    #        # appropriate MDF
    #        if "GMOS-S" in ad.types:
    #            mdf_prefix = "gsifu"
    #        if "GMOS-N" in ad.types:
    #            mdf_prefix = "gnifu"
    #        mdf_name = "%s%s" % (mdf_prefix, mdf_dict[maskname])
    #    else:
    #        # The MASKNAME keyword defines the actual name of an MDF
    #        if not maskname.endswith(".fits"):
    #            mdf_name = "%s.fits" % maskname
    #        else:
    #            mdf_name = str(maskname)
    #    # Check if the MDF exists in the current working directory
    #    if os.path.exists(mdf_name):
    #        mdf = AstroData(mdf_name)
    #    # Check if the MDF exists in the gemini_python package
    #    elif os.path.exists(lookup_path("Gemini/GMOS/MDF/%s" % mdf_name)):
    #        mdf = AstroData(lookup_path("Gemini/GMOS/MDF/%s" % mdf_name))
    #    else:
    #        raise Errors.InputError("The MDF file %s was not found " \
    #                                "either in the current working " \
    #                                "directory or in the gemini_python " \
    #                                "package" % (mdf_name))
    # If mdf is a single AstroData object, put it in a list
    #if not isinstance(mdf, list):
    #    mdf = [mdf]
    # Check if the MDF is a single extension fits file
    #for mdffile in mdf:
    #    if len(mdffile) > 1:
    #       raise Errors.InputError("Please provide a single extension fits " \
    #                                "file for the MDF")
    # Check if the input AstroData object already has an MDF
    #for ad in adinput:
    #    if ad["MDF"]:
    #        raise Errors.InputError("Input AstroData object already has an " \
    #                                "MDF attached")
    #mdfdict = {}
    #if len(mdf) > 1:
        # Check whether the number of MDFs match the number of input AstroData
        # objects
    #    if len(adinput) != len(mdf):
    #        raise Errors.InputError("Please supply either a single MDF to " \
    #                               "be applied to all AstroData objects OR " \
    #                                "the same number of MDFs as there are " \
    #                                "input AstroData objects")
    #    else:
            # Create a dictionary where the key is the input AstroData object
            # and the value is the MDF file to be added to the input AstroData
            # object
    #        while i in range (0,len(adinput)):
    #            mdfdict[adinput[i]] = mdf[i]
    # If mdf is a single AstroData object, put it in a list
    #if not isinstance(mdf, list):
    #    mdf = [mdf]
    # Check if the MDF is a single extension fits file
    #for mdffile in mdf:
    #    if len(mdffile) > 1:
    #       raise Errors.InputError("Please provide a single extension fits " \
    #                                "file for the MDF")
    # Check if the input AstroData object already has an MDF
    #for ad in adinput:
    #    if ad["MDF"]:
    #        raise Errors.InputError("Input AstroData object already has an " \
    #                                "MDF attached")
    #mdfdict = {}
    #if len(mdf) > 1:
        # Check whether the number of MDFs match the number of input AstroData
        # objects
    #    if len(adinput) != len(mdf):
    #        raise Errors.InputError("Please supply either a single MDF to " \
    #                               "be applied to all AstroData objects OR " \
    #                                "the same number of MDFs as there are " \
    #                                "input AstroData objects")
    #    else:
            # Create a dictionary where the key is the input AstroData object
            # and the value is the MDF file to be added to the input AstroData
            # object
    #        while i in range (0,len(adinput)):
    #            mdfdict[adinput[i]] = mdf[i]
    # Renaming the extension"s extname="MDF" and extver=1, even if 
    # they all ready these values just to be sure.
    #mdffile.rename_ext("MDF",1)
    #mdffile.set_key_value("EXTNAME","MDF", "Extension name")
    #mdffile.set_key_value("EXTVER",1,"Extension version")
    # Check the inputs
    #if not isinstance(input1, list):
    #    input1 = [input1]
    #if not isinstance(input2, list):
    #    input2 = [input2]
    ## Check if the input AstroData object already has input2 attached
    #for ad in input1:
    #    if ad[input2type]:
    #        raise Errors.InputError("Input AstroData object already has an " \
    #                                "%s attached", input2type)
    #if len(input2) > 1:
    #    # Check whether the number of MDFs match the number of input AstroData
    #    # objects
    #    if len(input1) != len(input2):
    #        raise Errors.InputError("Please supply either a single %s to " \
    #                               "be applied to all AstroData objects OR " \
    #                                "the same number of %ss as there are " \
    #                                "input AstroData objects" \
    #                                % (input2type, input2type))
    #    else:
    #        # Create a dictionary where the key is the input AstroData object
    #        # and the value is the input2 file to be added to the input
    #        # AstroData object
    #        while i in range (0,len(input1)):
    #            dict[input1[i]] = input2[i]

mdf_dict = {
    "IFU-2": "_slits_mdf.fits",
    "IFU-B": "_slitb_mdf.fits",
    "IFU-R": "_slitr_mdf.fits",
    "IFU-NS-2": "_ns_slits_mdf.fits",
    "IFU-NS-B": "_ns_slitb_mdf.fits",
    "IFU-NS-R": "_ns_slitr_mdf.fits",
    }
