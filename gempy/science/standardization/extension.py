# This module contains user level functions related to adding extensions to
# and removing extensions from the input dataset

import sys
from astrodata import Errors
from astrodata.adutils import gemLog
from gempy import geminiTools as gt

def add_bpm(adinput=None, bpm=None):
    """
    This function will add the provided BPM (Bad Pixel Mask) to the inputs.  
    The BPM will be added as frames matching that of the SCI frames and ensure
    the BPM's data array is the same size as that of the SCI data array. If the
    SCI array is larger (say SCI's were overscan trimmed, but BPMs were not), 
    the BPMs will have their arrays padded with zero's to match the sizes and
    use the data_section descriptor on the SCI data arrays to ensure the match
    is a correct fit.  There must be a matching number of DQ extensions in the
    BPM as the input the BPM frames are to be added to 
    (i.e., if input has 3 SCI extensions, the BPM must have 3 DQ extensions).
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adinput: Astrodata inputs to be converted to Electron pixel units
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param bpm: The BPM(s) to be added to the input(s).
    :type bpm: 
       AstroData objects in a list, or a single instance.
       Note: If there are multiple inputs and one BPM provided, then the
       same BPM will be applied to all inputs; else the BPMs list  
       must match the length of the inputs.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Check the bpm parameter
    if bpm is None:
        raise Errors.InputError("Please provide a BPM file")
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "ADDBPM"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput["SCI"]:
            # Check whether the add_bpm user level function has been run
            # previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "add_bpm" % (ad.filename))
            # Get the correct BPM for the AstroData object
            if isinstance(bpm, list):
                if len(bpm)>1:
                    bpm = bpm[count]
                else:
                    bpm = bpm[0]
            else:
                bpm = bpm
            # Append the BPM file to the input AstroData instance
            # Getting the filename for the BPM and removing any paths
            bpm_name = os.path.basename(bpm.filename)
            # Extracting the matching DQ extension from the BPM 
            bpm_data = bpm[("DQ",sciExt.extver())].data
            # Matching size of BPM array to that of the SCI data array
            if matchSize:
                # Getting the data section as a int list of form:
                # [y1, y2, x1, x2] 0-based and non-inclusive
                datsecList = sciExt.data_section().as_pytype()
                dsl = datsecList
                datasecShape = (dsl[1]-dsl[0], dsl[3]-dsl[2])
                
                # Creating a zeros array the same size as SCI array
                # for this extension
                BPMArrayOut = np.zeros(sciExt.data.shape, 
                                       dtype=np.int16)

                # Loading up zeros array with data from BPM array
                # if the sizes match then there is no change, else
                # output BPM array will be "padded with zeros" or 
                # "not bad pixels" to match SCI's size.
                if bpm_data.shape == datasecShape:
                    log.fullinfo("BPM data was found to be of a different "
                                 "size than the SCI, so padding the BPM's "
                                 "data to match the SCI.")
                    BPMArrayOut[dsl[0]:dsl[1], dsl[2]:dsl[3]] = bpm_data
                elif bpm_data.shape == BPMArrayOut.shape:
                    BPMArrayOut = bpm_data
            # Don't match size
            else:
                BPMArrayOut = bpm_data
            
            # Creating a header for the BPM array and updating
            # further updating to this header will take place in 
            # addDQ primitive
            BPMheader = pf.Header() 
            BPMheader.update("BITPIX", 16, 
                            "number of bits per data pixel")
            BPMheader.update("NAXIS", 2)
            BPMheader.update("PCOUNT", 0, 
                            "required keyword; must = 0")
            BPMheader.update("GCOUNT", 1, 
                            "required keyword; must = 1")
            BPMheader.update("BUNIT", "bit", "Physical units")
            BPMheader.update("BPMFILE", BPMfilename, 
                                "Bad Pixel Mask file name")
            BPMheader.update("EXTVER", sciExt.extver(), 
                                "Extension Version")
            # This extension will be renamed DQ in addDQ
            BPMheader.update("EXTNAME", "BPM", "Extension Name")
            
            # Creating an astrodata instance from the 
            # DQ array and header
            bpmAD = AstroData(header=BPMheader, data=BPMArrayOut)
            
            # Using rename_ext to correctly set the EXTVER and 
            # EXTNAME values in the header   
            bpmAD.rename_ext("BPM", ver=sciExt.extver())
            
            # Appending BPM astrodata instance to the input one
            log.debug("Appending new BPM HDU onto the file "+ 
                      adOut.filename)
            adOut.append(bpmAD)
            log.status("Appending BPM complete for "+ adOut.filename)
            # Add the appropriate time stamps to the PHU
            gt.markHistory(adinput=ad, keyword=keyword)
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def add_dq(adInputs):
    """
    This function will create a numpy array for the data quality 
    of each SCI frame of the input data. This will then have a 
    header created and append to the input using AstroData as a DQ 
    frame. The value of a pixel will be the sum of the following: 
    (0=good, 1=bad pixel (found in bad pixel mask), 
    2=value is non linear, 4=pixel is saturated)
    
    NOTE: 
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    NOTE:
    FIND A WAY TO TAKE CARE OF NO BPM EXTENSION EXISTS ISSUE, OR ALLOWING THEM
    TO PASS IT IN...
    
    A string representing the name of the log file to write all log messages to
    can be defined, or a default of 'gemini.log' will be used.  If the file
    all ready exists in the directory you are working in, then this file will 
    have the log messages during this function added to the end of it.
    
    :param adInputs: Astrodata inputs to have DQ extensions added to
    :type adInputs: Astrodata objects, either a single or a list of objects
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
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "add_dq" % (ad.filename))
            # Retrieving BPM extension 
            bpmAD = adOut[("BPM",sciExt.extver())]
            
            # Extracting the BPM data array for this extension
            BPMArrayIn = bpmAD.data
            
            # Extracting the BPM header for this extension to be 
            # later converted to a DQ header
            dqheader = bpmAD.header
            
            ## Matching size of BPM array to that of the SCI data array
            
            # Getting the data section as a int list of form:
            # [y1, y2, x1, x2] 0-based and non-inclusive
            datsecList = sciExt.data_section().as_pytype()
            dsl = datsecList
            datasecShape = (dsl[1]-dsl[0], dsl[3]-dsl[2])
            
            # Creating a zeros array the same size as SCI array
            # for this extension
            BPMArrayOut = np.zeros(sciExt.data.shape, 
                                   dtype=np.int16)

            # Loading up zeros array with data from BPM array
            # if the sizes match then there is no change, else
            # output BPM array will be "padded with zeros" or 
            # "not bad pixels" to match SCI's size.
            if BPMArrayIn.shape==datasecShape:
                log.fullinfo("BPM data was found to be of a different size "
                             "than the SCI, so padding the BPM's data to "
                             "match the SCI.")
                BPMArrayOut[dsl[0]:dsl[1], dsl[2]:dsl[3]] = BPMArrayIn
            elif BPMArrayIn.shape==BPMArrayOut.shape:
                BPMArrayOut = BPMArrayIn
            # Preparing the non linear and saturated pixel arrays
            # and their respective constants
            nonLinArray = np.zeros(sciExt.data.shape, 
                                   dtype=np.int16)
            saturatedArray = np.zeros(sciExt.data.shape, 
                                      dtype=np.int16)
            linear = sciExt.non_linear_level().as_pytype()
            saturated = sciExt.saturation_level().as_pytype()

            if (linear is not None) and (fl_nonlinear): 
                log.debug("Performing an np.where to find "+
                          "non-linear pixels for extension "+
                          str(sciExt.extver())+" of "+
                          adOut.filename)
                nonLinArray = np.where(sciExt.data>linear,2,0)
                log.status("Done calculating array of non-linear"+
                           " pixels")
            if (saturated is not None) and (fl_saturated):
                log.debug("Performing an np.where to find "+
                          "saturated pixels for extension "+
                          str(sciExt.extver())+" of "+
                          adOut.filename)
                saturatedArray = np.where(sciExt.data>saturated,4,0)
                log.status("Done calculating array of saturated"+
                           " pixels") 
            
            # Creating one DQ array from the three
            dqArray=np.add(BPMArrayOut, nonLinArray, saturatedArray) 
            # Updating data array for the BPM array to be the 
            # newly calculated DQ array
            adOut[("BPM",sciExt.extver())].data = dqArray
            
            # Renaming the extension to DQ from BPM
            dqheader.update("EXTNAME", "DQ", "Extension Name")
            
            # Using rename_ext to correctly set the EXTVER and 
            # EXTNAME values in the header   
            bpmAD.rename_ext("DQ", ver=sciExt.extver(), force=True)

            # Logging that the name of the BPM extension was changed
            log.fullinfo("BPM Extension "+str(sciExt.extver())+
                         " of "+adOut.filename+" had its EXTVER "+
                         "changed to "+
                         adOut[("DQ",
                                sciExt.extver())].header["EXTNAME"])
  
            # Add the appropriate time stamps to the PHU
            gt.markHistory(adinput=ad, keyword=keyword)
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def add_mdf_f2(adinput=None, mdf=None):
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
               MDFs list  must match the length of the inputs.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Check whether an MDF was defined
    if mdf is None:
        # If no MDF is supplied, try to find an appropriate one.
        # THIS DOESN'T WORK YET!
        raise Errors.InputError("No MDF supplied")
    # If mdf is a single AstroData object, put it in a list
    if not isinstance(mdf, list):
        mdf = [mdf]
    # Check if the MDF is a single extension fits file
    for mdffile in mdf:
        if len(mdffile) > 1:
            raise Errors.InputError("Please provide a single extension fits " \
                                    "file for the MDF")
    # Check if the input AstroData object already has an MDF
    for ad in adinput:
        if ad["MDF"]:
            raise Errors.InputError("Input AstroData object already has an " \
                                    "MDF attached")
    mdfdict = {}
    if len(mdf) > 1:
        # Check whether the number of MDFs match the number of input AstroData
        # objects
        if len(adinput) != len(mdf):
            raise Errors.InputError("Please supply either a single MDF to " \
                                    "be applied to all AstroData objects OR " \
                                    "the same number of MDFs as there are " \
                                    "input AstroData objects")
        else:
            # Create a dictionary where the key is the input AstroData object
            # and the value is the MDF file to be added to the input AstroData
            # object
            while i in range (0,len(adinput)):
                mdfdict[adinput[i]] = mdf[i]
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
            mdffile = mdfdict[ad]
            # Renaming the extension"s extname="MDF" and extver=1, even if 
            # they all ready these values just to be sure.
            mdffile.rename_ext("MDF",1)
            mdffile.set_key_value("EXTNAME","MDF", "Extension name")
            mdffile.set_key_value("EXTVER",1,"Extension version")
            # Append the MDF to the input AstroData object
            ad.append(moredata=mdffile)
            log.status("Adding the MDF %s to the input AstroData object %s" \
                       % (mdffile.filename, ad.filename))
            # Add the appropriate time stamps to the PHU
            gt.markHistory(adinput=ad, keyword=keyword)
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def add_mdf_gmos(adinput=None, mdf=None):
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
               MDFs list  must match the length of the inputs.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Check whether an MDF was defined
    if mdf is None:
        # If no MDF is supplied, try to find an appropriate one. First, check
        # the "MASKNAME" keyword
        maskname = ad.phu_get_key_value("MASKNAME")
        if maskname is not None:
            if "IFU" in ad.types:
                # The input AstroData object has an AstroData Type of "IFU".
                # Use the value of the MASKNAME keyword to determine the
                # appropriate MDF
                if "GMOS-S" in ad.types:
                    mdf_prefix = "gsifu"
                if "GMOS-N" in ad.types:
                    mdf_prefix = "gnifu"
                mdf_name = "%s%s" % (mdf_prefix, mdf_dict[maskname])
            else:
                # The MASKNAME keyword defines the actual name of an MDF
                if not maskname.endswith(".fits"):
                    mdf_name = "%s.fits" % maskname
                else:
                    mdf_name = str(maskname)
            # Check if the MDF exists in the current working directory
            if os.path.exists(mdf_name):
                mdf = AstroData(mdf_name)
            # Check if the MDF exists in the gemini_python package
            elif os.path.exists(lookup_path("Gemini/GMOS/MDF/%s" % mdf_name)):
                mdf = AstroData(lookup_path("Gemini/GMOS/MDF/%s" % mdf_name))
            else:
                raise Errors.InputError("The MDF file %s was not found " \
                                        "either in the current working " \
                                        "directory or in the gemini_python " \
                                        "package" % (mdf_name))
    # If mdf is a single AstroData object, put it in a list
    if not isinstance(mdf, list):
        mdf = [mdf]
    # Check if the MDF is a single extension fits file
    for mdffile in mdf:
        if len(mdffile) > 1:
            raise Errors.InputError("Please provide a single extension fits " \
                                    "file for the MDF")
    # Check if the input AstroData object already has an MDF
    for ad in adinput:
        if ad["MDF"]:
            raise Errors.InputError("Input AstroData object already has an " \
                                    "MDF attached")
    mdfdict = {}
    if len(mdf) > 1:
        # Check whether the number of MDFs match the number of input AstroData
        # objects
        if len(adinput) != len(mdf):
            raise Errors.InputError("Please supply either a single MDF to " \
                                    "be applied to all AstroData objects OR " \
                                    "the same number of MDFs as there are " \
                                    "input AstroData objects")
        else:
            # Create a dictionary where the key is the input AstroData object
            # and the value is the MDF file to be added to the input AstroData
            # object
            while i in range (0,len(adinput)):
                mdfdict[adinput[i]] = mdf[i]
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
            mdffile = mdfdict[ad]
            # Renaming the extension"s extname="MDF" and extver=1, even if 
            # they all ready these values just to be sure.
            mdffile.rename_ext("MDF",1)
            mdffile.set_key_value("EXTNAME","MDF", "Extension name")
            mdffile.set_key_value("EXTVER",1,"Extension version")
            # Append the MDF to the input AstroData object
            ad.append(moredata=mdffile)
            log.status("Adding the MDF %s to the input AstroData object %s" \
                       % (mdffile.filename, ad.filename))
            # Add the appropriate time stamps to the PHU
            gt.markHistory(adinput=ad, keyword=keyword)
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

mdf_dict = {
    "IFU-2": "_slits_mdf.fits",
    "IFU-B": "_slitb_mdf.fits",
    "IFU-R": "_slitr_mdf.fits",
    "IFU-NS-2": "_ns_slits_mdf.fits",
    "IFU-NS-B": "_ns_slitb_mdf.fits",
    "IFU-NS-R": "_ns_slitr_mdf.fits",
    }

def add_var(adInputs, outNames=None, suffix=None):
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
            # Check whether the add_var user level function has been run
            # previously
            if ad.phu_get_key_value(keyword) or ad["VAR"]:
                raise Errors.InputError("%s has already been processed by " \
                                        "add_var" % (ad.filename))
            for sci in ad["SCI"]:
                # Using the toolbox function calculateInitialVarianceArray
                # to conduct the actual calculation following:
                # var = (read noise/gain)**2 + max(data,0.0)/gain
                varArray = varutil.calculateInitialVarianceArray(sci)
                 
                # Creating the variance frame"s header and updating it     
                varHeader = varutil.createInitialVarianceHeader(
                                                    extver=sci.extver(),
                                                    shape=varArray.shape)
                # Turning individual variance header and data 
                # into one astrodata instance
                varAD = AstroData(header=varHeader, data=varArray)
                # Appending variance astrodata instance onto input one
                log.debug("Appending new VAR HDU onto the file "
                             +adOut.filename)
                adOut.append(varAD)
                log.status("appending VAR frame "+str(sciExt.extver())+
                           " complete for "+adOut.filename)
            # Add the appropriate time stamps to the PHU
            gt.markHistory(adinput=ad, keyword=keyword)
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
