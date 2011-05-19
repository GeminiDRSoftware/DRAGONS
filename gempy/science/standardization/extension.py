# This module contains user level functions related to adding extensions to
# and removing extensions from the input dataset

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
        raise Errors.InputError("Please provide at least one MDF")
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
            # Append the MDF to the input AstroData object
            mdffile = mdfdict[ad]
            # Renaming the extension's extname='MDF' and extver=1, even if 
            # they all ready these values just to be sure.
            mdffile.rename_ext('MDF',1)
            mdffile.set_key_value('EXTNAME','MDF', 'Extension name')
            mdffile.set_key_value('EXTVER',1,'Extension version')
            ad.append(moredata=mdffile)
            log.status('Input MDF file = %s is being appended to %s' \
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
