# This module contains user level functions related to adding extensions to
# and removing extensions from the input dataset

def add_bpm(adinput=None, output_names=None, suffix=None, bpm=None,
            matchSize=False):
    """
    This function will add the provided BPM (Bad Pixel Mask) to the inputs.  
    The BPM will be added as frames matching that of the SCI frames and ensure
    the BPM's data array is the same size as that of the SCI data array.  If the 
    SCI array is larger (say SCI's were overscan trimmed, but BPMs were not), 
    theBPMs will have their arrays padded with zero's to match the sizes and use  
    the data_section descriptor on the SCI data arrays to ensure the match is
    a correct fit.  There must be a matching number of DQ extensions in the BPM 
    as the input the BPM frames are to be added to 
    (ie. if input has 3 SCI extensions, the BPM must have 3 DQ extensions).
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adInputs: Astrodata inputs to be converted to Electron pixel units
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param BPMs: The BPM(s) to be added to the input(s).
    :type BPMs: 
       AstroData objects in a list, or a single instance.
       Note: If there are multiple inputs and one BPM provided, then the
       same BPM will be applied to all inputs; else the BPMs list  
       must match the length of the inputs.
           
    :param matchSize: A flag to use zeros and the key 'DETSEC' of the 'SCI'
                      extensions to match the size of the BPM arrays to those 
                      of for the 'SCI' data arrays.
    :type matchSize: Python boolean (True/False). Default: False.
                      
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same 
                    length as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """

    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adinput, output_names, suffix,
                                     funcName='add_bpm')
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
                   
    if bpm is None:
        log.critical('There must be at least one BPM provided, the '+
                                        '"bpm" parameter must not be None.')
        raise ScienceError()
                   
    try:
        # Set up counter for looping through outNames/BPMs lists
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Do the work on each ad in the inputs
        for ad in adInputs:
            # Getting the right BPM for this input
            if isinstance(bpm, list):
                if len(bpm)>1:
                    BPM = bpm[count]
                else:
                    BPM = bpm[0]
            else:
                BPM = bpm
            
            # Check if this input all ready has a BPM extension
            if not ad['BPM']:
                # Making a deepcopy of the input to work on
                # (ie. a truly new+different object that is a complete copy of the input)
                adOut = deepcopy(ad)
                
                # Getting the filename for the BPM and removing any paths
                BPMfilename = os.path.basename(BPM.filename)
                
                for sciExt in adOut['SCI']:
                    # Extracting the matching DQ extension from the BPM 
                    BPMArrayIn = BPM[('DQ',sciExt.extver())].data
                    
                    # logging the BPM file being used for this SCI extension
                    log.fullinfo('SCI extension number '+
                                 str(sciExt.extver())+', of file '+
                                 adOut.filename+ 
                                 ' is matched to DQ extension '
                                 +str(sciExt.extver())+' of BPM file '+
                                 BPMfilename)
                    
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
                        # output BPM array will be 'padded with zeros' or 
                        # 'not bad pixels' to match SCI's size.
                        if BPMArrayIn.shape==datasecShape:
                            log.fullinfo('BPM data was found to be of a'+
                                     ' different size than the SCI, so padding'+
                                     ' the BPM"s data to match the SCI.')                   
                            BPMArrayOut[dsl[0]:dsl[1], dsl[2]:dsl[3]] = \
                                                                BPMArrayIn
                        elif BPMArrayIn.shape==BPMArrayOut.shape:
                            BPMArrayOut = BPMArrayIn
                    
                    # Don't match size
                    else:
                        BPMArrayOut = BPMArrayIn
                    
                    # Creating a header for the BPM array and updating
                    # further updating to this header will take place in 
                    # addDQ primitive
                    BPMheader = pf.Header() 
                    BPMheader.update('BITPIX', 16, 
                                    'number of bits per data pixel')
                    BPMheader.update('NAXIS', 2)
                    BPMheader.update('PCOUNT', 0, 
                                    'required keyword; must = 0')
                    BPMheader.update('GCOUNT', 1, 
                                    'required keyword; must = 1')
                    BPMheader.update('BUNIT', 'bit', 'Physical units')
                    BPMheader.update('BPMFILE', BPMfilename, 
                                        'Bad Pixel Mask file name')
                    BPMheader.update('EXTVER', sciExt.extver(), 
                                        'Extension Version')
                    # This extension will be renamed DQ in addDQ
                    BPMheader.update('EXTNAME', 'BPM', 'Extension Name')
                    
                    # Creating an astrodata instance from the 
                    # DQ array and header
                    bpmAD = AstroData(header=BPMheader, data=BPMArrayOut)
                    
                    # Using rename_ext to correctly set the EXTVER and 
                    # EXTNAME values in the header   
                    bpmAD.rename_ext('BPM', ver=sciExt.extver())
                    
                    # Appending BPM astrodata instance to the input one
                    log.debug('Appending new BPM HDU onto the file '+ 
                              adOut.filename)
                    adOut.append(bpmAD)
                    log.status('Appending BPM complete for '+ adOut.filename)
        
            # If BPM frames exist, send a warning message to the logger
            else:
                log.warning('BPM frames all ready exist for '+
                             adOut.filename+', so add_bpm will add new ones')
                
            # Updating GEM-TLM (automatic) and ADDBPM time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='ADDBPM')

            # renaming the output ad filename
            adOut.filename = outNames[count]
                    
            log.status('File name updated to '+adOut.filename+'\n')
            
            # Appending to output list
            adOutputs.append(adOut)

            count=count+1
        
        log.status('**FINISHED** the add_bpm function')
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
    
def add_dq(adInputs, fl_nonlinear=True, fl_saturated=True, outNames=None, 
                suffix=None):
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
    
    :param fl_nonlinear: Flag to turn checking for nonlinear pixels on/off
    :type fl_nonLinear: Python boolean (True/False), default is True
    
    :param fl_saturated: Flag to turn checking for saturated pixels on/off
    :type fl_saturated: Python boolean (True/False), default is True
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length 
                    as adInputs.
    
    :param suffix: 
       string to add on the end of the input filenames 
       (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    
    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix, 
                                      funcName='add_dq')
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Loop through the inputs to perform the non-linear and saturated
        # pixel searches of the SCI frames to update the BPM frames into
        # full DQ frames. 
        for ad in adInputs:                
            # Check if DQ extensions all ready exist for this file
            if not ad['DQ']:
                # Making a deepcopy of the input to work on
                # (ie. a truly new+different object that is a complete copy of the input)
                adOut = deepcopy(ad)
                
                for sciExt in adOut['SCI']: 
                    # Retrieving BPM extension 
                    bpmAD = adOut[('BPM',sciExt.extver())]
                    
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
                    # output BPM array will be 'padded with zeros' or 
                    # 'not bad pixels' to match SCI's size.
                    if BPMArrayIn.shape==datasecShape:
                        log.fullinfo('BPM data was found to be of a'+
                                 ' different size than the SCI, so padding'+
                                 ' the BPM"s data to match the SCI.')                   
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
                        log.debug('Performing an np.where to find '+
                                  'non-linear pixels for extension '+
                                  str(sciExt.extver())+' of '+
                                  adOut.filename)
                        nonLinArray = np.where(sciExt.data>linear,2,0)
                        log.status('Done calculating array of non-linear'+
                                   ' pixels')
                    if (saturated is not None) and (fl_saturated):
                        log.debug('Performing an np.where to find '+
                                  'saturated pixels for extension '+
                                  str(sciExt.extver())+' of '+
                                  adOut.filename)
                        saturatedArray = np.where(sciExt.data>saturated,4,0)
                        log.status('Done calculating array of saturated'+
                                   ' pixels') 
                    
                    # Creating one DQ array from the three
                    dqArray=np.add(BPMArrayOut, nonLinArray, saturatedArray) 
                    # Updating data array for the BPM array to be the 
                    # newly calculated DQ array
                    adOut[('BPM',sciExt.extver())].data = dqArray
                    
                    # Renaming the extension to DQ from BPM
                    dqheader.update('EXTNAME', 'DQ', 'Extension Name')
                    
                    # Using rename_ext to correctly set the EXTVER and 
                    # EXTNAME values in the header   
                    bpmAD.rename_ext('DQ', ver=sciExt.extver(), force=True)

                    # Logging that the name of the BPM extension was changed
                    log.fullinfo('BPM Extension '+str(sciExt.extver())+
                                 ' of '+adOut.filename+' had its EXTVER '+
                                 'changed to '+
                                 adOut[('DQ',
                                        sciExt.extver())].header['EXTNAME'])
                    
            # If DQ frames exist, send a warning message to the logger
            else:
                log.warning('DQ frames all ready exist for '+
                             adOut.filename+
                             ', so addDQ will not calculate new ones')
                
            # Updating GEM-TLM (automatic) and ADDDQ time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='ADDDQ')

            # renaming the output ad filename
            adOut.filename = outNames[count]
                    
            log.status('File name updated to '+adOut.filename+'\n')
            
            # Appending to output list
            adOutputs.append(adOut)

            count=count+1
        
        log.status('**FINISHED** the add_dq function')
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
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

    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix, funcName='add_var')
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Loop through the inputs to perform the non-linear and saturated
        # pixel searches of the SCI frames to update the BPM frames into
        # full DQ frames. 
        for ad in adInputs:   
            # Making a deepcopy of the input to work on
            # (ie. a truly new+different object that is a complete copy of the input)
            adOut = deepcopy(ad)
            
            # To clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')
            # Check if there VAR frames all ready exist
            if not adOut['VAR']:                 
                # If VAR frames don't exist, loop through the SCI extensions 
                # and calculate a corresponding VAR frame for it, then 
                # append it
                for sciExt in adOut['SCI']:
                    # Using the toolbox function calculateInitialVarianceArray
                    # to conduct the actual calculation following:
                    # var = (read noise/gain)**2 + max(data,0.0)/gain
                    varArray = varutil.calculateInitialVarianceArray(sciExt)
                     
                    # Creating the variance frame's header and updating it     
                    varHeader = varutil.createInitialVarianceHeader(
                                                        extver=sciExt.extver(),
                                                        shape=varArray.shape)
                    
                    # Turning individual variance header and data 
                    # into one astrodata instance
                    varAD = AstroData(header=varHeader, data=varArray)
                    
                    # Appending variance astrodata instance onto input one
                    log.debug('Appending new VAR HDU onto the file '
                                 +adOut.filename)
                    adOut.append(varAD)
                    log.status('appending VAR frame '+str(sciExt.extver())+
                               ' complete for '+adOut.filename)
                    
            # If VAR frames all ready existed, 
            # make a warning message in the logger
            else:
                log.warning('VAR frames all ready exist for '+adOut.filename+
                             ', so addVAR will not calculate new ones')    
            
            # Updating GEM-TLM (automatic) and ADDVAR time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='ADDVAR')

            # renaming the output ad filename
            adOut.filename = outNames[count]
                    
            log.status('File name updated to '+adOut.filename+'\n')
            
            # Appending to output list
            adOutputs.append(adOut)

            count=count+1
        
        log.status('**FINISHED** the add_var function')
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 

