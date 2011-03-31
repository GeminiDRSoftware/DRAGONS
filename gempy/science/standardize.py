#Author: Kyle Mede, January 2011
#For now, this module is to hold the code which performs the actual work of the 
#primitives that is considered generic enough to be at the 'gemini' level of
#the hierarchy tree.

import os, sys

from copy import deepcopy
from astrodata.AstroData import AstroData
from astrodata.ConfigSpace import lookupPath
from astrodata.Errors import ScienceError
from gempy import geminiTools as gemt

def add_mdf(adInputs=None, MDFs=None, outNames=None, suffix=None):
    """
    This function is to attach the MDFs to the inputs as an extension. 
    It is assumed that the MDFs are single extensions fits files and will
    thus be appended as ('MDF',1) onto the inputs.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adInputs: Astrodata inputs to have their headers standardized
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param MDFs: The MDF(s) to be added to the input(s).
    :type MDFs: AstroData objects in a list, or a single instance.
                Note: If there are multiple inputs and one MDF provided, 
                then the same MDF will be applied to all inputs; else the 
                MDFs list  must match the length of the inputs.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same 
                    length as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    # Instantiate ScienceFunctionManager object
    sfm = gemt.ScienceFunctionManager(adInputs, outNames, suffix, 
                                                            funcName='add_mdf')
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
                   
    if MDFs==None:
        log.critical('There must be at least one MDF provided, the '+
                                        '"MDFs" parameter must not be None.')
        raise ScienceError()
                   
    try:
        # Set up counter for looping through outNames/MDFs lists
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Do the work on each ad in the inputs
        for ad in adInputs:
            # Getting the right MDF for this input
            if isinstance(MDFs, list):
                if len(MDFs)>1:
                    MDF = MDFs[count]
                else:
                    MDF = MDFs[0]
            else:
                MDF = MDFs
            
            # Check if this input all ready has a MDF extension
            if not ad['MDF']:
                # Making a deepcopy of the input to work on
                # (ie. a truly new+different object that is a complete copy of the input)
                adOut = deepcopy(ad)
                # moving the filename over as deepcopy doesn't do that
                # only for internal use, renamed below to final name.
                adOut.filename = ad.filename
                
                # checking that the MDF found is not a MEF, else raise
                if len(MDF)>1:
                    log.critical('The MDF file, '+MDFfilename+', was found '+
                    'to have '+str(len(MDF))+' extensions.  All MDFs MUST '+
                    'be SINGLE extensions fits files.')
                    raise ScienceError('MDF'+MDFfilename+' has more than '+
                                                            '1 extension.')
                
                # Getting the filename for the MDF and removing any paths
                MDFfilename = os.path.basename(MDF.filename)
                log.status('Input MDF file = '+MDFfilename+' is being appended '+
                           ' onto '+adOut.filename)
                # Renaming the extension's extname='MDF' and extver=1, even if 
                # they all ready these values just to be sure.
                MDF.renameExt('MDF',1)
                MDF.setKeyValue('EXTNAME','MDF', 'Extension name')
                MDF.setKeyValue('EXTVER',1,'Extension version')
                
                # log the current infostr() of both the ad and mdf for debugging
                log.debug('Info strings before appending MDF:')
                log.debug(adOut.infostr())
                log.debug(MDF.infostr())
               
                # Append the MDF to the input
                adOut.append(moredata=MDF)
                
                # log the final infostr() of the resultant ad for debugging
                log.debug('Info string after appending MDF:')
                log.debug(adOut.infostr())
                
                log.status('Appending the MDF complete for '+adOut.filename)
                
            # If MDF frames exist, send a warning message to the logger
            else:
                log.warning('An MDF frame all ready exist for '+
                             adOut.filename+', so add_mdf will not add new ones')
            
            # Updating GEM-TLM (automatic) and ADDMDF time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='ADDMDF')

            # renaming the output ad filename
            adOut.filename = outNames[count]
                    
            log.status('File name updated to '+adOut.filename+'\n')
            
            # Appending to output list
            adOutputs.append(adOut)

            count=count+1
        
        log.status('**FINISHED** the add_mdf function')
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise ScienceError('An error occurred while trying to run add_mdf')             
                
def standardize_headers_gemini(adInputs=None, outNames=None, suffix=None):
    """ 
    This function is used by the standardizeHeaders in primitive, through the
    Science Function standardize.standardize_headers_####; where #### 
    corresponds to the instrument's short name (ex. GMOS, F2...)
        
    It will add the PHU header keys NSCIEXT, NEXTEND and ORIGNAME.
    
    In the SCI extensions the header keys BUNIT, NONLINEA and SATLEVEL 
    will be added.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adInputs: Astrodata inputs to have their headers standardized
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same 
                    length as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    # Instantiate ScienceFunctionManager object
    sfm = gemt.ScienceFunctionManager(adInputs, outNames, suffix, 
                                      funcName='standardize_headers_gemini')
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    try:
        # Set up counter for looping through outNames lists during renaming
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Do the work on each ad in the inputs
        for ad in adInputs:
            # First check if the input has been ran through this before, to 
            # avoid accidentally re-updating keys to wrong values.
            if ad.phuGetKeyValue('STDHDRSG'):
                log.warning('Input, '+ad.filename+', has all ready had its \
                        general Gemini headers standardized, so \
                        standardize_headers_gemini will not add/update any keys.')
            
            else:
                # Making a deepcopy of the input to work on
                # (ie. a truly new&different object that is a complete copy 
                # of the input)
                ad.storeOriginalName()
                adOut = deepcopy(ad)
                # moving the filename over as deepcopy doesn't do that
                # only for internal use, renamed below to final name.
                adOut.filename = ad.filename
                
                # Formatting so logger looks organized for these messages
                log.fullinfo('*'*50, category='header') 
                log.fullinfo('file = '+adOut.filename, category='header')
                log.fullinfo('~'*50, category='header')
                log.fullinfo('PHU keywords updated/added:\n', category='header')
                
                # Keywords that are updated/added for all Gemini PHUs 
                gemt.update_key_value(adOut, 'countExts("SCI")')
                gemt.update_key_value(adOut,'storeOriginalName()')
                # updating keywords that are NOT calculated/looked up using 
                # descriptors or built-in ad functions.
                ad.phuSetKeyValue('NEXTEND', len(adOut) , 
                                  '(UPDATED) Number of extensions')
                log.fullinfo('NEXTEND = '+str(adOut.phuGetKeyValue('NEXTEND')), 
                             category='header' )
                
                log.fullinfo('-'*50, category='header')
                     
                # A loop to add the missing/needed keywords in the SCI extensions
                for ext in adOut['SCI']:
                     # Updating logger with new header key values
                    log.fullinfo('SCI extension number '+str(ext.extver())+
                                ' keywords updated/added:\n', category='header')      
                     
                    # Keywords that are updated/added for all Gemini SCI extensions
                    gemt.update_key_value(ext, 'non_linear_level()', phu=False)
                    gemt.update_key_value(ext, 'saturation_level()', phu=False)
                    # updating keywords that are NOT calculated/looked up using descriptors
                    # or built-in ad functions.
                    ext.setKeyValue('BUNIT','adu', '(NEW) Physical units')
                    log.fullinfo('BUNIT = '+str(ext.getKeyValue('BUNIT')), 
                             category='header' )
                    
                    log.fullinfo('-'*50, category='header') 
            # Updating GEM-TLM (automatic), STDHDRSG and PREPARE time stamps to 
            # the PHU and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='STDHDRSG') 
            sfm.markHistory(adOutputs=adOut, historyMarkKey='PREPARE')
            # This one shouldn't be needed, but just adding it just in case 
            sfm.markHistory(adOutputs=adOut, historyMarkKey='GPREPARE')
    
            # renaming the output ad filename
            adOut.filename = outNames[count]
            
            log.status('File name updated to '+adOut.filename+'\n')
                
            # Appending to output list
            adOutputs.append(adOut)
    
            count=count+1
        
        log.status('**FINISHED** the standardize_headers_gemini function')
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise ScienceError('An error occurred while trying to run \
                                                    standardize_headers_gemini')

def standardize_headers_gmos(adInputs=None, outNames=None, suffix=None):
    """
    This function is to update and add important keywords to the PHU and SCI
    extension headers, first those that are common to ALL Gemini data (performed
    by the standardize_headers_gemini science function) and then those specific
    to data from the GMOS instrument.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adInputs: Astrodata inputs to have their headers standardized
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same 
                    length as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string
    """
    # Instantiate ScienceFunctionManager object
    sfm = gemt.ScienceFunctionManager(adInputs, outNames, suffix, 
                                      funcName='standardize_headers_gmos')
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    try:
        # Set up counter for looping through outNames lists during renaming
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        ## update headers that are common to ALL Gemini data
        log.debug('Calling standardize_headers_gemini()')
        #NOTE: passing the outNames for this function directly to the gemini
        #      version, maybe consider having different names for each func !?!?
        ads = standardize_headers_gemini(adInputs, outNames)
        log.status('Common Gemini headers updated successfully')
        
        # Do the work on each ad in the outputs from standardize_headers_gemini
        for ad in ads:
            # First check if the input has been ran through this before, to 
            # avoid accidentally re-updating keys to wrong values.
            if ad.phuGetKeyValue('STDHDRSI'):
                log.warning('Input, '+ad.filename+', has all ready had its \
                        instrument specific headers standardized, so \
                        standardize_headers_gmos will not add/update any keys.')
            
            else:
                # Making a deepcopy of the input to work on
                # (ie. a truly new&different object that is a complete copy 
                # of the input)
                ad.storeOriginalName()
                adOut = deepcopy(ad)
                # moving the filename over as deepcopy doesn't do that
                # only for internal use, renamed below to final name.
                adOut.filename = ad.filename
                
                ## update headers that are GMOS specific
                log.status('Updating GMOS specific headers')
                # Formatting so logger looks organized for these messages
                log.fullinfo('*'*50, category='header') 
                log.fullinfo('file = '+adOut.filename, category='header')
                log.fullinfo('~'*50, category='header')
                
                # Adding the missing/needed keywords into the PHU
                ### NONE updated for PHU that### 
               
               # Adding the missing/needed keywords into the SCI extensions
                for ext in adOut['SCI']:
                    # Formatting so logger looks organized for these messages
                    log.fullinfo('SCI extension number '+
                                 str(ext.header['EXTVER'])+
                                 ' keywords updated/added:\n', 'header')       
                    
                    gemt.update_key_value(ext,'pixel_scale()', phu=False)
                    gemt.update_key_value(ext,'read_noise()', phu=False)               
                    gemt.update_key_value(ext,'gain()', phu=False)
                    if 'GMOS_IMAGE' not in ext.getTypes():
                        gemt.update_key_value(ext,'dispersion_axis()', 
                                              phu=False)
                    
                    log.fullinfo('-'*50, category='header')
        
            # Updating GEM-TLM (automatic), STDHDRSI and PREPARE time stamps to 
            # the PHU and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='STDHDRSI')
            sfm.markHistory(adOutputs=adOut, historyMarkKey='PREPARE')
            # This one shouldn't be needed, but just adding it just in case 
            sfm.markHistory(adOutputs=adOut, historyMarkKey='GPREPARE')
    
            # renaming the output ad filename
            adOut.filename = outNames[count]
            
            log.status('File name updated to '+adOut.filename+'\n')
                
            # Appending to output list
            adOutputs.append(adOut)
    
            count=count+1
        
        log.status('**FINISHED** the standardize_headers_gmos function')
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise ScienceError('An error occurred while trying to run \
                                                     standardize_headers_gmos')
    
def standardize_structure_gmos(adInputs=None, addMDF=False, mdfFiles=None, 
                                                    outNames=None, suffix=None):
    """
    This function ensures the MEF structure of GMOS data is ready for further 
    processing, through adding an MDF if necessary.  Appropriately all SPECT
    type data should have an MDF added, while that of IMAGE should not.  If 
    adInputs contains mixed types of GMOS data (ie. some IMAGE and some SPECT), 
    then only those of type SPECT will have MDFs attached.  The MDF to add can 
    be indicated by providing its filename in the MASKNAME PHU key, or the 
    mdfFiles parameter.
    This function is called by standardizeInstrumentStructure in both the GMOS 
    and GMOS_IMAGE primitives sets to perform their work.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adInputs: Astrodata inputs to have their headers standardized
    :type adInputs: Astrodata objects, either a single or a list of objects
    
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
    
    :param outNames: Filenames of output(s)
    :type outNames: String, either a single or a list of strings of same 
                    length as adInputs.
    
    :param suffix: String to add on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string
    """
    # Instantiate ScienceFunctionManager object
    sfm = gemt.ScienceFunctionManager(adInputs, outNames, suffix, 
                                      funcName='standardize_headers_gmos')
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    try:
        # Set up counter for looping through outNames lists during renaming
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        for ad in adInputs:
            if addMDF:
                # Ensuring data is not of type IMAGE, as they do not require an MDF
                if 'IMAGE' in ad.getTypes():
                    # if of type IMAGE then log critical message and pass input 
                    # to the outputs without looking up and appending an MDF
                    log.critical('Input '+ad.filename+' is an IMAGE and '+
                                    'should not have an MDF attached. Thus, '+
                                    'standardize_structure_gmos will pass '+
                                    'this input to the outputs unchanged.')
                    # Passing the input to be an output without appending any MDF
                    adOut = ad
                else:
                    log.status('Starting to hunt for matching MDF file')
                    # Input is not of type IMAGE so look up and append the right MDF
                    phuMDFkey = ad.phuGetKeyValue('MASKNAME')
                    # check if this key exists in the PHU or if
                    # the input is of type IFU.  If not there, only use the 
                    # provided MDF through the mdfFiles param, if = 'IFU-*' then 
                    # convert that to a filename.
                    if (phuMDFkey is None):
                        # it isn't in the PHU, so only use specified one, 
                        # ensuring to take get the right one from mdfFiles
                        if isinstance(mdfFiles,list):
                            if len(mdfFiles)>1:
                                MDFfilename = mdfFiles[count]
                            elif len(mdfFiles)==1:
                                MDFfilename = mdfFiles[0]
                            else:
                                # mdfFiles is an empty list so msg and raise
                                log.critical('Input '+ad.filename+' has no '+
                                'MASKNAME key in its PHU and no mdfFiles is '+
                                'an empty list.')
                                raise ScienceError('mdfFiles was an empty '+
                                'list so no suitible MDF could be found for '+
                                'input '+ad.filename)
                        elif isinstance(mdfFiles, str):
                            MDFfilename = mdfFiles
                        else:
                            # Provided mdfFiles is not a string or list of them
                            # so make critical msg and raise ScienceError
                            log.critical('The MASKNAME key did not exist in '+
                                'the PHU of '+ad.filename+' and the mdfFiles '+
                                'provided was of type '+repr(type(mdfFiles))+
                                ' and it MUST be a string or a list of them.')
                            raise ScienceError('Input '+ad.filename+' had no '+
                                            'MASKNAME key in the PHU and the '+
                                            'mdfFiles provided was invalid.')
                            
                    if (phuMDFkey is not None) and ('IFU' in ad.getTypes()):
                        # The input is of type IFU, so the value for the 
                        # MASKNAME PHU key needs to be used to find the 
                        # appropriate MDF filename
                        if 'GMOS-S' in ad.getTypes():
                            mdfPrefix = 'gsifu_'
                        if 'GMOS-N' in ad.getTypes():
                            mdfPrefix = 'gnifu_'
                        if phuMDFkey=='IFU-2':
                            MDFfilename = mdfPrefix+'slits_mdf.fits'
                        if phuMDFkey=='IFU-B':
                            MDFfilename = mdfPrefix+'slitb_mdf.fits'
                        if phuMDFkey=='IFU-R':
                            MDFfilename = mdfPrefix+'slitr_mdf.fits'
                        if phuMDFkey=='IFU-NS-2':
                            MDFfilename = mdfPrefix+'ns_slits_mdf.fits'
                        if phuMDFkey=='IFU-NS-B':
                            MDFfilename = mdfPrefix+'ns_slitb_mdf.fits'
                        if phuMDFkey=='IFU-NS-R':
                            MDFfilename = mdfPrefix+'ns_slitr_mdf.fits'    
                            
                    else:
                        # There was a value for MASKNAME in the PHU and the 
                        # input is not of IFU type, so ensure it has a .fits at
                        # the end and then use it
                        if isinstance(phuMDFkey,str):
                            if phuMDFkey[-5:]=='.fits':
                                MDFfilename = phuMDFkey
                            else:
                                MDFfilename = phuMDFkey+'.fits'
                    
                    # First check if file is in the current working directory
                    if os.path.exists(MDFfilename):
                        MDF = AstroData(MDFfilename)
                    # If not there, see if it is in lookups/GMOS/MDF dir
                    elif os.path.exists(lookupPath('Gemini/GMOS/MDF/'+MDFfilename)):
                        MDF = AstroData(lookupPath('Gemini/GMOS/MDF/'+MDFfilename))
                    else:
                        log.critical('MDF file '+MDFfilename+' was not found '+
                                        'on disk.')
                        raise ScienceError('MDF file '+MDFfilename+' was not '+
                                            'found on disk.')
                        
                    # log MDF file being used for current input ad    
                    log.status('MDF, '+MDF.filename+', was found for input, '+
                                                                    ad.filename)
        
                    # passing the found single MDF for the current input to add_mdf
                    # NOTE: This is another science function, so it performs the normal
                    #       deepcopy and filename handling that would normally go here.
                    log.debug('Calling add_mdf to append the MDF')
                    adOuts = add_mdf(adInputs=ad, MDFs=MDF,outNames=outNames[count])
                    # grab the single output in the list as only one went in
                    adOut = adOuts[0]
                    log.status('Input ,'+adOut.filename+', successfully had '+
                                                        'its MDF appended on.')
            else:
                # addMDF=False, so just pass the inputs through without 
                # bothering with looking up or attaching MDFs no matter what 
                # type the inputs are.
                log.status('addMDF was set to False so Input '+ad.filename+
                           ' was just passed through to the outputs.')
                adOut = ad
                
            # Updating GEM-TLM (automatic), STDSTRUC and PREPARE time stamps to 
            # the PHU and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='STDSTRUC')
            sfm.markHistory(adOutputs=adOut, historyMarkKey='PREPARE')
            # This one shouldn't be needed, but just adding it just in case 
            sfm.markHistory(adOutputs=adOut, historyMarkKey='GPREPARE')
    
            # renaming the output ad filename
            adOut.filename = outNames[count]
            
            log.status('File name updated to '+adOut.filename+'\n')
                
            # Appending to output list
            adOutputs.append(adOut)
    
            count=count+1
        
        log.status('**FINISHED** the standardize_structure_gmos function')
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise ScienceError('An error occurred while trying to run \
                                                    standardize_structure_gmos')
    
def validate_data_gmos(adInputs=None, repair=False, outNames=None, suffix=None):
    """
    This function will ensure the data is not corrupted or in an odd 
    format that will affect later steps in the reduction process.  
    It currently just checks if there are 1, 3, 6 or 12 SCI extensions 
    in the input. If there are issues 
    with the data, the flag 'repair' can be used to turn on the feature to 
    repair it or not.
    
    This function is called by standardizeInstrumentStructure in both the GMOS 
    and GMOS_IMAGE primitives sets to perform their work.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param adInputs: Astrodata inputs to have their headers standardized
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param repair: A flag to turn on/off repairing the data if there is a
                   problem with it. 
                   Note: this feature does not work yet.
    :type repair: Python boolean (True/False)
                  default: True
              
    :param mdfFiles: A file name (with path) of the MDF file to append onto the 
                     input(s).
                     Note: If there are multiple inputs and one mdfFiles  
                     provided, then the same MDF will be applied to all inputs;  
                     else the mdfFiles must be in a list of match the length of  
                     the inputs and the inputs must ALL be of type SPECT.
    :type mdfFiles: String, or list of strings.
    
    :param outNames: Filenames of output(s)
    :type outNames: String, either a single or a list of strings of same 
                    length as adInputs.
    
    :param suffix: String to add on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string
    """
    # Instantiate ScienceFunctionManager object
    sfm = gemt.ScienceFunctionManager(adInputs, outNames, suffix, 
                                      funcName='validate_data_gmos')
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    try:
        # Set up counter for looping through outNames lists during renaming
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        for ad in adInputs:
            # Making a deepcopy of the input to work on
            # (ie. a truly new&different object that is a complete copy 
            # of the input)
            ad.storeOriginalName()
            adOut = deepcopy(ad)
            # moving the filename over as deepcopy doesn't do that
            # only for internal use, renamed below to final name.
            adOut.filename = ad.filename
            
            if repair:
                ################################################################
                ######## This is where the code or a call to a function would ##
                ######## go that performs any repairs for GMOS type data      ##
                ################################################################
                log.warning('Currently there are no repair features for '+
                            'GMOS type data.  Maybe there will be in the '+
                            'future if someone writes of some.')
                pass
            
            length=adOut.countExts('SCI')
            # If there are 1, 3, 6, or 12 extensions, all good, if not log a  
            # critical message and raise an exception
            if length==1 or length==3 or length==6 or length==12:
                pass
            else: 
                raise ScienceError('There are NOT 1, 3, 6 or 12 extensions '+
                                    'in file = '+adOut.filename)
                    
            # Updating GEM-TLM (automatic), VALDATA and PREPARE time stamps to 
            # the PHU and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='VALDATA')
            sfm.markHistory(adOutputs=adOut, historyMarkKey='PREPARE')
            # This one shouldn't be needed, but just adding it just in case 
            sfm.markHistory(adOutputs=adOut, historyMarkKey='GPREPARE')
    
            # renaming the output ad filename
            adOut.filename = outNames[count]
            
            log.status('File name updated to '+adOut.filename+'\n')
                
            # Appending to output list
            adOutputs.append(adOut)
    
            count=count+1
        
        log.status('**FINISHED** the validate_data_gmos function')
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise ScienceError('An error occurred while trying to run \
                                                    validate_data_gmos')
