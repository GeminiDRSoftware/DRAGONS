# This module contains user level functions related to the preprocessing of
# the input dataset with a flat frame

import sys
import numpy as np
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict

def divide_by_flat(adInputs, flats=None, outNames=None, suffix=None):
    """
    This function will divide each SCI extension of the inputs by those
    of the corresponding flat.  If the inputs contain VAR or DQ frames,
    those will also be updated accordingly due to the division on the data.
    
    This is all conducted in pure Python through the arith "toolbox" of 
    astrodata. 
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adInputs: Astrodata inputs to have DQ extensions added to
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param flats: The flat(s) to divide the input(s) by.
    :type flats: AstroData objects in a list, or a single instance.
                Note: If there is multiple inputs and one flat provided, then the
                same flat will be applied to all inputs; else the flats   
                list must match the length of the inputs.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length 
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    # Instantiate ScienceFunctionManager object
    sfm = mgr.ScienceFunctionManager(adInputs, outNames, suffix,
                                      funcName='divide_by_flat') 
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    if flats==None:
        raise Errors.ScienceError('There must be at least one processed flat provided'+
                            ', the "flats" parameter must not be None.')
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Loop through the inputs to perform the non-linear and saturated
        # pixel searches of the SCI frames to update the BPM frames into
        # full DQ frames. 
        for ad in adInputs:                   
            # To clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')    

            # Getting the right flat for this input
            if isinstance(flats, list):
                if len(flats)>1:
                    processedFlat = flats[count]
                else:
                    processedFlat = flats[0]
            else:
                processedFlat = flats

            log.status('Input flat file being used for flat correction '
                       +processedFlat.filename)
            log.debug('Calling ad.div on '+ad.filename)
            
            # the div function of the arith toolbox performs a deepcopy so
            # it doesn't need to be done here.
            adOut = ad.div(processedFlat)
            
            log.status('ad.div successfully flat corrected '+ad.filename)   
            
            # Updating GEM-TLM (automatic) and BIASCORR time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.mark_history(adOutputs=adOut, historyMarkKey='FLATCORR')
            
            # renaming the output ad filename
            adOut.filename = outNames[count]
                    
            log.status('File name updated to '+adOut.filename+'\n')
        
            # Appending to output list
            adOutputs.append(adOut)

            count=count+1
        
        log.status('**FINISHED** the flat_correct function')
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise   
    

def normalize_flat_image(adinput):
    """
    The normalize_flat_image user level function will normalize each science
    extension of the input AstroData object(s) and automatically update the
    variance and data quality extensions, if they exist.
       
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adinput: Astrodata input flat(s) to be combined and normalized
    :type adinput: Astrodata
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "NORMFLAT"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the normalize_flat_image user level function has
            # been run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "normalize_flat_image" % (ad.filename))
            # Loop over each science extension in each input AstroData object
            for ext in ad["SCI"]:
                # Calculate the mean value of the science extension
                mean = np.mean(ext.data)
                # Divide the science extension by the mean value of the science
                # extension
                log.info("Normalizing %s[%s,%d] by dividing by the mean = %f" \
                         % (ad.filename, ext.extname(), ext.extver(), mean))
                ext = ext.div(mean)
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

def normalize_flat_image_gmos(adInputs, fl_trim=False, fl_over=False,  
                                fl_vardq='AUTO', outNames=None, suffix=None):
    """
    This function will combine the input flats (adInputs) and then normalize  
    them using the CL script giflat.
    
    WARNING: The giflat script used here replaces the previously 
    calculated DQ frames with its own versions.  This may be corrected 
    in the future by replacing the use of the giflat
    with a Python routine to do the flat normalizing.
    
    NOTE: The inputs to this function MUST be prepared. 

    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adInputs: Astrodata input flat(s) to be combined and normalized
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param fl_trim: Trim the overscan region from the frames?
    :type fl_trim: Python boolean (True/False)
    
    :param fl_over: Subtract the overscan level from the frames?
    :type fl_over: Python boolean (True/False)
    
    :param fl_vardq: Create variance and data quality frames?
    :type fl_vardq: Python boolean (True/False), OR string 'AUTO' to do 
                    it automatically if there are VAR and DQ frames in the 
                    inputs.
                    NOTE: 'AUTO' uses the first input to determine if VAR and  
                    DQ frames exist, so, if the first does, then the rest MUST 
                    also have them as well.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length 
                    as adInputs.
    
    :param suffix:
            string to add on the end of the input filenames 
            (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    # Instantiate ScienceFunctionManager object
    sfm = mgr.ScienceFunctionManager(adInputs, outNames, suffix,
                                       funcName='normalize_flat_image_gmos', 
                                       combinedInputs=True)
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    try:
        # loading and bringing the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()  
            
        # Converting input True/False to yes/no or detecting fl_vardq value
        # if 'AUTO' chosen with autoVardq in the ScienceFunctionManager
        fl_vardq = sfm.autoVardq(fl_vardq)
        
        # To clean up log and screen if multiple inputs
        log.fullinfo('+'*50, category='format')    
        
        # Preparing input files, lists, parameters... for input to 
        # the CL script
        clm=mgr.CLManager(imageIns=adInputs, imageOutsNames=outNames,  
                           suffix=suffix, funcName='normalizeFlat', 
                           log=log, combinedImages=True)
        
        # Check the status of the CLManager object, True=continue, False= issue warning
        if clm.status:                 
            # Creating a dictionary of the parameters set by the mgr.CLManager 
            # or the definition of the function 
            clPrimParams = {
              'inflats'     :clm.imageInsFiles(type='listFile'),
              # Maybe allow the user to override this in the future
              'outflat'     :clm.imageOutsFiles(type='string'), 
              # This returns a unique/temp log file for IRAF  
              'logfile'     :clm.templog.name,                   
                          }
            # Creating a dictionary of the parameters from the function call 
            # adjustable by the user
            clSoftcodedParams = {
               'fl_vardq'   :fl_vardq,
               'fl_over'    :gt.pyrafBoolean(fl_over),
               'fl_trim'    :gt.pyrafBoolean(fl_trim)
                               }
            # Grabbing the default params dict and updating it 
            # with the two above dicts
            clParamsDict = CLDefaultParamsDict('giflat')
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Logging the parameters that were not defaults
            log.fullinfo('\nParameters set automatically:', 
                         category='parameters')
            # Loop through the parameters in the clPrimParams dictionary
            # and log them
            gt.logDictParams(clPrimParams)
            
            log.fullinfo('\nParameters adjustable by the user:', 
                         category='parameters')
            # Loop through the parameters in the clSoftcodedParams 
            # dictionary and log them
            gt.logDictParams(clSoftcodedParams)
            
            log.debug('Calling the giflat CL script for inputs list '+
                  clm.imageInsFiles(type='listFile'))
        
            gemini.giflat(**clParamsDict)
            
            if gemini.giflat.status:
                raise Errors.ScienceError('giflat failed for inputs '+
                             clm.imageInsFiles(type='string'))
            else:
                log.status('Exited the giflat CL script successfully')
            
            # Renaming CL outputs and loading them back into memory 
            # and cleaning up the intermediate temp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL()
        
            # Renaming for symmetry
            adOutputs=imageOuts
        
            # Updating GEM-TLM (automatic) and COMBINE time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.mark_history(adOutputs=adOutputs, historyMarkKey='GIFLAT')    
        else:
            raise Errors.ScienceError('One of the inputs has not been prepared,'+
            'the normalizeFlat function can only work on prepared data.')
                
        log.status('**FINISHED** the normalize_flat_image_gmos function')
        
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
