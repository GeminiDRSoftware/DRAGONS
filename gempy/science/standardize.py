#Author: Kyle Mede, January 2011
#For now, this module is to hold the code which performs the actual work of the 
#primitives that is considered generic enough to be at the 'gemini' level of
#the hierarchy tree.

import os, sys

from copy import deepcopy
from astrodata.AstroData import AstroData
from astrodata.Errors import ScienceError
from gempy.instruments import geminiTools as gemt


def standardize_headers_gmos(adInputs=None, outNames=None, suffix=None):
    """
    This function is to update and add important keywords to the PHU and SCI
    extension headers, first those that are common to ALL Gemini data and then
    those specific to data from the GMOS instrument.
    
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
        
        # Do the work on each ad in the inputs
        for ad in adInputs:
            # First check if the input has been ran through this before, to 
            # avoid accidentally re-updating keys to wrong values.
            if ad.phuGetKeyValue('STDHDRS'):
                log.warning('Input, '+ad.filename+', has all ready had its \
                        headers standardized, so standardize_headers_gemini \
                        will not add/update any keys.')
            
            else:
                # Making a deepcopy of the input to work on
                # (ie. a truly new&different object that is a complete copy 
                # of the input)
                ad.storeOriginalName()
                adOut = deepcopy(ad)
                # moving the filename over as deepcopy doesn't do that
                # only for internal use, renamed below to final name.
                adOut.filename = ad.filename
                
                ## update headers that are common to ALL Gemini data
                log.debug('Calling gemt.standardize_headers_gemini()')
                gemt.standardize_headers_gemini(adOut)
                log.status('Common Gemini headers updated successfully')
                
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
        
            # Updating GEM-TLM (automatic), STDHDRS and PREPARE time stamps to 
            # the PHU and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='STDHDRS')
            sfm.markHistory(adOutputs=adOut, historyMarkKey='PREPARE')
            sfm.markHistory(adOutputs=adOut, historyMarkKey='GPREPARE')
    
            # renaming the output ad filename
            adOut.filename = outNames[count]
            
            log.status('File name updated to '+adOut.filename)
                
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
        raise
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    