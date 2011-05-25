# This module contains user level functions related to the preprocessing of
# the input dataset with a dark frame

import sys
from gempy import geminiTools as gt
from gempy import managers as man

def subtract_dark(adInputs, darks=None, outNames=None, suffix=None):
    """
    This function will subtract the SCI of the input darks from each SCI frame 
    of the inputs and take care of the VAR and DQ frames if they exist.  
    
    This is all conducted in pure Python through the arith "toolbox" of 
    astrodata. 
       
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adInputs: Astrodata input science data
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param darks: The dark(s) to be added to the input(s).
    :type darks: AstroData objects in a list, or a single instance.
                Note: If there are multiple inputs and one dark provided, 
                then the same dark will be applied to all inputs; else the 
                darks list must match the length of the inputs.
    
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
                                                    funcName='subtract_dark') 
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    # casting darks into a list if not one all ready for later indexing
    if not isinstance(darks, list):
        darks = [darks]
    
    # checking the inputs have matching filters, binning and SCI shapes.
    gt.checkInputsMatch(adInsA=darks, adInsB=adInputs)
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Loop through the inputs 
        for ad in adInputs:  
            # Getting the right dark for this input
            if len(darks)>1:
                dark = darks[count]
            else:
                dark = darks[0]
           
            # sub each dark SCI  from each input SCI and handle the updates to 
            # the DQ and VAR frames.
            # the sub function of the arith toolbox performs a deepcopy so
            # it doesn't need to be done here. 
            adOut = ad.sub(dark)
            
            # renaming the output ad filename
            adOut.filename = outNames[count]
                    
            log.status('File name updated to '+adOut.filename+'\n')
            
            # Updating GEM-TLM (automatic) and SUBDARK time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.mark_history(adOutputs=adOut, historyMarkKey='SUBDARK')
        
            # Appending to output list
            adOutputs.append(adOut)
    
            count=count+1
                
        log.status('**FINISHED** the subtract_dark function')
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
