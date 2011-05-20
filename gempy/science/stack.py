# This module contains user level functions related to the stacking of the
# input dataset

import sys
from astrodata import Errors
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as man
from gempy.geminiCLParDicts import CLDefaultParamsDict

def stack_frames(adInputs, fl_vardq=True, fl_dqprop=True, method='average', 
            outNames=None, suffix=None):
    """
    This function will average and combine the SCI extensions of the 
    inputs. It takes all the inputs and creates a list of them and 
    then combines each of their SCI extensions together to create 
    average combination file. New VAR frames are made from these 
    combined SCI frames and the DQ frames are propagated through 
    to the final file.
    
    NOTE: The inputs to this function MUST be prepared. 

    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adInputs: Astrodata inputs to be combined
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param fl_vardq: Create variance and data quality frames?
    :type fl_vardq: Python boolean (True/False), OR string 'AUTO' to do 
                    it automatically if there are VAR and DQ frames in the 
                    inputs. NOTE: 'AUTO' uses the first input to determine if 
                    VAR and DQ frames exist, so, if the first does, then the 
                    rest MUST also have them as well.
    
    :param fl_dqprop: propogate the current DQ values?
    :type fl_dqprop: Python boolean (True/False)
    
    :param method: type of combining method to use.
    :type method: string, options: 'average', 'median'.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    
    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix, 
                                      funcName='combine', combinedInputs=True)
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Ensuring there is more than one input to combine
        if (len(adInputs)>1):
            
            # loading and bringing the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()
            
            # Converting input True/False to yes/no or detecting fl_vardq value
            # if 'AUTO' chosen with autoVardq in the ScienceFunctionManager
            fl_vardq = sfm.autoVardq(fl_vardq)
                
            # Preparing input files, lists, parameters... for input to 
            # the CL script
            clm=man.CLManager(imageIns=adInputs, imageOutsNames=outNames, 
                               suffix=suffix, funcName='combine', 
                               combinedImages=True, log=log)
            
            # Check the status of the CLManager object, True=continue, False= issue warning
            if clm.status:
            
                # Creating a dictionary of the parameters set by the CLManager  
                # or the definition of the primitive 
                clPrimParams = {
                    # Retrieving the inputs as a list from the CLManager
                    'input'       :clm.imageInsFiles(type='listFile'),
                    # Maybe allow the user to override this in the future. 
                    'output'      :clm.imageOutsFiles(type='string'), 
                    # This returns a unique/temp log file for IRAF  
                    'logfile'     :clm.templog.name,   
                    'reject'      :'none'                
                              }
                
                # Creating a dictionary of the parameters from the Parameter 
                # file adjustable by the user
                clSoftcodedParams = {
                    'fl_vardq'      :fl_vardq,
                    # pyrafBoolean converts the python booleans to pyraf ones
                    'fl_dqprop'     :gt.pyrafBoolean(fl_dqprop),
                    'combine'       :method,
                                    }
                # Grabbing the default parameters dictionary and updating 
                # it with the two above dictionaries
                clParamsDict = CLDefaultParamsDict('gemcombine')
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
                
                log.debug('Calling the gemcombine CL script for input '+
                                 'list '+clm.imageInsFiles(type='listFile'))
                
                gemini.gemcombine(**clParamsDict)
                
                if gemini.gemcombine.status:
                    raise Errors.ScienceError('gemcombine failed for inputs '+
                                 clm.imageInsFiles(type='string'))
                else:
                    log.status('Exited the gemcombine CL script '+
                                                            'successfully')
                
                # Renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate temp files written to disk
                # refOuts and arrayOuts are None here
                imageOuts, refOuts, arrayOuts = clm.finishCL() 
            
                # Renaming for symmetry
                adOutputs=imageOuts
                
                # Updating GEM-TLM (automatic) and COMBINE time stamps to the PHU
                # and updating logger with updated/added time stamps
                sfm.markHistory(adOutputs=adOutputs, historyMarkKey='COMBINE')
            else:
                raise Errors.ScienceError('One of the inputs has not been prepared,'+
                'the combine function can only work on prepared data.')
        else:
            log.warning('Only one input was passed in for adInputs, so combine'+
                    'is simply passing the inputs into the outputs list '+
                    'without doing anything to them.')
            adOutputs = adInputs
        
        log.status('**FINISHED** the combine function')
        
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise
