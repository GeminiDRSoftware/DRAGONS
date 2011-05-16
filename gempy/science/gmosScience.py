# Author: Kyle Mede, February 2011
# For now, this module is to hold the code which performs the actual work of the 
# primitives that is considered generic enough to be at the 'GMOS' level of
# the hierarchy tree, but too specific for the 'gemini' level above this.

import os, sys

import pyfits as pf
import numpy as np
from copy import deepcopy

from astrodata.AstroData import AstroData
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.Errors import ScienceError
from gempy import geminiTools as gemt
from gempy import managers as man
from gempy.geminiCLParDicts import CLDefaultParamsDict
    
def make_fringe_frame_imaging(adInputs, fl_vardq='AUTO', method='median', 
            outNames=None, suffix=None):
    """
    This function will create and return a single fringe image from all the 
    inputs.  It utilizes the CL script gifringe to create the fringe image.
    
    NOTE: The inputs to this function MUST be prepared. 

    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adInputs: Astrodata inputs to be combined
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param fl_vardq: Create variance and data quality frames?
    :type fl_vardq: Python boolean (True/False), OR string 'AUTO' to do 
                    it automatically if there are VAR and DQ frames in the 
                    inputs.
                    NOTE: 'AUTO' uses the first input to determine if VAR and DQ  
                    frames exist, so, if the first does, then the rest MUST also 
                    have them as well.
    
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
                                       funcName='make_fringe_frame_imaging', 
                                       combinedInputs=True)
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Ensuring there is more than one input to make a fringe frame from
        if (len(adInputs)>1):
            
            # loading and bringing the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader() 
             
            # Creating empty list of ad's to be returned that will be filled below
            adOutputs=[]
                    
            # Converting input True/False to yes/no or detecting fl_vardq value
            # if 'AUTO' chosen with autoVardq in the ScienceFunctionManager
            fl_vardq = sfm.autoVardq(fl_vardq)
            
            # To clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')                                 
                
            # Preparing input files, lists, parameters... for input to 
            # the CL script
            clm = man.CLManager(imageIns=adInputs, imageOutsNames=outNames,  
                               suffix=suffix, funcName='makeFringeFrame', 
                               combinedImages=True, log=log)
            
            # Check the status of the CLManager object, True=continue, False= issue warning
            if clm.status:                     
            
                # Parameters set by the man.CLManager or the definition 
                # of the primitive 
                clPrimParams = {
                    # Retrieving the inputs as a list from the CLManager
                    'inimages'    :clm.imageInsFiles(type='listFile'),
                    # Maybe allow the user to override this in the future. 
                    'outimage'    :clm.imageOutsFiles(type='string'), 
                    # This returns a unique/temp log file for IRAF  
                    'logfile'     :clm.templog.name,                   
                              }
    
                # Creating a dictionary of the parameters from the Parameter 
                # file adjustable by the user
                clSoftcodedParams = {
                    'fl_vardq'      :gemt.pyrafBoolean(fl_vardq),
                    'combine'       :method,
                    'reject'        :'none',
                                    }
                # Grabbing the default parameters dictionary and updating 
                # it with the two above dictionaries
                clParamsDict = CLDefaultParamsDict('gifringe')
                clParamsDict.update(clPrimParams)
                clParamsDict.update(clSoftcodedParams)
                
                # Logging the values in the soft and prim parameter dictionaries
                log.fullinfo('\nParameters set by the CLManager or  '+
                         'dictated by the definition of the primitive:\n', 
                         category='parameters')
                gemt.logDictParams(clPrimParams)
                log.fullinfo('\nUser adjustable parameters in the '+
                             'parameters file:\n', category='parameters')
                gemt.logDictParams(clSoftcodedParams)
                
                log.debug('Calling the gifringe CL script for input list '+
                              clm.imageInsFiles(type='listFile'))
                
                gemini.gifringe(**clParamsDict)
                
                if gemini.gifringe.status:
                    raise ScienceError('gifringe failed for inputs '+
                                 rc.inputs_as_str())
                else:
                    log.status('Exited the gifringe CL script successfully')
                    
                # Renaming CL outputs and loading them back into memory 
                # and cleaning up the intermediate temp files written to disk
                # refOuts and arrayOuts are None here
                imageOuts, refOuts, arrayOuts = clm.finishCL() 
                
                # Renaming for symmetry
                adOutputs=imageOuts
            
                # Renaming for symmetry
                adOutputs=imageOuts
                
                # Updating GEM-TLM (automatic) and COMBINE time stamps to the PHU
                # and updating logger with updated/added time stamps
                sfm.markHistory(adOutputs=adOutputs, historyMarkKey='FRINGE')
            else:
                raise ScienceError('One of the inputs has not been prepared,'+
                'the combine function can only work on prepared data.')
        else:
            log.warning('Only one input was passed in for adInputs, so '+
                    'make_fringe_frame_imaging is simply passing the inputs  '+
                    'into the outputs list without doing anything to them.')
            adOutputs = adInputs
        
        log.status('**FINISHED** the make_fringe_frame_imaging function')
        
        # Return the outputs list, even if there is only one output
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
    
    
    
    