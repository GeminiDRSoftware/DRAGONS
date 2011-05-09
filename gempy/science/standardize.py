#Author: Kyle Mede, March 2011
#For now, this module is to hold the code which performs standardizing steps
#such as those of the primitives in the prepare recipe.
# The standardize module contains the user level functions that update the raw
# data to a specific standard

import os, sys
from copy import deepcopy
from astrodata.AstroData import AstroData
from astrodata.ConfigSpace import lookupPath
from astrodata.Errors import ScienceError
from gempy import managers as man

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
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix, 
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
                MDF.rename_ext('MDF',1)
                MDF.set_key_value('EXTNAME','MDF', 'Extension name')
                MDF.set_key_value('EXTVER',1,'Extension version')
                
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
        log.critical(repr(sys.exc_info()[1]))
        raise          
