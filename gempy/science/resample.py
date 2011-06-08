# This module contains user level functions related to resampling the input
# dataset

import sys
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as man
from gempy.geminiCLParDicts import CLDefaultParamsDict

def mosaic_detectors(adinput, tile=False, interpolator='linear'):
    """
    This function will mosaic the SCI frames of the input images, 
    along with the VAR and DQ frames if they exist.  
    
    WARNING: The gmosaic script used here replaces the previously 
    calculated DQ frames with its own versions.  This may be corrected 
    in the future by replacing the use of the gmosaic
    with a Python routine to do the frame mosaicing.
    
    NOTE: The inputs to this function MUST be prepared. 

    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adinput: Astrodata inputs to mosaic the extensions of
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param tile: Tile images instead of mosaic?
    :type tile: Python boolean (True/False)
    
    :param interpolator: type of interpolation algorithm to use for between 
                            the chip gaps.
    :type interpolator: string, options: 'linear', 'nearest', 'poly3', 
                           'poly5', 'spine3', 'sinc'.
    
    """
    
    # instantiate log
    log = gemLog.getGeminiLog()

    # ensure that adinput is not None and make it into a list
    # if it is not one already
    adinput = gt.validate_input(adinput=adinput)

    # time stamp keyword
    keyword = 'MOSAIC'
    
    # initialize output list
    adoutput_list = []    

    try:
        # load and bring the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()  
            
        for ad in adinput:

            # Clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')    

            # Determine whether VAR/DQ needs to be propagated
            if (ad.count_exts('VAR') == 
                ad.count_exts('DQ') == 
                ad.count_exts('SCI')):
                fl_vardq=yes
            else:
                fl_vardq=no

            # Prepare input files, lists, parameters... for input to 
            # the CL script
            clm=man.CLManager(imageIns=ad, suffix='_out', 
                              funcName='mosaicDetectors', log=log)
        
            # Check the status of the CLManager object, 
            # True=continue, False= issue warning
            if not clm.status: 
                raise Errors.ScienceError('One of the inputs has not been ' +
                                          'prepared, the ' + 
                                          'overscan_subtract_gmos function ' +
                                          'can only work on prepared data.')

            # Parameters set by the man.CLManager or the 
            # definition of the prim 
            clPrimParams = {
                # Retrieve the inputs as a string of filenames
                'inimages'    :clm.imageInsFiles(type='string'),
                'outimages'   :clm.imageOutsFiles(type='string'),
                # Set the value of FL_vardq set above
                'fl_vardq'    :fl_vardq,
                # This returns a unique/temp log file for IRAF 
                'logfile'     :clm.templog.name,               
                }
            # Parameters from the Parameter file adjustable by the user
            clSoftcodedParams = {
                # pyrafBoolean converts the python booleans to pyraf ones
                'fl_paste'    :gt.pyrafBoolean(tile),
                #'outpref'     :suffix,
                'geointer'    :interpolator,
                }
            # Grab the default params dict and update it with 
            # the two above dicts
            clParamsDict = CLDefaultParamsDict('gmosaic')
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)      
                
            # Log the parameters that were not defaults
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
            
            gemini.gmos.gmosaic(**clParamsDict)
    
            if gemini.gmos.gmosaic.status:
                raise Errors.ScienceError('gireduce failed for inputs '+
                             clm.imageInsFiles(type='string'))
            else:
                log.status('Exited the gmosaic CL script successfully')    
                
                
            # Rename CL outputs and load them back into memory 
            # and clean up the intermediate temp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL()   
            
            ad_out = imageOuts[0]
            ad_out.filename = ad.filename
                
            # Verify gireduce was actually run on the file
            # then log file names of successfully reduced files
            if ad_out.phu_get_key_value('GMOSAIC'): 
                log.fullinfo('\nFile '+ad_out.filename+\
                             ' mosaicked successfully')

            # Update GEM-TLM (automatic) and MOSAIC time stamps to the PHU
            # and update logger with updated/added time stamps
            gt.mark_history(adinput=ad_out, keyword=keyword)

            adoutput_list.append(ad_out)
                
        # Return the outputs list, even if there is only one output
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
