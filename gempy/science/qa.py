# This module contains user level functions related to the quality assessment
# of the input dataset

import os, sys
import time
from datetime import datetime
from astrodata import Errors
from astrodata.adutils import gemLog
from gempy import geminiTools as gt

def measure_iq(adinput=None, centroid_function='moffat', display=False, qa=True):
    """
    This function will detect the sources in the input images and fit
    either Gaussian or Moffat models to their profiles and calculate the 
    Image Quality and seeing from this.
    
    Since the resultant parameters are formatted into one nice string and 
    normally recorded in a logger message, the returned dictionary of these 
    parameters may be ignored. 
    The dictionary's format is:
    {adIn1.filename:formatted results string for adIn1, 
    adIn2.filename:formatted results string for adIn2,...}
    
    NOTE:
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created
    and used within this function.
    
    Warning:
    ALL inputs of adinput must have either 1 SCI extension, indicating they 
    have been mosaic'd, or 3 like a normal un-mosaic'd GMOS image.
    
    :param adinput: Astrodata inputs to have their image quality measured
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param centroid_function: Function for centroid fitting
    :type centroid_function: string, can be: 'moffat','gauss' or 'both'; 
                    Default: 'moffat'
                    
    :param display: Flag to turn on displaying the fitting to ds9
    :type display: Python boolean (True/False)
                   Default: False
                  
    :param qa: flag to use a limited number of sources to estimate the IQ
               NOTE: in the future, this parameters should be replaced with
               a max_sources parameter that determines how many sources to
               use.
    :type qa: Python boolean (True/False)
              default: True
    
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "MEASREIQ"

    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:
        # Import the getiq module to perform the source detection and IQ
        # measurements of the inputs
        from iqtool.iq import getiq
        
        # Initialize a total time sum variable for logging purposes 
        total_IQ_time = 0
        
        # Loop over each input AstroData object
        for ad in adinput:
            # Write the input to disk under a temp name in the current 
            # working directory for getiq to use (to be deleted after getiq)
            tmpWriteName = 'tmp_measure_iq'+os.path.basename(ad.filename)
            log.fullinfo('Writing input temporarily to file '+
                         tmpWriteName)
            ad.write(tmpWriteName, rename=False)
            
            # Automatically determine the 'mosaic' parameter for gemiq
            # if there are 3 SCI extensions -> mosaic=False
            # if only one -> mosaic=True, else raise error
            # NOTE: this may be a problem for the new GMOS-N CCDs
            numExts = ad.count_exts('SCI')
            if numExts==1:
                mosaic = True
            elif numExts==3:
                mosaic = False
            else:
                raise Errors.ScienceError('The input '+ad.filename+' had '+\
                                   str(numExts)+' SCI extensions and inputs '+
                                   'with only 1 or 3 extensions are allowed')
            
            # Start time for measuring IQ of current file
            st = time.time()
            
            log.debug('Calling getiq.gemiq for input '+ad.filename)
            
            # Call the gemiq function to detect the sources and then
            # measure the IQ of the current image 
            iqdata = getiq.gemiq(tmpWriteName, function=centroid_function, 
                                 display=display, mosaic=mosaic, qa=qa,
                                 verbose=False, debug=False)
            
            # End time for measuring IQ of current file
            et = time.time()
            total_IQ_time = total_IQ_time + (et - st)
            # Log the amount of time spent measuring the IQ 
            log.debug('MeasureIQ time: '+repr(et - st), category='IQ')
            log.fullinfo('~'*45, category='format')
            
            # If input was written to temp file on disk, delete it
            if os.path.exists(tmpWriteName):
                os.remove(tmpWriteName)
                log.fullinfo('Temporary file ' +
                             tmpWriteName+ ' removed from disk.')
            
            # Delete the .dat file from disk
            # NOTE: this information should be stored in the OBJCAT
            datName = os.path.splitext(tmpWriteName)[0]+'.dat'
            os.remove(datName)
            log.fullinfo('Temporary file '+
                         datName+ ' removed from disk.')
                
            # iqdata is list of tuples with image quality metrics
            # (ell_mean, ellSig, fwhmMean, fwhmSig)
            # First check if it is empty (ie. gemiq failed in some way)
            if len(iqdata) == 0:
                log.warning('Problem Measuring IQ Statistics, '+
                            'none reported')
            # If it all worked, then format the output and log it
            else:
                # Format this output for printing or logging                
                fnStr = 'Filename:'.ljust(19)+ad.filename
                fmStr = 'FWHM Mean:'.ljust(19)+str(iqdata[0][2])
                fsStr = 'FWHM Sigma:'.ljust(19)+str(iqdata[0][3])
                emStr = 'Ellipticity Mean:'.ljust(19)+str(iqdata[0][0])
                esStr = 'Ellipticity Sigma:'.ljust(19)+str(iqdata[0][1])
                # Create final formatted string
                finalStr = '-'*45+'\n'+fnStr+'\n'+fmStr+'\n'+fsStr+'\n'\
                                +emStr+'\n'+esStr+'\n'+'-'*45
                # Log final string
                log.stdinfo(finalStr, category='IQ')
                
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=keyword)

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Logging the total amount of time spent measuring the IQ of all
        # the inputs
        log.debug('Total measureIQ time: '+repr(total_IQ_time), 
                    category='IQ')
        
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
