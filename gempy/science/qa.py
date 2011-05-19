# This module contains user level functions related to the quality assessment
# of the input dataset

import os, sys
import time
from datetime import datetime
from astrodata import Errors
from gempy import managers as man

def measure_iq(adInputs, function='both', display=True, qa=True,
               keepDats=False):
    """
    This function will detect the sources in the input images and fit
    both Gaussian and Moffat models to their profiles and calculate the 
    Image Quality and seeing from this.
    
    Since the resultant parameters are formatted into one nice string and 
    normally recorded in a logger message, the returned dictionary of these 
    parameters may be ignored. 
    The dictionary's format is:
    {adIn1.filename:formatted results string for adIn1, 
    adIn2.filename:formatted results string for adIn2,...}
    
    There are also .dat files that result from this function written to the 
    current working directory under the names 'measure_iq'+adIn.filename+'.dat'.
    ex: input filename 'N20100311S0090.fits', 
    .dat filename 'measure_iqN20100311S0090.dat'
    
    NOTE:
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    Warning:
    ALL inputs of adInputs must have either 1 SCI extension, indicating they 
    have been mosaic'd, or 3 like a normal un-mosaic'd GMOS image.
    
    :param adInputs: Astrodata inputs to have their image quality measured
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param function: Function for centroid fitting
    :type function: string, can be: 'moffat','gauss' or 'both'; 
                    Default 'both'
                    
    :param display: Flag to turn on displaying the fitting to ds9
    :type display: Python boolean (True/False)
                   Default: True
                  
    :param qa: flag to use a grid of sub-windows for detecting the sources in 
               the image frames, rather than the entire frame all at once.
    :type qa: Python boolean (True/False)
              default: True
    
    :param keepDats: flag to keep the .dat files that provide detailed results 
                     found while measuring the input's image quality.
    :type keepDats: Python boolean (True/False)
                    default: False
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                    (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    
    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, None, 'tmp', 
                                      funcName='measure_iq') 
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    # NOTE: outNames are not needed, but sfm.startUp creates them automatically.
    adInputs, outNames, log = sfm.startUp()
    
    
    try:
        # Importing getiq module to perform the source detection and IQ
        # measurements of the inputs
        from iqtool.iq import getiq
        
        # Initializing a total time sum variable for logging purposes 
        total_IQ_time = 0
        
        # Creating dictionary for output strings to be returned in
        outDict = {}
        
        # Loop through the inputs to perform the non-linear and saturated
        # pixel searches of the SCI frames to update the BPM frames into
        # full DQ frames. 
        for ad in adInputs:                     
            # Writing the input to disk under a temp name in the current 
            # working directory for getiq to use to be deleted after getiq
            tmpWriteName = 'measure_iq'+os.path.basename(ad.filename)
            log.fullinfo('The inputs to measureIQ must be in the'+
                         ' current working directory for it to work '+\
                         'correctly, so writting it temperarily to file '+
                         tmpWriteName)
            ad.write(tmpWriteName, rename=False)
            
            # Automatically determine the 'mosaic' parameter for gemiq
            # if there are 3 SCI extensions -> mosaic=False
            # if only one -> mosaic=True, else raise error
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
            
            # Calling the gemiq function to detect the sources and then
            # measure the IQ of the current image 
            iqdata = getiq.gemiq(tmpWriteName, function=function, 
                                  verbose=True, display=display, 
                                  mosaic=mosaic, qa=qa,
                                  debug=True)####
            
            # End time for measuring IQ of current file
            et = time.time()
            total_IQ_time = total_IQ_time + (et - st)
            # Logging the amount of time spent measuring the IQ 
            log.debug('MeasureIQ time: '+repr(et - st), category='IQ')
            log.fullinfo('~'*45, category='format')
            
            # If input was writen to temp file on disk, delete it
            if os.path.exists(tmpWriteName):
                os.remove(tmpWriteName)
                log.fullinfo('The temporarily written to disk file, '+
                             tmpWriteName+ ', was removed from disk.')
            
            # Deleting the .dat file from disk if requested
            if not keepDats:
                datName = os.path.splitext(tmpWriteName)[0]+'.dat'
                os.remove(datName)
                log.fullinfo('The temporarily written to disk file, '+
                             datName+ ', was removed from disk.')
                
            # iqdata is list of tuples with image quality metrics
            # (ell_mean, ellSig, fwhmMean, fwhmSig)
            # First check if it is empty (ie. gemiq failed in someway)
            if len(iqdata) == 0:
                log.warning('Problem Measuring IQ Statistics, '+
                            'none reported')
            # If it all worked, then format the output and log it
            else:
                # Formatting this output for printing or logging                
                fnStr = 'Filename:'.ljust(19)+ad.filename
                emStr = 'Ellipticity Mean:'.ljust(19)+str(iqdata[0][0])
                esStr = 'Ellipticity Sigma:'.ljust(19)+str(iqdata[0][1])
                fmStr = 'FWHM Mean:'.ljust(19)+str(iqdata[0][2])
                fsStr = 'FWHM Sigma:'.ljust(19)+str(iqdata[0][3])
                sStr = 'Seeing:'.ljust(19)+str(iqdata[0][2])
                psStr = 'PixelScale:'.ljust(19)+str(ad.pixel_scale()[('SCI',1)])
                vStr = 'VERSION:'.ljust(19)+'None' #$$$$$ made on ln12 of ReductionsObjectRequest.py, always 'None' it seems.
                tStr = 'TIMESTAMP:'.ljust(19)+str(datetime.now())
                # Create final formated string
                finalStr = '-'*45+'\n'+fnStr+'\n'+emStr+'\n'+esStr+'\n'\
                                +fmStr+'\n'+fsStr+'\n'+sStr+'\n'+psStr+\
                                '\n'+vStr+'\n'+tStr+'\n'+'-'*45
                # Log final string
                log.stdinfo(finalStr, category='IQ')
                
                # appending formated string to the output dictionary
                outDict[ad.filename] = finalStr
                
        # Logging the total amount of time spent measuring the IQ of all
        # the inputs
        log.debug('Total measureIQ time: '+repr(total_IQ_time), 
                    category='IQ')
        
        #returning complete dictionary for use by the user if desired
        return outDict
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise
