# This module contains user level functions related to the quality assessment
# of the input dataset

import os
import sys
import math
import time
from datetime import datetime
import numpy as np
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import display as ds
from gempy.science import resample as rs

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

def measure_iq(adinput=None, centroid_function='moffat', display=False,
               qa=True, return_source_info=False):
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
    timestamp_key = timestamp_keys["measure_iq"]

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
            ad.write(tmpWriteName, rename=False, clobber=True)
            
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
            
            # If input was written to temp file on disk, delete it
            if os.path.exists(tmpWriteName):
                os.remove(tmpWriteName)
                log.fullinfo('Temporary file ' +
                             tmpWriteName+ ' removed from disk.')
            
            # Get star information from the .dat file on disk
            # NOTE: this information should be stored in the OBJCAT
            stars = []
            datName = os.path.splitext(tmpWriteName)[0]+'.dat'
            dat_fh = open(datName)
            for line in dat_fh:
                line = line.strip()
                if line=='' or line.startswith("#") or line.startswith("A"):
                    continue
                fields = line.split()
                cx = float(fields[7])
                cy = float(fields[8])
                fwhm = float(fields[10])
                stars.append({"CooX": cx,
                              "CooY": cy,
                              "FWHMpix": fwhm})

            # Delete the .dat file
            if os.path.exists(datName):
                os.remove(datName)
                log.fullinfo('Temporary file '+
                             datName+ ' removed from disk.')
                
            # iqdata is list of tuples with image quality metrics
            # (ellMean, ellSig, fwhmMean, fwhmSig)
            # First check if it is empty (ie. gemiq failed in some way)
            if len(iqdata) == 0:
                log.warning('Problem Measuring IQ Statistics, '+
                            'none reported')
            # If it all worked, then format the output and log it
            else:

                # correct the seeing measurement to zenith
                fwhm = iqdata[0][2]
                airmass = float(ad.airmass())
                if fwhm is None:
                    log.warning("No good seeing measurement found.")
                elif airmass is None:
                    log.warning("Airmass not found, not correcting to zenith")
                    corr = fwhm
                else:
                    corr = fwhm * airmass**(-0.6)

                    # Get IQ constraint band corresponding to
                    # the corrected FWHM number
                    iq_band = _iq_band(adinput=ad,fwhm=corr)[0]

                    # Format output for printing or logging                
                    llen = 27
                    dlen = 32
                    fnStr = 'Filename: '+ad.filename
                    emStr = 'Ellipticity Mean:'.ljust(llen)+'%.3f'%iqdata[0][0]
                    esStr = 'Ellipticity Sigma:'.ljust(llen)+'%.3f'%iqdata[0][1]
                    fmStr = 'FWHM Mean:'.ljust(llen)+'%.3f'%iqdata[0][2]
                    fsStr = 'FWHM Sigma:'.ljust(llen)+'%.3f'%iqdata[0][3]
                    csStr = 'Zenith-corrected FWHM:'.ljust(llen)+'%.3f'%corr
                    if iq_band!='':
                        filter = ad.filter_name(pretty=True)
                        iqStr = 'IQ band for %s filter:' % filter
                        iqStr = iqStr.ljust(llen) + ('IQ'+iq_band).rjust(5)
                    else:
                        iqStr = '(IQ band could not be determined)'
                    # Create final formatted string
                    finalStr = '\n    '+fnStr+'\n    '+'-'*dlen+'\n    '+emStr+\
                               '\n    '+esStr+'\n    '+fmStr+'\n    '+fsStr+\
                               '\n    '+csStr+'\n    '+iqStr+'\n    '+\
                               '-'*dlen+'\n'
                    # Log final string
                    if display:
                        log.stdinfo('Sources used to measure IQ are marked ' +
                                    'with blue circles.')
                    log.stdinfo(finalStr, category='IQ')
                
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Logging the total amount of time spent measuring the IQ of all
        # the inputs
        log.debug('Total measureIQ time: '+repr(total_IQ_time), 
                    category='IQ')
        
        # Return the list of output AstroData objects
        if return_source_info:
            return adoutput_list, stars
        else:
            return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def iq_display_gmos(adinput=None, frame=1):

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Initialize the list of output AstroData objects
    adoutput_list = []

    try:
        # Loop over each input AstroData object in the input list

        if frame is None:
            frame = 1
        display = True

        for ad in adinput:

            # Tile the data into one science extension
            ad = rs.tile_arrays(adinput=ad,tile_all=True)

            # Measure IQ on the image
            log.stdinfo("Measuring FWHM of stars")
            ad,stars = measure_iq(adinput=ad, 
                                  centroid_function="moffat",
                                  display=False, qa=True, 
                                  return_source_info=True)
            if len(stars)==0:
                frame+=1
                continue

            # Display the image with IQ stars marked
            if display:
                data_shape = ad[0]["SCI",1].data.shape
                iqmask = _iq_overlay(stars,data_shape)

                log.stdinfo('Sources used to measure IQ are marked ' +
                            'with blue circles.')
                try:
                    ad = ds.display_gmos(adinput=ad,
                                         frame=frame,
                                         saturation=58000,
                                         overlay=iqmask)
                except:
                    log.warning("Could not display %s" % ad[0].filename)
                    display = False

            frame+=1
            

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Return the list of output AstroData objects
        return adoutput_list

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################


def _iq_band(adinput=None,fwhm=None):
    """
    Helper function to take WFS, filter, and airmass information from
    an AstroData instance and use it to convert a seeing FWHM into
    an IQ constraint band.

    :param adinput: Input images for which the FWHM has been measured
    :type adinput: Astrodata objects, either a single or a list of objects

    :param fwhm: Measured FWHM of stars in image, in arcsec
    :type fwhm: float, either a single or a list that matches the length
                of the adinput list
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    fwhm = gt.validate_input(adinput=fwhm)

    # Initialize the list of output IQ band strings
    str_output = []

    try:

        # get the IQ band definitions from a lookup table
        iqConstraints = Lookups.get_lookup_table("Gemini/IQConstraints",
                                                 "iqConstraints")

        # if there is only one FWHM listed for all adinput, copy
        # that entry into a list that matches the length of the
        # adinput list
        if len(fwhm)==1:
            fwhm = [fwhm[0] for ad in adinput]
        else:
            # otherwise check that length of fwhm list matches 
            # the length of the adinput list
            if len(adinput)!=len(fwhm):
                raise Errors.InputError("fwhm list must match length of " +
                                        "adinput list")
        
        # Loop over each input AstroData object in the input list
        count=0
        for ad in adinput:

            try:
                wfs = str(ad.wavefront_sensor())
            except:
                wfs = None
            try:
                filter = str(ad.filter_name(pretty=True))
            except:
                filter = None

            # default value for iq band
            iq = ''

            # check that ad has valid WFS, filter
            if wfs is not None and filter is not None:
                if filter in iqConstraints.keys():
                    if wfs in iqConstraints[filter].keys():

                        # get limits for this observation
                        iq20 = iqConstraints[filter][wfs]['20']
                        iq70 = iqConstraints[filter][wfs]['70']
                        iq85 = iqConstraints[filter][wfs]['85']

                        # get iq band
                        if fwhm[count]<iq20:
                            iq='20'
                        elif fwhm[count]<iq70:
                            iq='70'
                        elif fwhm[count]<iq85:
                            iq='85'
                        else:
                            iq='Any'
            
            # Append the iq band string to the output
            str_output.append(iq)
            count+=1

        # Return the list of output AstroData objects
        return str_output

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def _iq_overlay(stars,data_shape):
    """
    Generates a tuple of numpy arrays that can be used to mask a display with
    circles centered on the stars' positions and radii that reflect the
    measured FWHM.
    Eg. data[iqmask] = some_value

    The circle definition is based on numdisplay.overlay.circle, but circles
    are two pixels wide to make them easier to see.
    """

    xind = []
    yind = []
    width = data_shape[1]
    height = data_shape[0]
    for star in stars:
        x0 = star["CooX"]
        y0 = star["CooY"]
        radius = star["FWHMpix"]
        r2 = radius**2
        quarter = int(math.ceil(radius * math.sqrt (0.5)))

        for dy in range(-quarter,quarter+1):
            if r2>dy**2:
                dx = math.sqrt(r2 - dy**2)
            else:
                dx = 0
            j = int(round(dy+y0))
            i = int(round(x0-dx))           # left arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i-2])
                yind.extend([j-1,j-1])
            i = int(round(x0+dx))           # right arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i])
                yind.extend([j-1,j-1])

        for dx in range(-quarter, quarter+1):
            if r2>dx**2:
                dy = math.sqrt(r2 - dx**2)
            else:
                dy = 0
            i = int(round(dx + x0))
            j = int(round(y0 - dy))           # bottom arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i-1])
                yind.extend([j-1,j-2])
            j = int (round (y0 + dy))           # top arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i-1])
                yind.extend([j-1,j])

    iqmask = (np.array(yind),np.array(xind))
    return iqmask
    
