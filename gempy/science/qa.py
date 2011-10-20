# This module contains user level functions related to the quality assessment
# of the input dataset

import os
import sys
import math
import time
from copy import deepcopy
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

def iq_display_gmos_iqtool(adinput=None, frame=1, saturation=58000):

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function (use the one for measure_iq -- that's the only function
    # called that modifies the input data)
    timestamp_key = timestamp_keys["measure_iq"]

    # Initialize the list of output AstroData objects
    adoutput_list = []

    try:
        # Loop over each input AstroData object in the input list

        if frame is None:
            frame = 1
        display = True

        for ad in adinput:

            # Make a copy of the input, so that we can modify it
            # without affecting the original
            disp_ad = deepcopy(ad)

            # Tile the data into one science extension
            disp_ad = rs.tile_arrays(adinput=disp_ad,tile_all=True)

            # Measure IQ on the image
            log.stdinfo("Measuring FWHM of stars")
            disp_ad,stars = measure_iq_iqtool(adinput=disp_ad, 
                                       centroid_function="moffat",
                                       display=False, qa=True, 
                                       return_source_info=True)
            if len(stars)==0:
                adoutput_list.append(ad)
                frame+=1
                continue

            # Display the image with IQ stars marked
            if display:
                data_shape = disp_ad[0]["SCI",1].data.shape
                iqmask = _iq_overlay(stars,data_shape)

                log.stdinfo('Sources used to measure IQ are marked ' +
                            'with blue circles.')
                try:
                    disp_ad = ds.display_gmos(adinput=disp_ad,
                                         frame=frame,
                                         saturation=saturation,
                                         overlay=iqmask)
                except:
                    log.warning("Could not display %s" % disp_ad[0].filename)
                    display = False

            frame+=1
            
            # Update headers in original file
            mean_fwhm = disp_ad[0].phu_get_key_value("MEANFWHM")
            mean_ellp = disp_ad[0].phu_get_key_value("MEANELLP")
            if mean_fwhm is not None:
                ad.phu_set_key_value("MEANFWHM",mean_fwhm,
                                     comment=("Mean point source FWHM "+
                                              "(arcsec)"))
            if mean_ellp is not None:
                ad.phu_set_key_value("MEANELLP",mean_ellp,
                                     comment=("Mean point source "+
                                              "ellipticity"))
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Return the list of output AstroData objects
        return adoutput_list

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


def measure_iq_iqtool(adinput=None, centroid_function='moffat', display=False,
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
                try:
                    cx = float(fields[7])
                    cy = float(fields[8])
                    fwhm = float(fields[10])
                except:
                    continue
                stars.append({"x": cx,
                              "y": cy,
                              "fwhm": fwhm})

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

                log.stdinfo("%d sources used to measure IQ." % len(stars))

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
                    llen = 32
                    rlen = 24
                    dlen = llen+rlen
                    pm = '+/-'
                    fnStr = 'Filename: '+ad.filename
                    fmStr = ('FWHM Mean %s Sigma:' % pm).ljust(llen) + \
                            ('%.3f %s %.3f arcsec' % (iqdata[0][2], pm,
                                                      iqdata[0][3])).rjust(rlen)
                    emStr = ('Ellipticity Mean %s Sigma:' % pm).ljust(llen) + \
                            ('%.3f %s %.3f' % (iqdata[0][0], pm, 
                                               iqdata[0][1])).rjust(rlen)
                    csStr = ('Zenith-corrected FWHM (AM %.2f):'% \
                             airmass).ljust(llen) + \
                            ('%.3f arcsec' % corr).rjust(rlen)
                    if iq_band!='':
                        filter = ad.filter_name(pretty=True)
                        iqStr = ('IQ band for %s filter:'%filter).ljust(llen)+\
                                iq_band.rjust(rlen)
                    else:
                        iqStr = '(IQ band could not be determined)'

                    # Warn if high ellipticity
                    if iqdata[0][0]>0.1:
                        ell_warn = "\n    "+\
                                   "WARNING: high ellipticity".rjust(dlen)
                    else:
                        ell_warn = ""                    

                    # Create final formatted string
                    finalStr = '\n    '+fnStr+'\n    '+'-'*dlen+\
                               '\n    '+fmStr+'\n    '+emStr+\
                               '\n    '+csStr+'\n    '+iqStr+ell_warn+\
                               '\n    '+'-'*dlen+'\n'
                    # Log final string
                    if display:
                        log.stdinfo('Sources used to measure IQ are marked ' +
                                    'with blue circles.')
                    log.stdinfo(finalStr, category='IQ')
                
                    # Store average FWHM and ellipticity to header
                    ad.phu_set_key_value("MEANFWHM", iqdata[0][2],
                                         comment=("Mean point source FWHM "+
                                                  "(arcsec)"))
                    ad.phu_set_key_value("MEANELLP", iqdata[0][0],
                                         comment=("Mean point source "+
                                                  "ellipticity"))

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


def iq_display_gmos(adinput=None, display=True, frame=1, saturation=58000):

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function (use the one for measure_iq -- that's the only function
    # called that modifies the input data)
    timestamp_key = timestamp_keys["measure_iq"]

    # Initialize the list of output AstroData objects
    adoutput_list = []

    try:
        # Loop over each input AstroData object in the input list

        if frame is None:
            frame = 1
        display = True

        for ad in adinput:

            # Make a copy of the input, so that we can modify it
            # without affecting the original
            disp_ad = deepcopy(ad)

            # Tile the data into one science extension
            disp_ad = rs.tile_arrays(adinput=disp_ad,tile_all=True)

            # Measure IQ on the image
            disp_ad,stars = measure_iq(adinput=disp_ad,
                                       return_source_info=True)

            # Display the image with IQ stars marked
            if display:
                data_shape = disp_ad[0]["SCI",1].data.shape

                if len(stars)==0:
                    iqmask = None
                else:
                    iqmask = _iq_overlay(stars,data_shape)

                    log.stdinfo('Sources used to measure IQ are marked ' +
                                'with blue circles.')

                try:
                    disp_ad = ds.display_gmos(adinput=disp_ad[0],
                                              frame=frame,
                                              saturation=saturation,
                                              overlay=iqmask)
                except:
                    log.warning("Could not display %s" % disp_ad[0].filename)
                    display = False

            frame+=1
            
            # Update headers in original file
            mean_fwhm = disp_ad[0].phu_get_key_value("MEANFWHM")
            mean_ellp = disp_ad[0].phu_get_key_value("MEANELLP")
            if mean_fwhm is not None:
                ad.phu_set_key_value("MEANFWHM",mean_fwhm,
                                     comment=("Mean point source FWHM "+
                                              "(arcsec)"))
            if mean_ellp is not None:
                ad.phu_set_key_value("MEANELLP",mean_ellp,
                                     comment=("Mean point source "+
                                              "ellipticity"))
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Return the list of output AstroData objects
        return adoutput_list

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


def measure_iq(adinput=None, return_source_info=False, separate_ext=False):
    """
    This function is for use with sextractor-style source-detection.
    FWHM are already in OBJCAT; this function does the clipping and
    reporting only.
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
        
        # Loop over each input AstroData object
        for ad in adinput:

            # Clip sources from the OBJCAT
            good_source = _clip_sources(ad,separate_ext=separate_ext)

            if len(good_source.keys())==0:
                log.warning('No good sources found in %s' % ad.filename)
                gt.mark_history(adinput=ad, keyword=timestamp_key)
                adoutput_list.append(ad)
                good_source['all'] = []
                continue

            for key in good_source:
                src = good_source[key]

                if len(src)==0:
                    log.warning('No good sources found in %s, %s extensions' %
                                (ad.filename,key))
                    gt.mark_history(adinput=ad, keyword=timestamp_key)
                    continue

                # Clipped mean of FWHM and ellipticity
                if len(src)>1:
                    mean_fwhm,std_fwhm,mean_ellip,std_ellip = _clipped_mean(src)
                elif len(src)==1:
                    log.warning('Only one source found. IQ numbers may ' +
                                'not be accurate.')
                    mean_fwhm = src[0]['fwhm']
                    std_fwhm = np.nan
                    mean_ellip = src[0]['ellipticity']
                    std_ellip = np.nan
                    
                log.stdinfo("%d sources used to measure IQ." % len(src))

                airmass = float(ad.airmass())
                if airmass is None:
                    log.warning("Airmass not found, not correcting to zenith")
                    corr = mean_fwhm
                else:
                    corr = mean_fwhm * airmass**(-0.6)

                # Get IQ constraint band corresponding to
                # the corrected FWHM number
                iq_band = _iq_band(adinput=ad,fwhm=corr)[0]

                # Format output for printing or logging                
                llen = 32
                rlen = 24
                dlen = llen+rlen
                pm = '+/-'
                fnStr = 'Filename: %s' % ad.filename
                if separate_ext:
                    fnStr += "[%s,%s]" % key
                fmStr = ('FWHM Mean %s Sigma:' % pm).ljust(llen) + \
                        ('%.3f %s %.3f arcsec' % (mean_fwhm, pm,
                                                  std_fwhm)).rjust(rlen)
                emStr = ('Ellipticity Mean %s Sigma:' % pm).ljust(llen) + \
                        ('%.3f %s %.3f' % (mean_ellip, pm, 
                                           std_ellip)).rjust(rlen)
                csStr = ('Zenith-corrected FWHM (AM %.2f):'%airmass).ljust(llen) + \
                        ('%.3f arcsec' % corr).rjust(rlen)
                if iq_band!='':
                    filter = ad.filter_name(pretty=True)
                    iqStr = ('IQ band for %s filter:'%filter).ljust(llen)+\
                            iq_band.rjust(rlen)
                else:
                    iqStr = '(IQ band could not be determined)'

                # Warn if high ellipticity
                if mean_ellip>0.1:
                    ell_warn = "\n    "+\
                               "WARNING: high ellipticity".rjust(dlen)
                else:
                    ell_warn = ""                    

                # Create final formatted string
                finalStr = '\n    '+fnStr+'\n    '+'-'*dlen+\
                           '\n    '+fmStr+'\n    '+emStr+\
                           '\n    '+csStr+'\n    '+iqStr+ell_warn+\
                           '\n    '+'-'*dlen+'\n'
                # Log final string
                log.stdinfo(finalStr, category='IQ')
                
                # Store average FWHM and ellipticity to header
                if separate_ext:
                    sciext = ad[key]
                    sciext.set_key_value("MEANFWHM", mean_fwhm,
                                         comment=("Mean point source FWHM "+
                                                  "(arcsec)"))
                    sciext.set_key_value("MEANELLP", mean_ellip,
                                         comment=("Mean point source "+
                                                  "ellipticity"))
                else:
                    ad.phu_set_key_value("MEANFWHM", mean_fwhm,
                                         comment=("Mean point source FWHM "+
                                                  "(arcsec)"))
                    ad.phu_set_key_value("MEANELLP", mean_ellip,
                                         comment=("Mean point source "+
                                                  "ellipticity"))

                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        if return_source_info:
            if separate_ext:
                return adoutput_list, good_source
            else:
                return adoutput_list, good_source['all']
        else:
            return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


def measure_zp(adinput=None):
    """
    This function is for use with sextractor-style source-detection.
    It relies on having already added a reference catalog and done the
    cross match to populate the refmag column of the objcat
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["measure_zp"]

    # Initialize the list of output AstroData objects
    adoutput_list = []

    try:

        # Loop over each input AstroData object
        for ad in adinput:

            # Loop over OBJCATs extensions
            for objcat in ad['OBJCAT']:
                extver = objcat.extver()

                mags = objcat.data['mag']

                # Need to correct the mags for the exposure time
                et = float(ad.exposure_time())
                magcor = 2.5*math.log10(et)
                mags += magcor

                # FIXME: Need to determine if we're in electrons or ADUs and correct
                # the mags for the gain if we're in ADU here. ZPs are in electrons.

                # FIXME: Need to apply the appropreate nominal extinction correction here

                refmags = objcat.data['refmag']

                zps = refmags - mags
 
                zps = np.where((zps > -500), zps, None)

                zps = zps[np.flatnonzero(zps)]

                mean = np.mean(zps)
                sigma = np.std(zps)

                log.fullinfo("Unclipped zeropoint measurement: %f +/- %f" % (mean, sigma))

                zps = np.where(((zps < mean+sigma) & (zps > mean-sigma)), zps, None)
                zps = zps[np.flatnonzero(zps)]
                mean = np.mean(zps)
                sigma = np.std(zps)
                log.stdinfo("Filename: %s" % ad.filename)
                log.stdinfo("--------------------------------------------------------")
                log.stdinfo("%d sources used to measure Zeropoint" % len(zps))
                log.stdinfo("Zeropoint measurement (%s band): %.3f +/- %.3f" % (ad.filter_name(pretty=True), mean, sigma))
                log.stdinfo("--------------------------------------------------------")

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

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
                            iq='IQ20 (<%.2f arcsec)' % iq20
                        elif fwhm[count]<iq70:
                            iq='IQ70 (%.2f-%.2f arcsec)' % (iq20,iq70)
                        elif fwhm[count]<iq85:
                            iq='IQ85 (%.2f-%.2f arcsec)' % (iq70,iq85)
                        else:
                            iq='IQAny (>%.2f arcsec)' % iq85
            
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
        x0 = star['x']
        y0 = star['y']
        #radius = star['fwhm']
        radius = 16
        r2 = radius*radius
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
    

def _clip_sources(ad, separate_ext=False):
    """
    This function takes the source data from the OBJCAT and returns the best
    sources for IQ measurement.
    
    :param ad: input image
    :type ad: AstroData instance with OBJCAT attached

    :param separate_ext: Flag to treat each extension separately
    :type separate_ext: boolean
    """

    good_source = {}
    first = True
    for sciext in ad["SCI"]:
        extver = sciext.extver()

        objcat = ad["OBJCAT",extver]
        if objcat is None:
            continue
        if objcat.data is None:
            continue

        try:
            x = objcat.data.field("x")
            y = objcat.data.field("y")
            fwhm_pix = objcat.data.field("fwhm_pix")
            fwhm_arcsec = objcat.data.field("fwhm_arcsec")
            ellip = objcat.data.field("ellipticity")
            flags = objcat.data.field("flags")
            class_star = objcat.data.field("class_star")
        except:
            continue
        
        good = (flags==0) & (class_star>0.6) & (ellip<0.5)

        rec = np.rec.fromarrays([x[good],y[good],fwhm_pix[good],
                                 fwhm_arcsec[good],ellip[good]],
                                names=["x","y","fwhm","fwhm_arcsec","ellipticity"])

        if separate_ext:
            good_source[("SCI",extver)] = rec
        else:
            if first:
                good_source["all"] = rec
                first = False
            else:
                good_source["all"] = np.hstack([good_source["all"],rec])

    return good_source

def _clipped_mean(src):
    """
    Clipping is done on FWHM
    """

    fwhm = src['fwhm_arcsec']
    ellip = src['ellipticity']

    mean=0.7
    sigma=1
    num_total = len(fwhm)
    if num_total < 3:
        return np.mean(fwhm),np.std(fwhm),np.mean(ellip),np.std(ellip)

    num = num_total
    clip=0
    while (num > 0.5*num_total):

        num=0

        # for fwhm
        sum = 0
        sumsq = 0

        # for ellipticity
        esum = 0
        esumsq = 0

        upper = mean+sigma
        lower = mean-(3*sigma)

        i = 0
        for f in fwhm:
            e = ellip[i]
            if(f<upper and f>lower):
                sum += f
                sumsq += f*f
                esum += e
                esumsq += e*e
                num+=1
            i+=1

        if num>0:
            # for fwhm
            mean = sum / num
            var = (sumsq / num) - mean*mean
            if var>=0:
                sigma = math.sqrt(var)
            else:
                var = np.nan

            # for ellipticity
            emean = esum / num
            evar = (esumsq / num) - emean*emean
            if evar>=0:
                esigma = math.sqrt(evar)
            else:
                esigma = np.nan

        elif clip==0:
            return np.mean(fwhm),np.std(fwhm),np.mean(ellip),np.std(ellip)
        else:
            break

        clip+=1
        if clip>10:
            break

    return mean,sigma,emean,esigma
