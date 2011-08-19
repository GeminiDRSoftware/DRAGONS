# This module contains user level functions related to the preprocessing of
# the input dataset with a flat frame

import sys
from copy import deepcopy
import numpy as np
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict
from gempy.science import resample as rs

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

def divide_by_flat(adinput=None, flat=None):
    """
    The divide_by_flat user level function will divide the science extension of
    the input science frames by the science extension of the input flat frames.
    The variance and data quality extension will be updated, if they exist.
    
    :param adinput: Astrodata inputs to have DQ extensions added to
    :type adinput: Astrodata
    
    :param flat: The flat(s) to divide the input(s) by. The flats can be a list
                 of AstroData objects or a single AstroData object. Note: If
                 there is multiple inputs and one flat provided, then the same
                 flat will be applied to all inputs; else the flats list must
                 match the length of the inputs.
    :type flat: AstroData
    """
    
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    flat = gt.validate_input(adinput=flat)
    
    # Create a dictionary that has the AstroData objects specified by adinput
    # as the key and the AstroData objects specified by flat as the value
    flat_dict = gt.make_dict(key_list=adinput, value_list=flat)
    
    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["divide_by_flat"]
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the divide_by_flat user level function has been
            # run previously
            if ad.phu_get_key_value(timestamp_key):
                raise Errors.InputError("%s has already been processed by " \
                                        "divide_by_flat" % (ad.filename))
            
            # Get the right flat frame for this input
            this_flat = flat_dict[ad]
            
            # Check for the case that the science data is a CCD2-only
            # frame and the flat is a full frame                
            if ad.count_exts("SCI")==1 and this_flat.count_exts("SCI")>1:
                sciext = ad["SCI",1]
                for flatext in this_flat["SCI"]:
                    # Use this extension if the flat detector section
                    # matches the science detector section
                    if (str(flatext.detector_section()) == 
                        str(sciext.detector_section())):
                        
                        extver = flatext.extver()
                        log.fullinfo("Using flat extension [SCI,%i]" % 
                                     extver)

                        varext = this_flat["VAR",extver]
                        dqext = this_flat["DQ",extver]

                        this_flat = deepcopy(flatext)
                        this_flat.rename_ext(name="SCI",ver=1)
                        if varext is not None:
                            newvar = deepcopy(varext)
                            newvar.rename_ext(name="VAR",ver=1)
                            this_flat.append(newvar)
                        if dqext is not None:
                            newdq = deepcopy(dqext)
                            newdq.rename_ext(name="DQ",ver=1)
                            this_flat.append(newdq)
                        break


            # Check the inputs have matching binning and SCI shapes.
            gt.checkInputsMatch(adInsA=ad, adInsB=this_flat, 
                                check_filter=True) 

            # Divide the adinput by the flat
            log.fullinfo("Dividing the input AstroData object (%s) " \
                         "by this flat:\n%s" % (ad.filename,
                                                this_flat.filename))
            ad = ad.div(this_flat)
            
            # Add the appropriate time stamps to the PHU
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

def normalize_image(adinput=None):
    """
    The normalize_image user level function will normalize each science
    extension of the input AstroData object(s) and automatically update the
    variance and data quality extensions, if they exist.
    
    :param adinput: Astrodata input flat(s) to be combined and normalized
    :type adinput: Astrodata
    """
    
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    
    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["normalize_image"]
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the normalize_image user level function has
            # been run previously
            if ad.phu_get_key_value(timestamp_key):
                raise Errors.InputError("%s has already been processed by " \
                                        "normalize_image" % (ad.filename))
            
            # Loop over each science extension in each input AstroData object
            for ext in ad["SCI"]:
                
                # Normalise the input AstroData object. Calculate the mean
                # value of the science extension
                mean = np.mean(ext.data)
                # Divide the science extension by the mean value of the science
                # extension
                log.fullinfo("Normalizing %s[%s,%d] by dividing by the mean " \
                             "= %f" % (ad.filename, ext.extname(),
                                       ext.extver(), mean))
                ext = ext.div(mean)
            
            # Add the appropriate time stamps to the PHU
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

def normalize_image_gmos(adinput=None, saturation=45000):
    """
    This function will calculate a normalization factor from statistics
    on CCD2, then divide by this factor and propagate variance accordingly.
    CCD2 is used because of the dome-like shape of the GMOS detector response:
    CCDs 1 and 3 have lower average illumination than CCD2, and that needs
    to be corrected for by the flat.
    
    :param adinput: Astrodata input flat(s) to be normalized
    :type adinput: Astrodata
    
    :param saturation: Defines saturation level for the raw frame, in ADU
    :type saturation: float. If None, the saturation_level descriptor is used.
    """
    
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    
    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["normalize_image_gmos"]
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        for ad in adinput:
            
            if saturation is None:
                saturation = ad.saturation_level()
                
            # Get all CCD2 data
            if ad.count_exts("SCI")==1:
                # Only one CCD present, assume it is CCD2
                sciext = ad["SCI",1]
            else:
                # Otherwise, take the second science extension

                # Tile the data into one CCD per science extension,
                # reordering if necessary
                temp_ad = deepcopy(ad)
                temp_ad = rs.tile_arrays(adinput=temp_ad)[0]

                sciext = temp_ad["SCI",2]

            central_data = sciext.data

            # Check units of CCD2; if electrons, convert saturation
            # limit from ADU to electrons. Also subtract overscan
            # level if needed
            overscan_level = sciext.get_key_value("OVERSCAN")
            if overscan_level is not None:
                saturation -= overscan_level
                log.fullinfo("Subtracting overscan level " +
                             "%.2f from saturation parameter" % overscan_level)
            bunit = sciext.get_key_value("BUNIT")
            if bunit=="electron":
                gain = sciext.gain().as_pytype()
                saturation *= gain 
                log.fullinfo("Saturation parameter converted to " +
                             "%.2f electrons" % saturation)
            
            # Take off 5% of the width as a border
            xborder = int(0.05 * central_data.shape[1])
            yborder = int(0.05 * central_data.shape[0])
            if xborder<20:
                xborder = 20
            if yborder<20:
                yborder = 20
            log.fullinfo("Using data section [%i:%i,%i:%i] for statistics" %
                         (xborder,central_data.shape[1]-xborder,
                          yborder,central_data.shape[0]-yborder))
            stat_region = central_data[yborder:-yborder,xborder:-xborder]
            
            # Remove negative and saturated values
            stat_region = stat_region[np.logical_and(stat_region>0,
                                                     stat_region<saturation)]
            
            # Find the mode and standard deviation
            hist,edges = np.histogram(stat_region, bins=saturation/0.1)
            mode = edges[np.argmax(hist)]
            std = np.std(stat_region)
            
            # Find the values within 3 sigma of the mode; the normalization
            # factor is the median of these values
            central_values = stat_region[
                np.logical_and(stat_region > mode - 3 * std,
                               stat_region < mode + 3 * std)]
            norm_factor = np.median(central_values)
            log.fullinfo("Normalization factor: %.2f" % norm_factor)
            
            # Divide by the normalization factor and propagate the
            # variance appropriately
            ad = ad.div(norm_factor)
            
            # Add the appropriate time stamp to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            adoutput_list.append(ad)
        
        # Return the output AstroData object
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################

