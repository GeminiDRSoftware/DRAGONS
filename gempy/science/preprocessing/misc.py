# This module contains miscellaneous user level functions related to the
# preprocessing of the input dataset

import sys
import numpy as np
from astrodata import Errors
from astrodata.adutils import gemLog
from gempy import geminiTools as gt

def adu_to_electrons(adinput):
    """
    This function will convert the inputs from having pixel values in ADU to 
    that of electrons by use of the arith 'toolbox'.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.

    Note: 
    the SCI extensions of the input AstroData objects must have 'GAIN'
    header key values available to multiply them by for conversion to 
    e- units.
    
    :param adinput: Astrodata inputs to be converted to Electron pixel units
    :type adinput: Astrodata objects, either a single or a list of objects
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(input=adinput)
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "ADUTOELE"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the adu_to_electrons user level function has been
            # run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "adu_to_electrons" % (ad.filename))
            # Get the gain value using the appropriate descriptors
            gain = ad.gain().as_pytype()
            # Multiply the science extension by the gain and the variance
            # extension by the gain squared
            log.info("Converting %s from ADU to electrons by multiplying " \
                     "the science extension by the gain = %s" % \
                     (ad.filename, gain))
            ad = ad.mult(gain)
            # Update the physical units keyword in the header of the AstroData
            # object. Formatting so logger looks organized for these messages
            log.fullinfo("*"*50, category="header")
            log.fullinfo("File = %s" % ad.filename, category="header")
            log.fullinfo("~"*50, category="header")
            # Data now has units of electrons
            gt.update_key_value(adinput=ad, function="bunit",
                                value="electron", extname="SCI")
            gt.update_key_value(adinput=ad, function="bunit",
                                value="electron*electron", extname="VAR")
            log.fullinfo("-"*50, category="header") 
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=keyword)
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def nonlinearity_correct(adinput=None):
    """
    Run on raw or nprepared Gemini NIRI data, this script calculates and
    applies a per-pixel linearity correction based on the counts in the pixel,
    the exposure time, the read mode, the bias level and the ROI. Pixels over
    the maximum correctable value are set to BADVAL unless given the force
    flag. Note that you may use glob expansion in infile, however, any pattern
    matching characters (*,?) must be either quoted or escaped with a
    backslash. Do we need a badval parameter that defines a value to assign to
    uncorrectable pixels, or do we want to just add those pixels to the DQ
    plane with a specific value?
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = gt.validate_input(input=adinput)
    # Define the keyword to be used for the time stamp for this user level
    # function
    keyword = "LINCORR"
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            # Check whether the nonlinearity_correct user level function has
            # been run previously
            if ad.phu_get_key_value(keyword):
                raise Errors.InputError("%s has already been processed by " \
                                        "nonlinearity_correct" % (ad.filename))
            # Get the appropriate information using the descriptors
            coadds = ad.coadds()
            read_mode = ad.read_mode().as_pytype()
            total_exposure_time = ad.exposure_time()
            well_depth_setting = ad.well_depth_setting().as_pytype()
            if coadds is None or read_mode is None or \
                total_exposure_time is None or well_depth_setting is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the
                # exception.
                if hasattr(ad, "exception_info"):
                    raise ad.exception_info
            # Check the raw exposure time (i.e., per coadd). First, convert
            # the total exposure time returned by the descriptor back to
            # the raw exposure time
            exposure_time = total_exposure_time / coadds
            if exposure_time > 600.:
                log.critical("The raw exposure time is outside the " + \
                    "range used to derive correction.")
                raise Errors.InvalidValueError()
            # Check the read mode and well depth setting values
            if read_mode == "Invalid" or well_depth_setting == "Invalid":
                raise Errors.CalcError()
            # Print the descriptor values
            log.info("The number of coadds = %s" % coadds)
            log.info("The read mode = %s" % read_mode)
            log.info("The total exposure time = %s" % total_exposure_time)
            log.info("The well depth = %s" % well_depth_setting)
            # Loop over each science extension in each input AstroData object
            for ext in ad["SCI"]:
                # Get the size of the raw pixel data
                naxis2 = ext.get_key_value("NAXIS2")
                # Get the raw pixel data
                raw_pixel_data = ext.data
                # Divide the raw pixel data by the number of coadds
                if coadds > 1:
                    raw_pixel_data = raw_pixel_data / coadds
                # Determine the mean of the raw pixel data
                raw_mean_value = np.mean(raw_pixel_data)
                log.info("The mean value of the raw pixel data in " +
                    "%s is %.8f" % (ext.filename, raw_mean_value))
                # Create the key used to access the coefficients that are
                # used to correct for non-linearity
                key = (read_mode, naxis2, well_depth_setting)
                # Get the coefficients from the lookup table
                if lincorlookup[key]:
                    maximum_counts, coeff1, coeff2, coeff3 = \
                        lincorlookup[key]
                else:
                    raise Errors.TableKeyError()
                log.info("Coefficients used = %.12f, %.9e, %.9e" % \
                    (coeff1, coeff2, coeff3))
                # Create a new array that contains the corrected pixel data
                corrected_pixel_data = raw_pixel_data + \
                    coeff2 * raw_pixel_data**2 + coeff3 * raw_pixel_data**3
                # nirlin replaces pixels greater than maximum_counts with 0
                # Set the pixels to 0 if they have a value greater than the
                # maximum counts
                #log.info("Setting pixels to zero if above %f" % \
                #    maximum_counts)
                #corrected_pixel_data[corrected_pixel_data > \
                # maximum_counts] = 0
                # Should probably add the above to the DQ plane
                # Multiply the corrected pixel data by the number of coadds
                if coadds > 1:
                    corrected_pixel_data = corrected_pixel_data * coadds
                # Write the corrected pixel data to the output object
                ext.data = corrected_pixel_data
                # Determine the mean of the corrected pixel data
                corrected_mean_value = np.mean(ext.data)
                log.info("The mean value of the corrected pixel data in " +
                    "%s is %.8f" % (ext.filename, corrected_mean_value))
            # Correct the exposure time by adding coeff1
            total_exposure_time = total_exposure_time + coeff1
            log.info("The corrected total exposure time = %f" % \
                total_exposure_time)
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=keyword)
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

lincorlookup = {
    # In the following form for NIRI data:
    #("read_mode", naxis2, "well_depth_setting"):
    #    (maximum counts, exposure time correction, gamma, eta)
    ("Low Background", 1024, "Shallow"):
        (12000, 1.2662732, 7.3877618e-06, 1.940645271e-10),
    ("Medium Background", 1024, "Shallow"):
        (12000, 0.09442515154, 3.428783846e-06, 4.808353308e-10),
    ("Medium Background", 256, "Shallow"):
        (12000, 0.01029262589, 6.815415667e-06, 2.125210479e-10),
    ("High Background", 1024, "Shallow"):
        (12000, 0.009697324059, 3.040036696e-06, 4.640788333e-10),
    ("High Background", 1024, "Deep"):
        (21000, 0.007680816203, 3.581914163e-06, 1.820403678e-10),
               }
