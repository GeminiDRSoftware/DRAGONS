# This module contains miscellaneous user level functions related to the
# preprocessing of the input dataset

from copy import deepcopy
import numpy as np
from astrodata import Errors
from gempy import geminiTools as gt

def nonlinearity_correct(input=None, output=None, suffix=None):
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
    # Perform checks on the input AstroData instances specified by the 'input'
    # parameter, determine the name of the output AstroData instances using
    # the 'output' and 'suffix' parameters, and instantiate the log using the
    # ScienceFunctionManager
    sfm = gt.ScienceFunctionManager(adInputs=input,
                                    outNames=output,
                                    suffix=suffix,
                                    funcName='nonlinearity_correct')
    input_list, output_names_list, log = sfm.startUp()
    # Define the keyword to be used to time stamp this user level function
    timestampkey = 'LINCORR'
    # Set up a counter to keep track of the output names in the 'output' list
    count = 0
    # Initialise output object list
    output_list = []
    try:
        # Loop over each input object in the input list
        for ad in input_list:
            # Check whether nonlinearity_correct has been run on the data
            # before
            if ad.phuGetKeyValue(timestampkey):
                log.warning('%s has already been corrected for non-linearity' \
                    % (ad.filename))
                adout = ad
            else:
                # Store the original name of the file in the header of the
                # output object
                ad.storeOriginalName()
                # Create the output object by making a 'deep copy' of the input
                # object.
                adout = deepcopy(ad)
                # Set the output file name of the output object
                adout.filename = output_names_list[count]
                count += count
                log.info('Setting the output filename to %s' % adout.filename)
                # Get the appropriate information using the descriptors
                coadds = adout.coadds()
                read_mode = adout.read_mode().asPytype()
                total_exposure_time = adout.exposure_time()
                well_depth_setting = adout.well_depth_setting().asPytype()
                if coadds is None or read_mode is None or \
                    total_exposure_time is None or well_depth_setting is None:
                    # The descriptor functions return None if a value cannot be
                    # found and stores the exception info. Re-raise the
                    # exception.
                    if hasattr(adout, 'exception_info'):
                        raise adout.exception_info
                # Check the raw exposure time (i.e., per coadd). First, convert
                # the total exposure time returned by the descriptor back to
                # the raw exposure time
                exposure_time = total_exposure_time / coadds
                if exposure_time > 600.:
                    log.critical('The raw exposure time is outside the ' + \
                        'range used to derive correction.')
                    raise Errors.InvalidValueError()
                # Check the read mode and well depth setting values
                if read_mode == 'Invalid' or well_depth_setting == 'Invalid':
                    raise Errors.CalcError()
                # Print the descriptor values
                log.info('The number of coadds = %s' % coadds)
                log.info('The read mode = %s' % read_mode)
                log.info('The total exposure time = %s' % total_exposure_time)
                log.info('The well depth = %s' % well_depth_setting)
                # Loop over the extensions in each input object
                for ext in adout:
                    # Get the size of the raw pixel data
                    naxis2 = ext.getKeyValue('NAXIS2')
                    # Get the raw pixel data
                    raw_pixel_data = ext.data
                    # Divide the raw pixel data by the number of coadds
                    if coadds > 1:
                        raw_pixel_data = raw_pixel_data / coadds
                    # Determine the mean of the raw pixel data
                    raw_mean_value = np.mean(raw_pixel_data)
                    log.info('The mean value of the raw pixel data in ' +
                        '%s is %.8f' % (ext.filename, raw_mean_value))
                    # Create the key used to access the coefficients that are
                    # used to correct for non-linearity
                    key = (read_mode, naxis2, well_depth_setting)
                    # Get the coefficients from the lookup table
                    if lincorlookup[key]:
                        maximum_counts, coeff1, coeff2, coeff3 = \
                            lincorlookup[key]
                    else:
                        raise Errors.TableKeyError()
                    log.info('Coefficients used = %.12f, %.9e, %.9e' % \
                        (coeff1, coeff2, coeff3))
                    # Create a new array that contains the corrected pixel data
                    corrected_pixel_data = raw_pixel_data + \
                        coeff2 * raw_pixel_data**2 + coeff3 * raw_pixel_data**3
                    # nirlin replaces pixels greater than maximum_counts with 0
                    # Set the pixels to 0 if they have a value greater than the
                    # maximum counts
                    #log.info('Setting pixels to zero if above %f' % \
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
                    log.info('The mean value of the corrected pixel data in ' +
                        '%s is %.8f' % (ext.filename, corrected_mean_value))
                # Correct the exposure time by adding coeff1
                total_exposure_time = total_exposure_time + coeff1
                log.info('The corrected total exposure time = %f' % \
                    total_exposure_time)
                # Add the appropriate time stamps to the PHU
                sfm.markHistory(adOutputs=adout, historyMarkKey=timestampkey)
            # Append the output object to the output list
            output_list.append(adout)
    except:
        raise

    # Return the output list
    return output_list

lincorlookup = {
    # In the following form for NIRI data:
    #('read_mode', naxis2, 'well_depth_setting'):
    #    (maximum counts, exposure time correction, gamma, eta)
    ('Low Background', 1024, 'Shallow'):
        (12000, 1.2662732, 7.3877618e-06, 1.940645271e-10),
    ('Medium Background', 1024, 'Shallow'):
        (12000, 0.09442515154, 3.428783846e-06, 4.808353308e-10),
    ('Medium Background', 256, 'Shallow'):
        (12000, 0.01029262589, 6.815415667e-06, 2.125210479e-10),
    ('High Background', 1024, 'Shallow'):
        (12000, 0.009697324059, 3.040036696e-06, 4.640788333e-10),
    ('High Background', 1024, 'Deep'):
        (21000, 0.007680816203, 3.581914163e-06, 1.820403678e-10),
               }
