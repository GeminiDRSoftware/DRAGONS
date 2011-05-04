# This module contains user level functions related to validating the input
# dataset

import sys
from copy import deepcopy
from astrodata import Errors
from gempy import managers as mgr

def validate_data_f2(input=None, output_names=None, suffix=None, repair=None):
    """
    This user level function is used to validate FLAMINGOS-2 data.
    """
    try:
        # Perform checks on the input AstroData instances specified by the
        # "input" parameter, determine the name of the output AstroData
        # instances using the "output_names" and "suffix" parameters, and
        # instantiate the log using the ScienceFunctionManager
        sfm = mgr.ScienceFunctionManager(input=input,
                                         output_names=output_names,
                                         suffix=suffix)
        input_list, output_names_list, log = sfm.startUp()
        # Define the keyword to be used for the time stamp for this user level
        # function
        timestampkey = "VALDATA"
        # Initialize output_names counter and output object list
        count = 0
        output_list = []
        # Loop over each input object in the input list
        for ad in input_list:
            # Create the output object by making a "deep copy" of the input
            # object.
            output = deepcopy(ad)
            # Check whether validate_data_f2 has been run on the data
            # before
            if ad.phuGetKeyValue(timestampkey):
                log.warning("%s has already been validated" % (ad.filename))
                break
            # Validate the output object. NEED TO ACTUALLY DO SOMETHING HERE?
            log.status("No validation required for FLAMINGOS-2")
            if repair:
                # This would be where we would attempt to repair the data 
                log.warning("Currently, the 'repair' parameter does " +
                            "not work. Please come back later.")
            # Set the output file name of the output object
            output.filename = output_names_list[count]
            count += count
            log.info("Setting the output filename to %s" % output.filename)
            # Append the output object to the output list
            output_list.append(output)
            # Add the appropriate time stamps to the PHU
            sfm.markHistory(adOutputs=output, historyMarkKey=timestampkey)
            sfm.markHistory(adOutputs=output, historyMarkKey="PREPARE")
        # Return the output list
        return output_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def validate_data_gmos(input=None, output_names=None, suffix=None,
                       repair=False):
    """
    This user level function is used to validate GMOS data. It will ensure the
    data is not corrupted or in an odd format that will affect later steps in
    the reduction process. It currently just checks if there are 1, 3, 6 or 12
    SCI extensions in the input. If there are issues with the data, the flag
    'repair' can be used to turn on the feature to repair it or not.
    
    This user level function is called by validateData.
    
    Either a 'main' type logger object, if it exists, or a null logger (i.e.,
    no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
          
    :param input: A list containing one or more AstroData instances to be
                  validated
    :type input: Astrodata
    
    :param repair: Set to True (the default) to repair the data 
                   Note: this feature does not work yet.
    :type repair: Python boolean
              
    :param output_names: A list containing the names of the output AstroData
                         instances. The number of names in this list should be
                         either one or match the number of AstroData instances
                         given in 'input'
    :type output_names: List containing one or more strings.
    
    :param suffix: String to post pend to the names of the input AstroData
                   instances. Used only if output_names is not None.
    :type suffix: string
    """
    try:
        # Perform checks on the input AstroData instances specified by the
        # "input" parameter, determine the name of the output AstroData
        # instances using the "output_names" and "suffix" parameters, and
        # instantiate the log using the ScienceFunctionManager
        sfm = mgr.ScienceFunctionManager(input=input,
                                         output_names=output_names,
                                         suffix=suffix)
        input_list, output_names_list, log = sfm.startUp()
        # Define the keyword to be used for the time stamp for this user level
        # function
        timestampkey = "VALDATA"
        # Initialize output_names counter and output object list
        count = 0
        output_list = []
        # Loop over each input object in the input list
        for ad in input_list:
            # Create the output object by making a "deep copy" of the input
            # object.
            output = deepcopy(ad)
            # Check whether validate_data_f2 has been run on the data
            # before
            if ad.phuGetKeyValue(timestampkey):
                log.warning("%s has already been validated" % (ad.filename))
                break
            # Validate the output object. Ensure that the input have 1, 3, 6
            # or 12 extensions
            extensions = output.countExts("SCI")
            if (extensions != 1 and extensions != 3 and extensions != 6 and
                extensions != 12):
                if repair:
                    # This would be where we would attempt to repair the data 
                    log.warning("Currently, the 'repair' parameter does not " +
                                "work. Please come back later.")
                raise Errors.Error("The number of extensions in %s do match " +
                                   "with the number of extensions expected " +
                                   "in raw GMOS data." % output.filename)
            # Set the output file name of the output object
            output.filename = output_names_list[count]
            count += count
            log.info("Setting the output filename to %s" % output.filename)
            # Append the output object to the output list
            output_list.append(output)
            # Add the appropriate time stamps to the PHU
            sfm.markHistory(adOutputs=output, historyMarkKey=timestampkey)
            sfm.markHistory(adOutputs=output, historyMarkKey="PREPARE")
        # Return the output list
        return output_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
