# This module contains user level functions related to validating the input
# dataset

import sys
from copy import deepcopy
from astrodata import Errors
from gempy import managers as mgr

def validate_data_f2(adinput=None, output_names=None, suffix=None,
                     repair=None):
    """
    This user level function is used to validate FLAMINGOS-2 data.
    """
    # Perform checks on the input AstroData objects specified by the "adinput"
    # parameter, determine the name of the output AstroData objects using the
    # "output_names" and "suffix" parameters, and instantiate the log using the
    # ScienceFunctionManager. This needs to be done outside of the try block,
    # since the log object is used in the except block
    sfm = mgr.ScienceFunctionManager(adinput=adinput,
                                     output_names=output_names,
                                     suffix=suffix)
    adinput_list, output_names_list, log = sfm.startUp()
    try:
        # Define the keyword to be used for the time stamp for this user level
        # function
        timestampkey = "VALDATA"
        # Initialize the output_names_list counter and the list of output
        # AstroData objects
        count = 0
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in adinput_list:
            # Check whether the validate_data_f2 user level function has been
            # run previously
            if ad.phuGetKeyValue(timestampkey):
                msg = "%s has already been processed by validate_data_f2" \
                      % (ad.filename)
                log.critical(msg)
                raise Errors.Error(msg)
            # Validate the output object. ACTUALLY DO SOMETHING HERE?
            log.info("No validation required for FLAMINGOS-2")
            #if repair:
            #    # This would be where we would attempt to repair the data 
            #    log.warning("Currently, the 'repair' parameter does not " +
            #                "work. Please come back later.")
            # Set the output file name of the output AstroData object
            ad.filename = output_names_list[count]
            count += 1
            log.info("Setting the output filename to %s" % ad.filename)
            # Add the appropriate time stamps to the PHU
            sfm.markHistory(adOutputs=ad, historyMarkKey=timestampkey)
            sfm.markHistory(adOutputs=ad, historyMarkKey="PREPARE")
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        # Return a single output AstroData object or a list of output AstroData
        # objects
        if len(adoutput_list) == 1:
            return adoutput_list[0]
        else:
            return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def validate_data_gmos(adinput=None, output_names=None, suffix=None,
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
          
    :param adinput: A list containing one or more AstroData instances to be
                  validated
    :type adinput: Astrodata
    
    :param repair: Set to True (the default) to repair the data 
                   Note: this feature does not work yet.
    :type repair: Python boolean
              
    :param output_names: A list containing the names of the output AstroData
                         instances. The number of names in this list should be
                         either one or match the number of AstroData instances
                         given in 'adinput'
    :type output_names: List containing one or more strings.
    
    :param suffix: String to post pend to the names of the input AstroData
                   instances. Used only if output_names is not None.
    :type suffix: string
    """
    # Perform checks on the input AstroData instances specified by the
    # "adinput" parameter, determine the name of the output AstroData
    # instances using the "output_names" and "suffix" parameters, and
    # instantiate the log using the ScienceFunctionManager
    sfm = mgr.ScienceFunctionManager(adinput=adinput,
                                     output_names=output_names,
                                     suffix=suffix)
    adinput_list, output_names_list, log = sfm.startUp()
    try:
        # Define the keyword to be used for the time stamp for this user level
        # function
        timestampkey = "VALDATA"
        # Initialize output_names_list counter and output object list
        count = 0
        output_list = []
        # Loop over each input object in the input list
        for ad in adinput_list:
            # Create the output object by making a "deep copy" of the input
            # object.
            output = deepcopy(ad)
            # Check whether validate_data_f2 has been run on the data
            # before
            if output.phuGetKeyValue(timestampkey):
                log.warning("%s has already been validated" \
                            % (output.filename))
            else:
                # Validate the output object. Ensure that the input have 1, 3,
                # 6 or 12 extensions
                log.info("Raw GMOS data should only have 1, 3, 6 or 12 " +
                         "extensions ... checking")
                extensions = output.countExts("SCI")
                if (extensions != 1 and extensions != 3 and extensions != 6 and
                    extensions != 12):
                    if repair:
                        # This would be where we would attempt to repair the
                        # data 
                        log.warning("Currently, the 'repair' parameter does " +
                                    "not work. Please come back later.")
                    raise Errors.Error("The number of extensions in %s do " +
                                       "match with the number of extensions " +
                                       "expected in raw GMOS data." \
                                       % output.filename)
            # Set the output file name of the output object
            output.filename = output_names_list[count]
            count += 1
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
