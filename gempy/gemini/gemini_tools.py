#
#                                                                  gemini_python
#
#                                                                   gempy.gemini
#                                                                gemini_tools.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
import os
import re
import sys
import numbers
import warnings
import numpy as np

from copy import deepcopy
from datetime import datetime
from importlib import import_module

from functools import wraps

from astropy import stats
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.table import vstack, Table, Column

from scipy.special import erf

from ..library import astrotools as at
from ..utils import logutils

import astrodata
from astrodata import __version__ as ad_version

@models.custom_model
def CumGauss1D(x, mean=0.0, stddev=1.0):
    return 0.5*(1.0+erf((x-mean)/(1.414213562*stddev)))

# ------------------------------------------------------------------------------
# Allows all functions to treat input as a list and return a list without the
# specific need to check.
def handle_single_adinput(fn):
    @wraps(fn)
    def wrapper(adinput, *args, **kwargs):
        if not isinstance(adinput, list):
            ret = fn([adinput], *args, **kwargs)
            return ret[0] if isinstance(ret, list) else ret
        else:
            return fn(adinput, *args, **kwargs)
    return wrapper
# ------------------------------------------------------------------------------
@handle_single_adinput
def add_objcat(adinput=None, extver=1, replace=False, table=None, sx_dict=None):
    """
    Add OBJCAT table if it does not exist, update or replace it if it does.

    Parameters
    ----------
    adinput: AstroData
        AD object(s) to add table to

    extver: int
        Extension number for the table (should match the science extension)

    replace: bool
        replace (overwrite) with new OBJCAT? If False, the new table must
        have the same length as the existing OBJCAT

    table: Table
        new OBJCAT Table or new columns. For a new table, X_IMAGE, Y_IMAGE,
        X_WORLD, and Y_WORLD are required columns
    """
    log = logutils.get_logger(__name__)

    # ensure caller passes the sextractor default dictionary of parameters.
    try:
        assert isinstance(sx_dict, dict) and sx_dict.has_key(('dq', 'param'))
    except AssertionError:
        log.error("TypeError: A SExtractor dictionary was not received.")
        raise TypeError("Require SExtractor parameter dictionary.")
    
    # Initialize the list of output AstroData objects
    adoutput = []
    # Parse sextractor parameters for the list of expected columns
    expected_columns = parse_sextractor_param(sx_dict['dq', 'param'])
    # Append a few more that don't come from directly from sextractor
    expected_columns.extend(["REF_NUMBER","REF_MAG","REF_MAG_ERR",
                             "PROFILE_FWHM","PROFILE_EE50"])

    # Loop over each input AstroData object in the input list
    for ad in adinput:
        ext = ad.extver(extver)
        # Check if OBJCAT already exists and just update if desired
        objcat = getattr(ext, 'OBJCAT', None)
        if objcat and not replace:
            log.fullinfo("Table already exists; updating values.")
            for name in table.columns:
                objcat[name].data[:] = table[name].data
        else:
            # Append columns in order of definition in SExtractor params
            new_objcat = Table()
            nrows = len(table)
            for name in expected_columns:
                # Define Column properties with sensible placeholder data
                if name in ["NUMBER"]:
                    default = range(1, nrows+1)
                    dtype = np.int32
                elif name in ["FLAGS", "IMAFLAGS_ISO", "REF_NUMBER"]:
                    default = [-999] * nrows
                    dtype = np.int32
                else:
                    default = [-999] * nrows
                    dtype = np.float32
                # Use input table column if given, otherwise the placeholder
                new_objcat.add_column(table[name] if name in table.columns else
                                Column(data=default, name=name, dtype=dtype))

            # Replace old version or append new table to AD object
            if objcat:
                log.fullinfo("Replacing existing OBJCAT in {}".
                             format(ad.filename))
            ext.OBJCAT = new_objcat
        adoutput.append(ad)

    return adoutput

@handle_single_adinput
def array_information(adinput=None):
    """
    Returns information about the relationship between amps (extensions)
    and physical CCDs. Used only for GMOS imaging to remove fringes and,
    even then, half the stuff it returns isn't used!

    Parameters
    ----------
    adinput: list/AD
        single or list of AD objects

    Returns
    -------
    list of dicts/dicts
    Each dict contains:
        amps_per_array: dict of {array number (1-indexed) : number of amps}
        amps_order: list of ints giving the AD slices in order of increasing x
        array_number: list of ints giving the number of the array each slice
                      is on (1-indexed)
        reference_extension: extver of the extension in the middle
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)
    
    # Initialize the list of dictionaries of output array numbers
    # Keys will be (extname,extver)
    array_info_list = []

    # Loop over each input AstroData object in the input list
    for ad in adinput:
        arrayinfo = {}

        # Get the correct order of the extensions by sorting on first element
        # in detector section (raw ordering is whichever amps read out first)
        detx1 = [sec.x1 for sec in ad.detector_section()]
        ampsorder  = list(np.argsort(detx1))

        # Get array sections for determining when a new array is found
        arrx1 = [sec.x1 for sec in ad.array_section()]

        # Initialize these so that first extension will always start a new array
        last_detx1   = detx1[ampsorder[0]] - 1
        last_arrx1 = arrx1[ampsorder[0]]

        array_number = np.empty_like(ampsorder)
        this_array_number = 0
        amps_per_array = {}

        for i in ampsorder:
            this_detx1 = detx1[i]
            this_arrx1 = arrx1[i]

            if (this_detx1 > last_detx1 and this_arrx1 <= last_arrx1):
                # New array found
                this_array_number += 1
                amps_per_array[this_array_number] = 1
            else:
                amps_per_array[this_array_number] += 1
            array_number[i] = this_array_number
            last_detx1 = this_detx1
            last_arrx1 = this_arrx1

        # Reference extension if tiling/mosaicing all data together
        refext = ad[ampsorder[int(0.5 * (len(ampsorder)-1))]].hdr['EXTVER']

        arrayinfo['array_number'] = list(array_number)
        arrayinfo['amps_order'] = ampsorder
        arrayinfo['amps_per_array'] = amps_per_array
        arrayinfo['reference_extension'] = refext

        # Append the output AstroData object to the list of output
        # AstroData objects
        array_info_list.append(arrayinfo)

    return array_info_list

#TODO: CJS believes this is never called in normal operations and doesn't even
# do what it's supposed to do. So we'll comment it and see what happens
#
#def calc_nbiascontam(adInputs=None, biassec=None):
#    """
#    This function will find the largest difference between the horizontal
#    component of every BIASSEC value and those of the biassec parameter.
#    The returned value will be that difference as an integer and it will be
#    used as the value for the nbiascontam parameter used in the gireduce
#    call of the overscanSubtract primitive.
#
#    Parameters
#    ----------
#    adInputs: list
#        list of AD instances
#    biassec: str
#        biassec parameter of format '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
#
#    Returns
#    -------
#    int
#        number of columns to ignore in overscan section
#    """
#    log = logutils.get_logger(__name__)
#
#    try:
#        # Prepare a stored value to be compared between the inputs
#        retvalue=0
#        # Loop through the inputs
#        for ad in adInputs:
#            # Split up the input triple list into three separate sections
#            biassecStrList = biassec.split('],[')
#            # Prepare the to-be list of lists
#            biassecIntList = []
#            for biassecStr in biassecStrList:
#                # Use sectionStrToIntList function to convert
#                # each string version of the list into actual integer tuple
#                # and load it into the lists of lists
#                # of form [y1, y2, x1, x2] 0-based and non-inclusive
#                biassecIntList.append(gmu.sectionStrToIntList(biassecStr))
#
#            # Setting the return value to be updated in the loop below
#            retvalue=0
#            for ext in ad['SCI']:
#                # Retrieving current BIASSEC value                    #  THIS WHERE THE
#                BIASSEC = ext.get_key_value('BIASSEC')                #  bias_section()
#                # Converting the retrieved string into a integer list #  descriptor
#                # of form [y1, y2, x1, x2] 0-based and non-inclusive  #  would be used!!!!
#                BIASSEClist = sectionStrToIntList(BIASSEC)     #
#                # Setting the lower case biassec list to the appropriate
#                # list in the lists of lists created above the loop
#                biasseclist = biassecIntList[ext.extver() - 1]
#                # Ensuring both biassec's have the same vertical coords
#                if (biasseclist[0] == BIASSEClist[0]) and \
#                (biasseclist[1] == BIASSEClist[1]):
#                    # If overscan/bias section is on the left side of chip
#                    if biasseclist[3] < 50:
#                        # Ensuring right X coord of both biassec's are equal
#                        if biasseclist[2] == BIASSEClist[2]:
#                            # Set the number of contaminating columns to the
#                            # difference between the biassec's left X coords
#                            nbiascontam = BIASSEClist[3] - biasseclist[3]
#                        # If left X coords of biassec's don't match, set
#                        # number of contaminating columns to 4 and make a
#                        # error log message
#                        else:
#                            log.error('right horizontal components of '+
#                                      'biassec and BIASSEC did not match, '+
#                                      'so using default nbiascontam=4')
#                            nbiascontam = 4
#                    # If overscan/bias section is on the right side of chip
#                    else:
#                        # Ensuring left X coord of both biassec's are equal
#                        if biasseclist[3] == BIASSEClist[3]:
#                            # Set the number of contaminating columns to the
#                            # difference between the biassec's right X coords
#                            nbiascontam = BIASSEClist[2] - biasseclist[2]
#                        else:
#                            log.error('left horizontal components of '+
#                                      'biassec and BIASSEC did not match, '+
#                                      'so using default nbiascontam=4')
#                            nbiascontam = 4
#                # Overscan/bias section is not on left or right side of chip
#                # , so set to number of contaminated columns to 4 and log
#                # error message
#                else:
#                    log.error('vertical components of biassec and BIASSEC '+
#                              'parameters did not match, so using default '+
#                              'nbiascontam=4')
#                    nbiascontam = 4
#                # Find the largest nbiascontam value throughout all chips
#                # and set it as the value to be returned
#                if nbiascontam > retvalue:
#                    retvalue = nbiascontam
#        return retvalue
#    # If all the above checks and attempts to calculate a new nbiascontam
#    # fail, make a error log message and return the value 4. so exiting
#    # 'gracefully'.
#    except:
#        log.error('An error occurred while trying to calculate the '+
#                  'nbiascontam, so using default value = 4')
#        return 4

def check_inputs_match(adinput1=None, adinput2=None, check_filter=True):
    """
    This function will check if the inputs match.  It will check the filter,
    binning and shape/size of the every SCI frames in the inputs.
    
    There must be a matching number of inputs for 1 and 2.

    Parameters
    ----------
    adinput1: list/AD
    adinput2: list/AD
        single AstroData instances or length-matched lists

    check_filter: bool
        if True, also check the filter name of each pair
    """
    log = logutils.get_logger(__name__)

    # Turn inputs into lists for ease of manipulaiton later
    if not isinstance(adinput1, list):
        adinput1 = [adinput1]
    if not isinstance(adinput2, list):
        adinput2 = [adinput2]
    if len(adinput1) != len(adinput2):
        log.error('Inputs do not match in length')
        raise ValueError('Inputs do not match in length')

    for ad1, ad2 in zip(adinput1, adinput2):
        log.fullinfo('Checking inputs {} and {}'.format(ad1.filename,
                                                        ad2.filename))
        if len(ad1) != len(ad2):
            log.error('Inputs have different numbers of SCI extensions.')
            raise ValueError('Mismatching number of SCI extensions in inputs')

        # Now check each extension
        for ext1, ext2 in zip(ad1, ad2):
            log.fullinfo('Checking EXTVER {}'.format(ext1.hdr['EXTVER']))
            
            # Check shape/size
            if ext1.data.shape != ext2.data.shape:
                log.error('Extensions have different shapes')
                raise ValueError('Extensions have different shape')
            
            # Check binning
            if (ext1.detector_x_bin() != ext2.detector_x_bin() or
                        ext1.detector_y_bin() != ext2.detector_y_bin()):
                log.error('Extensions have different binning')
                raise ValueError('Extensions have different binning')

        # Check filter if desired
        if check_filter and (ad1.filter_name() != ad2.filter_name()):
            log.error('Extensions have different filters')
            raise ValueError('Extensions have different filters')

        log.fullinfo('Inputs match')
    return


def matching_inst_config(ad1=None, ad2=None, check_exposure=False):
    """
    Compare two AstroData instances and report whether their instrument
    configurations are identical, including the exposure time (but excluding
    the telescope pointing). This function was added for NIR images but
    should probably be merged with check_inputs_match() above, after checking
    carefully that the changes needed won't break any GMOS processing.

    Parameters
    ----------
    ad1: AD
    ad2: AD
        single AstroData instances or length-matched lists

    check_exposure: bool
        if True, also check the exposure time of each pair

    Returns
    -------
    bool
        Do they match?
    """
    log = logutils.get_logger(__name__)
    log.debug('Comparing instrument config for AstroData instances')

    result = True
    if len(ad1) != len(ad2):
        result = False
        log.debug('  Number of SCI extensions differ')

    for ext1, ext2 in zip(ad1, ad2):
        if ext1.data.shape != ext2.data.shape:
            result = False
            log.debug('  Array dimensions differ')
            break

    # Check all these descriptors for equality
    things_to_check = ['data_section', 'detector_roi_setting', 'read_mode',
                       'well_depth_setting', 'gain_setting', 'detector_x_bin',
                       'detector_y_bin', 'coadds', 'camera', 'filter_name',
                       'focal_plane_mask', 'lyot_stop', 'decker',
                       'pupil_mask', 'disperser']
    for descriptor in things_to_check:
        if getattr(ad1, descriptor)() != getattr(ad2, descriptor)():
            result = False
            log.debug('  Descriptor failure for {}'.descriptor)

    # This is the tolerance Paul used for NIRI & F2 in FITS store:
    try:
        same = abs(ad1.central_wavelength() - ad2.central_wavelength()) < 0.001
    except TypeError:
        same = ad1.central_wavelength() ==  ad2.central_wavelength()
    if not same:
        result = False
        log.debug('  Central wavelengths differ')

    if check_exposure:
        try:
            # This is the tolerance Paul used for NIRI in FITS store:
            if abs(ad1.exposure_time() - ad2.exposure_time()) > 0.01:
                result = False
                log.debug('  Exposure times differ')
        except TypeError:
            log.error('Non-numeric type from exposure_time() descriptor')

    if result:
        log.debug('  Configurations match')
    
    return result

@handle_single_adinput
def clip_auxiliary_data(adinput=None, aux=None, aux_type=None, 
                        return_dtype=None, keyword_comments=None):
    """
    This function clips auxiliary data like calibration files or BPMs
    to the size of the data section in the science. It will pad auxiliary
    data if required to match un-overscan-trimmed data, but otherwise
    requires that the auxiliary data contain the science data.

    Parameters
    ----------
    adinput: list/AstroData
        input science image(s)
    aux: list/AstroData
        auxiliary file(s) (e.g., BPM, flat) to be clipped
    aux_type: str
        type of auxiliary file
    return_dtype: dtype
        datatype of returned objects
    keyword_comments: dict
        comments to add for any new header keywords

    Returns
    -------
    list/AD:
        auxiliary file(s), appropriately clipped
    """
    log = logutils.get_logger(__name__)

    # ensure caller passes the sextractor default dictionary of parameters.
    try:
        assert isinstance(keyword_comments, dict)
    except AssertionError:
        log.error("TypeError: keyword comments dict was not received.")
        raise TypeError("keyword comments dict required")

    if not isinstance(aux, list):
        aux = [aux]
    
    # Initialize the list of output AstroData objects
    aux_output_list = []

    # Loop over each input AstroData object in the input list
    for ad, this_aux in zip(adinput, aux):
        # Make a new auxiliary file for appending to, starting with PHU
        new_aux = astrodata.create(this_aux.header[0])

        # Get the detector section, data section, array section and the
        # binning of the x-axis and y-axis values for the science AstroData
        # object using the appropriate descriptors
        sci_detsec = ad.detector_section()
        sci_datasec = ad.data_section()
        sci_arraysec = ad.array_section()
        sci_xbin = ad.detector_x_bin()
        sci_ybin = ad.detector_y_bin()

        datasec_keyword = ad._keyword_for('data_section')
        detsec_keyword = ad._keyword_for('detector_section')
        arraysec_keyword = ad._keyword_for('array_section')

        # Get the detector section, data section and array section values
        # for the auxiliary AstroData object using the appropriate
        # descriptors
        aux_detsec   = this_aux.detector_section()
        aux_datasec  = this_aux.data_section()
        aux_arraysec = this_aux.array_section()

        for ext, detsec, datasec, arraysec in zip(ad, sci_detsec,
                                            sci_datasec, sci_arraysec):

            # Array section is unbinned; to use as indices for
            # extracting data, need to divide by the binning
            arraysec = [
              arraysec[0] / sci_xbin, arraysec[1] / sci_xbin,
              arraysec[2] / sci_ybin, arraysec[3] / sci_ybin]

            # Check whether science data has been overscan-trimmed
            science_shape = ext.data.shape[-2:]
            # Offsets give overscan regions on either side of data:
            # [left offset, right offset, bottom offset, top offset]
            science_offsets = [datasec[0], science_shape[1] - datasec[1],
                               datasec[2], science_shape[0] - datasec[3]]
            science_trimmed = all([off==0 for off in science_offsets])

            found = False
            for auxext, adetsec, adatasec, aarraysec in zip(this_aux,
                                aux_detsec, aux_datasec, aux_arraysec):

                # Array section is unbinned; to use as indices for
                # extracting data, need to divide by the binning
                aarraysec = [
                    aarraysec[0] / sci_xbin, aarraysec[1] / sci_xbin,
                    aarraysec[2] / sci_ybin, aarraysec[3] / sci_ybin]

                # Check whether auxiliary detector section contains
                # science detector section
                if (adetsec[0] <= detsec[0] and # x lower
                    adetsec[1] >= detsec[1] and # x upper
                    adetsec[2] <= detsec[2] and # y lower
                    adetsec[3] >= detsec[3]):   # y upper
                    # Auxiliary data contains or is equal to science data
                    found = True
                else:
                    continue

                # Check whether auxiliary data has been overscan-trimmed
                aux_shape = auxext.data.shape
                aux_offsets = [adatasec[0], aux_shape[1] - adatasec[1],
                               adatasec[2], aux_shape[0] - adatasec[3]]
                aux_trimmed = all([off==0 for off in aux_offsets])

                # Define data extraction region corresponding to science
                # data section (not including overscan)
                x_translation = (arraysec[0] - datasec[0] -
                                 aarraysec[0] + adatasec[0])
                y_translation = (arraysec[2] - datasec[2]
                                 - aarraysec[2] + adatasec[2])
                region = [datasec[2] + y_translation, datasec[3] + y_translation,
                          datasec[0] + x_translation, datasec[1] + x_translation]

                # Deepcopy auxiliary SCI/VAR/DQ planes, allowing the same
                # aux extension to be used for a difference sci extension
                ext_to_clip = deepcopy(auxext)

                # Pull out specified data region:
                if science_trimmed or aux_trimmed:
                    clipped = ext_to_clip[0].nddata[region[0]:region[1],
                                                region[2]:region[3]]

                    # Where no overscan is needed, just use the data region:
                    ext_to_clip[0].reset(clipped)

                    # Pad trimmed aux arrays with zeros to match untrimmed
                    # science data:
                    if aux_trimmed and not science_trimmed:
                        # Science decision: trimmed calibrations can't be
                        # meaningfully matched to untrimmed science data
                        if aux_type != 'bpm':
                            raise IOError(
                                "Auxiliary data {} is trimmed, but "
                                "science data {} is untrimmed.".
                                format(auxext.filename, ext.filename))

                        # Use duplicate iterators over the reversed science_offsets
                        # list to unpack its values in pairs and reverse them:
                        padding = tuple((bef, aft) for aft, bef in \
                                        zip(*[reversed(science_offsets)]*2))

                        # Replace the array with one that's padded with the
                        # appropriate number of zeros at each edge:
                        ext_to_clip.operate(np.pad, padding, 'constant',
                                          constant_values=0)

                # If nothing is trimmed, just use the unmodified data
                # after checking that the regions match (a condition
                # preserved from r5564 without revisiting its logic):
                elif not all(off1 == off2 for off1, off2 in
                             zip(aux_offsets, science_offsets)):
                    raise ValueError(
                        "Overscan regions do not match in {}, {}".
                        format(auxext.filename, ext.filename))

                # Convert the dtype if requested
                if return_dtype is not None:
                    ext_to_clip.operate(np.ndarray.astype, return_dtype)

                # Append the data to the AD object
                new_aux.append(ext_to_clip[0].nddata, reset_ver=True)

            if not found:
                raise IOError(
                  "No auxiliary data in {} matches the detector section "
                  "{} in {}[SCI,{}]".format(this_aux.filename, detsec,
                                       ad.filename, ext.hdr['EXTVER']))

        log.stdinfo("Clipping {} to match science data.".
                    format(os.path.basename(this_aux.filename)))
        aux_output_list.append(new_aux)

    return aux_output_list

@handle_single_adinput
def clip_auxiliary_data_GSAOI(adinput=None, aux=None, aux_type=None, 
                        return_dtype=None, keyword_comments=None):
    """
    This function clips auxiliary data like calibration files or BPMs
    to the size of the data section in the science. It will pad auxiliary
    data if required to match un-overscan-trimmed data, but otherwise
    requires that the auxiliary data contain the science data.

    This is a GSAOI-specific version that uses the FRAMEID keyword,
    rather than DETSEC to match up extensions between the science and
    auxiliary data.

    Parameters
    ----------
    adinput: list/AstroData
        input science image(s)
    aux: list/AstroData
        auxiliary file(s) (e.g., BPM, flat) to be clipped
    aux_type: str
        type of auxiliary file
    return_dtype: dtype
        datatype of returned objects
    keyword_comments: dict
        comments to add for any new header keywords

    Returns
    -------
    list/AD:
        auxiliary file(s), appropriately clipped
    """
    log = logutils.get_logger(__name__)

    # ensure caller passes the sextractor default dictionary of parameters.
    try:
        assert isinstance(keyword_comments, dict)
    except AssertionError:
        log.error("TypeError: keyword comments dict was not received.")
        raise TypeError("keyword comments dict required")

    if not isinstance(aux, list):
        aux = [aux]

    # Initialize the list of output AstroData objects
    aux_output_list = []

    # Loop over each input AstroData object in the input list
    for ad, this_aux in zip(adinput, aux):
        # Make a new auxiliary file for appending to, starting with PHU
        new_aux = astrodata.create(this_aux.header[0])

        # Get the detector section, data section, array section and the
        # binning of the x-axis and y-axis values for the science AstroData
        # object using the appropriate descriptors
        sci_detsec = ad.detector_section()
        sci_datasec = ad.data_section()
        sci_arraysec = ad.array_section()

        datasec_keyword = ad._keyword_for('data_section')
        detsec_keyword = ad._keyword_for('detector_section')
        arraysec_keyword = ad._keyword_for('array_section')

        # Get the detector section, data section and array section values
        # for the auxiliary AstroData object using the appropriate
        # descriptors
        aux_datasec  = this_aux.data_section()
        aux_arraysec = this_aux.array_section()

        for ext, detsec, datasec, arraysec in zip(ad, sci_detsec,
                                            sci_datasec, sci_arraysec):

            frameid = ext.hdr['FRAMEID']

            # Check whether science data has been overscan-trimmed
            science_shape = ext.data.shape[-2:]
            # Offsets give overscan regions on either side of data:
            # [left offset, right offset, bottom offset, top offset]
            science_offsets = [datasec[0], science_shape[1] - datasec[1],
                               datasec[2], science_shape[0] - datasec[3]]
            science_trimmed = all([off==0 for off in science_offsets])


            found = False
            for auxext, adatasec, aarraysec in zip(this_aux, aux_datasec,
                                                   aux_arraysec):
                # Retrieve the extension number for this extension
                aux_frameid = auxext.hdr['FRAMEID']
                aux_shape = auxext.data.shape

                if (aux_frameid == frameid and
                    aux_shape[0] >= science_shape[0] and
                    aux_shape[1] >= science_shape[1]):

                    # Auxiliary data is big enough as has right FRAMEID
                    found = True
                else:
                    continue

                # Check whether auxiliary data has been overscan-trimmed
                aux_shape = auxext.data.shape
                aux_offsets = [adatasec[0], aux_shape[1] - adatasec[1],
                               adatasec[2], aux_shape[0] - adatasec[3]]
                aux_trimmed = all([off==0 for off in aux_offsets])


                # Define data extraction region corresponding to science
                # data section (not including overscan)
                x_translation = (arraysec[0] - datasec[0] -
                                 aarraysec[0] + adatasec[0])
                y_translation = (arraysec[2] - datasec[2]
                                 - aarraysec[2] + adatasec[2])
                region = [datasec[2] + y_translation, datasec[3] + y_translation,
                          datasec[0] + x_translation, datasec[1] + x_translation]

                # Deepcopy auxiliary SCI/VAR/DQ planes, allowing the same
                # aux extension to be used for a difference sci extension
                ext_to_clip = deepcopy(auxext)

                # Pull out specified data region:
                if science_trimmed or aux_trimmed:
                    clipped = ext_to_clip[0].nddata[region[0]:region[1],
                              region[2]:region[3]]

                    # Where no overscan is needed, just use the data region:
                    ext_to_clip[0].reset(clipped)

                    # Pad trimmed aux arrays with zeros to match untrimmed
                    # science data:
                    if aux_trimmed and not science_trimmed:
                        # Science decision: trimmed calibrations can't be
                        # meaningfully matched to untrimmed science data
                        if aux_type != 'bpm':
                            raise IOError(
                                "Auxiliary data {} is trimmed, but "
                                "science data {} is untrimmed.".
                                    format(auxext.filename, ext.filename))

                        # Use duplicate iterators over the reversed science_offsets
                        # list to unpack its values in pairs and reverse them:
                        padding = tuple((bef, aft) for aft, bef in \
                                        zip(*[reversed(science_offsets)] * 2))

                        # Replace the array with one that's padded with the
                        # appropriate number of zeros at each edge:
                        ext_to_clip.operate(np.pad, padding, 'constant',
                                                   constant_values=0)

                # If nothing is trimmed, just use the unmodified data
                # after checking that the regions match (a condition
                # preserved from r5564 without revisiting its logic):
                elif not all(off1 == off2 for off1, off2 in
                             zip(aux_offsets, science_offsets)):
                    raise ValueError(
                        "Overscan regions do not match in {}, {}".
                            format(auxext.filename, ext.filename))

                # Convert the dtype if requested
                if return_dtype is not None:
                    ext_to_clip.operate(np.ndarray.astype, return_dtype)

                # Append the data to the AD object
                new_aux.append(ext_to_clip[0].nddata, reset_ver=True)

            if not found:
                raise IOError(
                    "No auxiliary data in {} matches the detector section "
                    "{} in {}[SCI,{}]".format(this_aux.filename, detsec,
                                              ad.filename, ext.EXTVER))

        log.stdinfo("Clipping {} to match science data.".
                    format(os.path.basename(this_aux.filename)))
        aux_output_list.append(new_aux)

    return aux_output_list

def clip_sources(ad):
    """
    This function takes the source data from the OBJCAT and returns the best
    (stellar) sources for IQ measurement.

    Parameters
    ----------
    ad: AstroData
        image with attached OBJCAT
    
    Returns
    -------
    list of Tables
        each Table contains a subset of information on the good stellar sources
    """
    # Columns in the output table
    column_mapping = {'x': 'X_IMAGE',
                      'y': 'Y_IMAGE',
                      'fwhm': 'PROFILE_FWHM',
                      'fwhm_arcsec': 'PROFILE_FWHM',
                      'isofwhm': 'FWHM_IMAGE',
                      'isofwhm_arcsec': 'FWHM_WORLD',
                      'ee50d': 'PROFILE_EE50',
                      'ee50d_arcsec': 'PROFILE_EE50',
                      'ellipticity': 'ELLIPTICITY',
                      'pa': 'THETA_WORLD',
                      'flux': 'FLUX_AUTO',
                      'flux_max': 'FLUX_MAX',
                      'background': 'BACKGROUND',
                      'flux_radius': 'FLUX_RADIUS'}

    is_ao = ad.is_ao()
    sn_limit = 25 if is_ao else 50

    good_sources = []
    for ext in ad:
        try:
            objcat = ext.OBJCAT
        except AttributeError:
            good_sources.append(Table())
            continue

        stellar = np.fabs(objcat['FWHM_IMAGE']/1.08 - objcat['PROFILE_FWHM']
                          ) < 0.2*objcat['FWHM_IMAGE'] if is_ao else \
                    objcat['CLASS_STAR'] > 0.8

        good = np.logical_and.reduce((
            objcat['PROFILE_FWHM'] > 0,
            objcat['ELLIPTICITY'] > 0,
            objcat['ELLIPTICITY'] < 0.5,
            objcat['B_IMAGE'] > 1.1,
            objcat['FLUX_AUTO'] > sn_limit * objcat['FLUXERR_AUTO'],
            stellar))

        # Source is good if more than 20 connected pixels
        # Ignore criterion if all undefined (-999)
        if not np.all(objcat['ISOAREA_IMAGE'] == -999):
            good &= objcat['ISOAREA_IMAGE'] > 20

        if not np.all(objcat['FLUXERR_AUTO'] == -999):
            good &= objcat['FLUX_AUTO'] > sn_limit * objcat['FLUXERR_AUTO']

        # For each source, we allow zero saturated pixels but up to 2% of
        # other "bad" types
        max_bad_pix = np.where(objcat['IMAFLAGS_ISO'] & 4, 0,
                               0.02*objcat['ISOAREA_IMAGE'])
        if not np.all(objcat['NIMAFLAGS_ISO'] == -999):
            good &= objcat['NIMAFLAGS_ISO'] <= max_bad_pix

        good &= objcat['PROFILE_EE50'] > objcat['PROFILE_FWHM']

        # Create new tables with the columns and rows we want
        table = Table()
        for new_name, old_name in column_mapping.iteritems():
            table[new_name] = objcat[old_name][good]
        pixscale = ext.pixel_scale()
        table['fwhm_arcsec'] *= pixscale
        table['ee50d_arcsec'] *= pixscale

        # Clip outliers in FWHM - single 2-sigma clip if more than 3 sources.
        if len(table) >= 3:
            table = table[~stats.sigma_clip(table['fwhm_arcsec'], sigma=2, iters=1).mask]

        good_sources.append(table)

    return good_sources

@handle_single_adinput
def convert_to_cal_header(adinput=None, caltype=None, keyword_comments=None):
    """
    This function replaces position, object, and program information 
    in the headers of processed calibration files that are generated
    from science frames, eg. fringe frames, maybe sky frames too.
    It is called, for example, from the storeProcessedFringe primitive.

    Parameters
    ----------
    adinput: list/AstroData
        input image(s)

    caltype: str
        type of calibration ('fringe', 'sky', 'flat' allowed)

    keyword_comments: dict
        comments to add to the header of the output

    Returns
    -------
    list/AD
        modified version of input
    """
    log = logutils.get_logger(__name__)

    try:
        assert isinstance(keyword_comments, dict)
    except AssertionError:
        log.error("TypeError: keyword comments dict was not received.")
        raise TypeError("keyword comments dict required")
    
    if caltype is None:
        raise ValueError("Caltype should not be None")

    fitsfilenamecre = re.compile("^([NS])(20\d\d)([01]\d[0123]\d)(S)"
                                 "(?P<fileno>\d\d\d\d)(.*)$")

    for ad in adinput:
        log.fullinfo("Setting OBSCLASS, OBSTYPE, GEMPRGID, OBSID, "
                     "DATALAB, RELEASE, OBJECT, RA, DEC, CRVAL1, "
                     "and CRVAL2 to generic defaults")

        # Do some date manipulation to get release date and
        # fake program number

        # Get date from day data was taken if possible
        date_taken = ad.ut_date()
        if date_taken is None:
            # Otherwise use current time
            date_taken = datetime.today().date()
        release = date_taken.strftime("%Y-%m-%d")

        # Fake ID is G(N/S)-CALYYYYMMDD-900-fileno
        prefix = 'GN-CAL' if 'north' in ad.telescope().lower() else 'GS-CAL'

        prgid = "{}{}".format(prefix, date_taken.strftime("%Y%m%d"))
        obsid = "{}-900".format(prgid)

        m = fitsfilenamecre.match(ad.filename)
        if m:
            fileno = m.group("fileno")
            try:
                fileno = int(fileno)
            except:
                fileno = None
        else:
            fileno = None

        # Use a random number if the file doesn't have a Gemini filename
        if fileno is None:
            import random
            fileno = random.randint(1,999)
        datalabel = "{}-{:03d}".format(obsid, fileno)

        # Set class, type, object to generic defaults
        ad.phu.set("OBSCLASS", "partnerCal", keyword_comments["OBSCLASS"])
        if "fringe" in caltype:
            ad.phu.set("OBSTYPE", "FRINGE", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "Fringe Frame", keyword_comments["OBJECT"])
        elif "sky" in caltype:
            ad.phu.set("OBSTYPE", "SKY", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "Sky Frame", keyword_comments["OBJECT"])
        elif "flat" in caltype:
            ad.phu.set("OBSTYPE", "FLAT", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "Flat Frame", keyword_comments["OBJECT"])
        else:
            raise ValueError("Caltype {} not supported".format(caltype))

        # Blank out program information
        ad.phu.set("GEMPRGID", prgid, keyword_comments["GEMPRGID"])
        ad.phu.set("OBSID", obsid, keyword_comments["OBSID"])
        ad.phu.set("DATALAB", datalabel, keyword_comments["DATALAB"])

        # Set release date
        ad.phu.set("RELEASE", release, keyword_comments["RELEASE"])

        # Blank out positional information
        ad.phu.set("RA", 0.0, keyword_comments["RA"])
        ad.phu.set("DEC", 0.0, keyword_comments["DEC"])

        # Blank out RA/Dec in WCS information in PHU if present
        if ad.phu.get("CRVAL1") is not None:
            ad.phu.set("CRVAL1", 0.0, keyword_comments["CRVAL1"])
        if ad.phu.get("CRVAL2") is not None:
            ad.phu.set("CRVAL2", 0.0, keyword_comments["CRVAL2"])

        # Do the same for each extension as well as the object name
        # Can't do simply with ad.hdr.set() because we don't know
        # what's already in each extension
        for ext in ad:
            if ext.hdr.get("CRVAL1") is not None:
                ext.hdr.set("CRVAL1", 0.0, keyword_comments["CRVAL1"])
            if ext.hdr.get("CRVAL2") is not None:
                ext.hdr.set("CRVAL2", 0.0, keyword_comments["CRVAL2"])
            if ext.hdr.get("OBJECT") is not None:
                if "fringe" in caltype:
                    ext.hdr.set("OBJECT", "Fringe Frame",
                                      keyword_comments["OBJECT"])
                elif "sky" in caltype:
                    ext.hdr.set("OBJECT", "Sky Frame",
                                      keyword_comments["OBJECT"])
                elif "flat" in caltype:
                    ext.hdr.set("OBJECT", "Flat Frame",
                                      keyword_comments["OBJECT"])
    return adinput


def filename_updater(adinput=None, infilename='', suffix='', prefix='',
                     strip=False):
    """
    This function is for updating the file names of astrodata objects.
    A prefix and/or suffix can be added, either to the current filename
    or to the original filename (strip=True does NOT attempt to parse
    the current filename to find the original root).

    Note: 
    1.if the input filename has a path, the returned value will have
    path stripped off of it.
    2. if strip is set to True, then adinput must be defined.

    Parameters
    ----------
    adinput: AstroData
        input astrodata instance having its filename updated
    suffix: str
        string to put between end of current filename and extension
    prefix: str
        string to put at the beginning of a filename
    strip: bool
        if True, use the original filename of the AD object, not what it has now
    """
    # We need the original filename if we're going to strip
    if strip:
        try:
            filename = adinput.phu['ORIGNAME']
        except KeyError:
            # If it's not there, grab the AD attr instead and add the keyword
            filename = adinput.orig_filename
            adinput.phu.set('ORIGNAME', filename,
                            'Original filename prior to processing')
    else:
        filename = infilename if infilename else adinput.filename

    # Possibly, filename could be None
    try:
        name, filetype = os.path.splitext(filename)
    except AttributeError:
        name, filetype = '', '.fits'

    outFileName = prefix+name+suffix+filetype
    return outFileName

@handle_single_adinput
def finalise_adinput(adinput=None, timestamp_key=None, suffix=None,
                     allow_empty=False):
    """
    Adds a timestamp and updates the filename of one or more AstroData
    instances to mark the end of a step in the reduction.

    Parameters
    ----------
    adinput: list/AD
        input AstroData object or list thereof
    timestamp_key: str/None
        name of keyword to add with the timestamp
    suffix: str/None
        suffix to add to files
    allow_empty: bool
        allows an empty list to be sent and returned

    Returns
    -------
    list
        of updated AD objects
    """
    if not (adinput or allow_empty):
        raise ValueError("Empty input list received")

    adoutput_list = []
    
    # Loop over each input AstroData object in the list
    for ad in adinput:
        # Add the appropriate time stamps to the PHU
        if timestamp_key is not None:
            mark_history(adinput=ad, keyword=timestamp_key)
        # Update the filename
        if suffix is not None:
            ad.filename = filename_updater(adinput=ad, suffix=suffix,
                                           strip=True)
        adoutput_list.append(ad)
    return adoutput_list


def fit_continuum(ad):
    """
    This function fits Gaussians to the spectral continuum, a bit like
    clip_sources for spectra
    
    Parameters
    ----------
    ad: AstroData
        input image

    Returns
    -------
    list
        of Tables with information about the sources it found
    """
    log = logutils.get_logger(__name__)
    import warnings

    good_sources = []
    
    # Get the pixel scale
    pixel_scale = ad.pixel_scale()
    
    # Set full aperture to 4 arcsec
    ybox = int(2.0 / pixel_scale)

    # Average 512 unbinned columns together
    xbox = 256 / ad.detector_x_bin()

    # Average 16 unbinned background rows together
    bgbox = 8 / ad.detector_x_bin()

    # Initialize the Gaussian width to FWHM = 1.2 arcsec
    init_width = 1.2 / (pixel_scale * (2 * np.sqrt(2 * np.log(2))))

    # Ignore spectrum if not >1.5*background
    s2n_bg = 1.5

    # Ignore spectrum if mean not >.9*std
    s2n_self = 0.9

    tags = ad.tags
    for ext in ad:
        data = ext.data
        dqdata = ext.mask
        vardata = ext.variance

        ####here - dispersion axis
        # Taking the 95th percentile should remove CRs
        signal = np.percentile(np.where(dqdata==0, data, 0), 95, axis=1)
        acq_star_positions = ad.phu.get("ACQSLITS")
        if acq_star_positions is None:
            if 'MOS' in tags:
                log.warning("{} is MOS but has no acquisition slits. "
                            "Not trying to find spectra.".format(ad.filename))
                if not hasattr(ad, 'MDF'):
                    log.warning("No MDF is attached. Did addMDF find one?")
                continue
            else:
                shuffle = int(ad.nod_pixels() / ad.detector_y_bin())
                centers = [shuffle + np.argmax(signal[shuffle:shuffle*2])]
                half_widths = [ybox]
        else:
            try:
                centers = []
                half_widths = []
                for x in acq_star_positions.split():
                    c,w = x.split(':')
                    # The -1 here because of python's 0-count
                    centers.append(int(c)-1)
                    half_widths.append(int(0.5*int(w)))
            except ValueError:
                log.warning("Image {} has unparseable ACQSLITS keyword".
                            format(ad.filename))
                centers = [np.argmax(signal)]
                half_widths = [ybox]
        
        fwhm_list = []
        y_list = []
        x_list = []
        weight_list = []
                
        for center,hwidth in zip(centers,half_widths):
            if center+hwidth>data.shape[0]:
                #print 'too high'
                continue
            if center-hwidth<0:
                #print 'too low'
                continue
            
            # Don't think it needs to do an overall check for each object
            #bg_mean = np.mean([data[center - ybox - bgbox:center-ybox],
            #                   data[center + ybox:center + ybox + bgbox]], dtype=np.float64)
            #ctr_mean = np.mean(data[center,dqdata[center]==0], dtype=np.float64)
            #ctr_std = np.std(data[center,dqdata[center]==0])
    
            #print 'mean ctr',ctr_mean,ctr_std
            #print 'mean bg',bg_mean
    
            #if ctr_mean < s2n_bg * bg_mean:
            #    print 'too faint'
            #    continue
            #if ctr_mean < s2n_self * ctr_std:
            #    print 'too noisy'
            #    continue
            
            for i in range(xbox, data.shape[1]-xbox, xbox):
                databox = data[center-hwidth-1:center+hwidth,i-xbox:i+xbox]
                if dqdata is None:
                    dqbox = np.zeros_like(databox)
                else:
                    dqbox = dqdata[center-hwidth-1:center+hwidth,i-xbox:i+xbox]
                if vardata is None:
                    varbox = databox # Any better ideas?
                else:
                    varbox = vardata[center-hwidth-1:center+hwidth,i-xbox:i+xbox]

                # Collapse in dispersion direction, using good pixels only
                dqcol = np.sum(dqbox==0, axis=1)
                if np.any(dqcol==0):
                    continue
                col = np.sum(databox, axis=1) / dqcol
                maxflux = np.max(abs(col))
                
                # Crude SNR test; is target bright enough in this wavelength range?
                if np.percentile(col,90)*xbox < np.mean(databox[dqbox==0]) \
                            + 10*np.sqrt(np.median(varbox)):
                    continue

                # Check that the spectrum looks like a continuum source.
                # This is needed to avoid cases where another slit is shuffled onto
                # the acquisition slit, resulting in an edge that the code tries
                # to fit with a Gaussian. That's not good, but if source is very
                # bright and dominates spectrum, then it doesn't matter.
                # All non-N&S data should pass this step, which checks whether 80%
                # of the spectrum has SNR>2
                spectrum = databox[np.argmax(col),:]
                if np.percentile(spectrum,20) < 2*np.sqrt(np.median(varbox)):
                    continue
                
                if 'NODANDSHUFFLE' in tags:
                    # N&S; background should be close to zero
                    bg = models.Const1D(0.)
                    # Fix background=0 if slit is in region where sky-subtraction will occur 
                    if center > ad.nod_pixels()/ad.detector_y_bin():
                            bg.amplitude.fixed = True
                else:
                    # Not N&S; background estimated from image
                    bg = models.Const1D(
                            amplitude=np.median([data[center-ybox-bgbox:center-ybox],
                            data[center+ybox:center+ybox+bgbox,i-xbox:i+xbox]], 
                            dtype=np.float64))
                g_init = models.Gaussian1D(amplitude=maxflux, mean=np.argmax(col), 
                            stddev=init_width) + models.Gaussian1D(amplitude=-maxflux,
                                mean=np.argmin(col), stddev=init_width) + bg
                # Set one profile to be +ve, one -ve, and widths equal
                g_init.amplitude_0.min = 0.
                g_init.amplitude_1.max = 0.
                g_init.stddev_1.tied = lambda f: f.stddev_0
                fit_g = fitting.LevMarLSQFitter()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    g = fit_g(g_init, np.arange(len(col)), col)
                
                #print g
                #x = np.arange(len(col))
                #plt.plot(x, col)
                #plt.plot(x, g(x))
                #plt.show()
                if fit_g.fit_info['ierr']<5:
                    # This is kind of ugly and empirical; philosophy is that peak should
                    # be away from the edge and should be similar to the maximum
                    if (g.amplitude_0 > 0.5*maxflux and
                        g.mean_0 > 1 and g.mean_0 < len(col)-2):
                        fwhm = abs(2*np.sqrt(2*np.log(2))*g.stddev_0)
                        fwhm_list.append(fwhm)
                        y_list.append(center)
                        x_list.append(i)
                        # Add a "weight" (multiply by 1. to convert to float
                        # If only one spectrum, all weights will be basically the same
                        if g.mean_1 > g.mean_0:
                            weight_list.append(1.*max(g.mean_0,
                                                      2*hwidth-g.mean_1))
                        else:
                            weight_list.append(1.*max(g.mean_1,
                                                      2*hwidth-g.mean_0))


        # Now do something with the list of measurements
        fwhm_pix = np.array(fwhm_list)
        fwhm_arcsec = pixel_scale * fwhm_pix

        table = Table([x_list, y_list, fwhm_pix, fwhm_arcsec, weight_list],
                    names=("x", "y", "fwhm", "fwhm_arcsec", "weight"))
        #plt.hist(rec["fwhm_arcsec"], 10)
        #plt.show()

        # Clip outliers in FWHM
        if len(table) >= 3:
            table = table[~stats.sigma_clip(table['fwhm_arcsec']).mask]

        good_sources.append(table)
    return good_sources

def log_message(function=None, name=None, message_type=None):
    """
    Creates a log message describing some function-related fun, e.g.,
    log_message('primitive', 'makeFringe', 'starting')

    Parameters
    ----------
    function: str
        type of the function being messaged about
    name: str
        name of the function being messaged about
    message_type
        what's happening to this function

    Returns
    -------
    str
        the message
    """
    message = None
    if message_type in ['calling', 'starting', 'finishing']:
        message = '{} the {} {}'.format(message_type.capitalize(), function, name)
    elif message_type == 'completed':
        message = 'The {} {} completed successfully'.format(name, function)
    return message

def make_dict(key_list=None, value_list=None):
    """
    The make_dict function creates a dictionary with the elements in 'key_list'
    as the key and the elements in 'value_list' as the value to create an
    association between the input science dataset (the 'key_list') and a, for
    example, dark that is needed to be subtracted from the input science
    dataset. This function also does some basic checks to ensure that the
    filters, exposure time etc are the same. No it doesn't.

    Parameters
    ----------
    key_list: list of AstroData objects
        the keys for the dict
    value_list: list of AstroData objects
        the values for the dict

    Returns
    -------
    dict
        the dict made from the keys and values
    """
    # Check the inputs have matching filters, binning and SCI shapes.
    ret_dict = {}
    if not isinstance(key_list, list):
        key_list = [key_list]
    if not isinstance(value_list, list):
        value_list = [value_list]
    # We allow only one value that can be assigned to multiple keys
    if len(value_list) == 1:
        value_list *= len(key_list)

    for key, value in zip(key_list, value_list):
        ret_dict[key] = value
        # Old error for incompatible list lengths
        #   msg = """Number of AstroData objects in key_list does not match
        #   with the number of AstroData objects in value_list. Please provide
        #   lists containing the same number of AstroData objects. Please
        #   supply either a single AstroData object in value_list to be applied
        #   to all AstroData objects in key_list OR the same number of
        #   AstroData objects in value_list as there are in key_list"""
    
    return ret_dict

def make_lists(key_list=None, value_list=None, force_ad=False):
    """
    The make_list function returns two lists, one of the keys and one of the
    values. It ensures that both inputs are made into lists if they weren't
    originally. It also expands the list of values to be the same length as
    the list of keys, if only one value was given.

    Parameters
    ----------
    key_list: list of AstroData objects
        the keys for the dict
    value_list: list of AstroData objects
        the values for the dict
    force_ad: bool
        coerce strings into AD objects?

    Returns
    -------
    2-tuple of lists
        the lists made from the keys and values
    """
    if not isinstance(key_list, list):
        key_list = [key_list]
    if not isinstance(value_list, list):
        value_list = [value_list]
    # We allow only one value that can be assigned to multiple keys
    if len(value_list) == 1:
        value_list *= len(key_list)
    if force_ad:
        key_list = [x if isinstance(x, astrodata.AstroData) or x is None else
                    astrodata.open(x) for x in key_list]
        # We only want to open as many AD objects as there are unique entries
        # in value_list, so collapse to set and multiple keys with the same
        # value will be assigned references to the same open AD object
        ad_map_dict = {}
        for x in set(value_list):
            try:
                ad_map_dict.update({x: x if isinstance(x, astrodata.AstroData)
                                        or x is None else astrodata.open(x)})
            except:
                ad_map_dict.update({x: None})
        value_list = [ad_map_dict[x] for x in value_list]

    return key_list, value_list

@handle_single_adinput
def mark_history(adinput=None, keyword=None, primname=None, comment=None):
    """
    Add or update a keyword with the UT time stamp as the value (in the form
    <YYYY>-<MM>-<DD>T<HH>:<MM>:<SS>) to the header of the PHU of the AstroData
    object to indicate when and what function was just performed on the
    AstroData object

    Parameters
    ----------
    adinput: list/AstroData
        the input file(s) to be timestamped
    keyword: str
        name of the keyword to be added/modified
    primname: str
        name of the primitive calling this
    comment: str
        comment (if any) to be added to the keyword
    """
    log = logutils.get_logger(__name__)

    try:
        assert keyword
    except AssertionError:
        log.error("TypeError: A keyword was not received.")
        raise TypeError("argument 'keyword' required")

    # Get the current time to use for the time of last modification
    tlm = datetime.now().isoformat()[0:-7]
    
    # Construct the default comment
    if comment is None:
        comment = "UT time stamp for {}".format(primname if primname
                                                else keyword)
    
    # The GEM-TLM keyword will always be added or updated
    keyword_dict = {"GEM-TLM": "UT last modification with GEMINI",
                    keyword: comment}

    # Loop over each input AstroData object in the input list
    for ad in adinput:
        for key, comm in keyword_dict.iteritems():
            ad.phu.set(key, tlm, comm)

    return


def measure_bg_from_image(ad, sampling=10, value_only=False, gaussfit=True):
    """
    Return background value, and its std deviation, as measured directly
    from pixels in the SCI image. DQ plane are used (if they exist)
    If extver is set, return a double for that extension only,
    otherwise return a list of doubles, as long as the number of extensions.

    Parameters
    ----------
    ad: AstroData
        input image (NOT a list)
    extver: int/None
        if not None, use only this extension
    sampling: int
        1-in-n sampling factor
    value_only: bool
        if True, return only background values, not the standard deviations
    gaussfit: bool
        if True, fit a Gaussian to the pixel values, instead of sigma-clipping?

    Returns
    -------
    list/value/tuple
        if use_extver is set, returns a bg value or (bg, std) tuple; otherwise
        returns a list of such things
    """
    try:
        input_list = [ext for ext in ad]
    except:
        input_list = [ext]

    output_list = []
    for ext in input_list:
        # Use DQ and OBJMASK to flag pixels
        flags = ext.mask | getattr(ext, 'OBJMASK', 0) if ext.mask is not None \
            else getattr(ext, 'OBJMASK', None)
        bg_data = ext.data[flags==0] if flags is not None else ext.data

        bg_data = bg_data.ravel()[::sampling]
        if len(bg_data) > 0:
            if gaussfit:
                # An ogive fit is more robust than a histogram fit
                bg_data = np.sort(bg_data)
                bg = np.median(bg_data)
                bg_std = 0.5*(np.percentile(bg_data, 84.13) -
                              np.percentile(bg_data, 15.87))
                g_init = CumGauss1D(bg, bg_std)
                fit_g = fitting.LevMarLSQFitter()
                g = fit_g(g_init, bg, np.linspace(0.,1.,len(bg_data)+1)[1:])
                bg, bg_std = g.mean.value, abs(g.stddev.value)
                #binsize = bg_std * 0.1
                # Fit from -5 to +1 sigma
                #bins = np.arange(bg - 5 * bg_std, bg + bg_std, binsize)
                #histdata, _ = np.histogram(bg_data, bins)
                # bin centers
                #x = bins[:-1] + 0.5 * (bins[1] - bins[0])
                # Eliminate bins with no data (e.g., if data are quantized)
                #x = x[histdata > 0]
                #histdata = histdata[histdata > 0]
                #g_init = models.Gaussian1D(amplitude=np.max(histdata),
                #                           mean=bg, stddev=bg_std)
                #fit_g = fitting.LevMarLSQFitter()
                #g = fit_g(g_init, x, histdata)
                #bg, bg_std = g.mean.value, abs(g.stddev.value)
            else:
                # Sigma-clipping will screw up the stats of course!
                bg_data = stats.sigma_clip(bg_data, sigma=2.0, iters=2)
                bg_data = bg_data.data[~bg_data.mask]
                bg = np.median(bg_data)
                bg_std = np.std(bg_data)
        else:
            bg, bg_std = None, None

        if value_only:
            output_list.append(bg)
        else:
            output_list.append((bg, bg_std, len(bg_data)))

    return output_list


def measure_bg_from_objcat(ad, min_ok=5, value_only=False):
    """
    Return a list of triples of background values, and their std deviations
    derived from the OBJCATs in ad, plus the number of objects used.
    If there are too few good BG measurements, None is returned.
    If the input has SCI extensions, then the output list contains one tuple
    per SCI extension, even if no OBJCAT is associated with that extension

    Parameters
    ----------
    ad: AstroData/Table
        image with OBJCATs to measure background from, or OBJCAT
    min_ok: int
        return None if fewer measurements than this (after sigma-clipping)
    value_only: bool
        if True, only return the background values, not the other stuff

    Returns
    -------
    list
        as described above
    """
    # Populate input list with either the OBJCAT or a list
    input_list = [ad] if isinstance(ad, Table) else [
        getattr(ext, 'OBJCAT', None) for ext in ad]

    output_list = []
    for objcat in input_list:
        bg = None
        bg_std = None
        nsamples = None
        if objcat is not None:
            bg_data = objcat['BACKGROUND']
            # Don't use objects with dodgy flags
            if len(bg_data)>0 and not np.all(bg_data==-999):
                flags = objcat['FLAGS']
                dqflags = objcat['IMAFLAGS_ISO']
                if not np.all(dqflags==-999):
                    # Non-linear/saturated pixels are OK
                    flags |= (dqflags & 65529)
                bg_data = bg_data[flags==0]
                # Sigma-clip, and only give results if enough objects are left
                if len(bg_data) > min_ok:
                    clipped_data = stats.sigma_clip(bg_data, sigma=3.0, iters=1)
                    if np.sum(~clipped_data.mask) > min_ok:
                        bg = np.mean(clipped_data)
                        bg_std = np.std(clipped_data)
                        nsamples = np.sum(~clipped_data.mask)
        if value_only:
            output_list.append(bg)
        else:
            output_list.append((bg, bg_std, nsamples))
    return output_list

def obsmode_add(ad):
    """
    Adds 'OBSMODE' keyword to input PHU for IRAF routines in GMOS package

    Parameters
    ----------
    ad: AstroData
        image to fix the PHU of

    Returns
    -------
    AstroData
        the fixed image
    """
    tags = ad.tags
    if 'GMOS' in tags:
        if 'PREPARED' in tags:
            try:
                ad.phu.set('PREPARE', ad.phu['GPREPARE'],
                           'UT Time stamp for GPREPARE')
            except:
                ad.phu.set('GPREPARE', ad.phu['PREPARE'],
                           'UT Time stamp for GPREPARE')

        if {'PROCESSED', 'BIAS'}.issubset(tags):
            mark_history(adinput=ad, keyword="GBIAS",
                            comment="Temporary key for GIREDUCE")
        if {'LS', 'FLAT'}.issubset(tags):
            mark_history(adinput=ad, keyword="GSREDUCE",
                comment="Temporary key for GSFLAT")

        # Reproducing inexplicable behaviour of the old system
        try:
            typeStr, = {'IMAGE', 'IFU', 'MOS', 'LS'} & tags
        except:
            typeStr = 'LS'

        if typeStr == 'LS':
            typeStr = 'LONGSLIT'
        ad.phu.set('OBSMODE', typeStr,
                   'Observing mode (IMAGE|IFU|MOS|LONGSLIT)')
    return ad

def obsmode_del(ad):
    """
    Deletes the 'OBSMODE' keyword from the PHUs of images produced by
    IRAF routines in the GMOS package.

    Parameters
    ----------
    ad: AstroData
        image to remove header keywords from

    Returns
    -------
    AstroData
        the fixed image
    """
    if 'GMOS' in ad.tags:
        for keyword in ['OBSMODE', 'GPREPARE', 'GBIAS', 'GSREDUCE']:
            if keyword in ad.phu:
                ad.phu.remove(keyword)
    return ad
    

def parse_sextractor_param(param_file):
    """
    Provides a list of columns being produced by SExtractor

    Parameters
    ----------
    default_dict: dict
        dictionary containing paths to the SExtractor input files

    Returns
    -------
    list
        names of all the columns in the SExtractor output catalog
    """
    columns = []
    fp = open(param_file)
    for line in fp:
        fields = line.split()
        if len(fields)==0:
            continue
        if fields[0].startswith("#"):
            continue
        name = fields[0]
        columns.append(name)
    return columns

def read_database(ad, database_name=None, input_name=None, output_name=None):
    """
    Reads IRAF wavelength calibration files from a database and attaches
    them to an AstroData object

    Parameters
    ----------
    ad: AstroData
        AstroData object to which the WAVECAL tables will be attached
    database_name: st
        IRAF database directory name
    input_name: str
        filename of the image used for wavelength calibration
        (if None, use the filename of the input AD object)
    output_name: str
        name to be assigned to the output table record
        (if None, use the filename of the input AD object)
    Returns
    -------
    AstroData
        version of the input AD object with WAVECAL tables attached
    """
    if database_name is None:
        raise IOError('No database name specified')
    if not os.path.isdir(database_name):
        raise IOError('Database directory {} does not exist'.format(
            database_name))
    if input_name is None:
        input_name = ad.filename
    if output_name is None:
        output_name = ad.filename

    basename = os.path.basename(input_name)
    basename,filetype = os.path.splitext(basename)
    out_basename = os.path.basename(output_name)
    out_basename,filetype = os.path.splitext(out_basename)

    for ext in ad:
        extver = ext.hdr['EXTVER']
        record_name = '{}_{:0.3d}'.format(basename, extver)
        db = at.SpectralDatabase(database_name,record_name)
        out_record_name = '{}_{:0.3d}'.format(out_basename, extver)
        table = db.as_binary_table(record_name=out_record_name)

        ext.WAVECAL = table
    return ad

def tile_objcat(adinput, adoutput, ext_mapping, sx_dict=None):
    """
    This function tiles together separate OBJCAT extensions, converting
    the pixel coordinates to the new WCS.

    Parameters
    ----------
    adinput: AstroData
        input AD object with all the OBJCATs
    adoutput: AstroData
        output AD object to which we want to append the new tiled OBJCATs
    ext_mapping: array
        contains the output extension onto which each input has been placed
    sx_dict: dict
        SExtractor dictionary
    """
    for ext, header in zip(adoutput, adoutput.header[1:]):
        outextver = ext.hdr['EXTVER']
        output_wcs = WCS(header)
        indices = [i for i in range(len(ext_mapping))
                   if ext_mapping[i] == outextver]
        inp_objcats = [adinput[i].OBJCAT for i in indices if
                       hasattr(adinput[i], 'OBJCAT')]

        if inp_objcats:
            out_objcat = vstack(inp_objcats, metadata_conflicts='silent')

            # Get new pixel coords for objects from RA/Dec and the output WCS
            ra = out_objcat["X_WORLD"]
            dec = out_objcat["Y_WORLD"]
            newx, newy = output_wcs.all_world2pix(ra, dec, 1)
            out_objcat["X_IMAGE"] = newx
            out_objcat["Y_IMAGE"] = newy

            # Remove the NUMBER column so add_objcat renumbers
            out_objcat.remove_column('NUMBER')

            adoutput = add_objcat(adinput=adoutput, extver=outextver,
                            replace=True, table=out_objcat, sx_dict=sx_dict)
    return adoutput

@handle_single_adinput
def trim_to_data_section(adinput=None, keyword_comments=None):
    """
    This function trims the data in each extension to the section returned
    by its data_section descriptor. This is intended for use in removing
    overscan sections, or other unused parts of the data array.

    Parameters
    ----------
    adinput: list/AD
        input image(s) to be trimmed
    keyword_comments: dict

    Returns
    -------
    list/AstroData
        same as input images, but trimmed
    """
    log = logutils.get_logger(__name__)

    try:
        assert isinstance(keyword_comments, dict)
    except AssertionError:
        log.error("TypeError: keyword comments dict was not received.")
        raise TypeError("keyword comments dict required")

    # Initialize the list of output AstroData objects
    adoutput_list = []
 
    for ad in adinput:
        for ext in ad:
            # Get data section as string and as a tuple
            datasecStr = ext.data_section(pretty=True)
            dsl = ext.data_section()

            # Get the keyword associated with the data_section descriptor
            ds_kw = ext._keyword_for('data_section')

            # Check whether data need to be trimmed
            sci_shape = ext.data.shape
            if (sci_shape[0]==dsl.y2 and sci_shape[1]==dsl.x2 and
                dsl.x1==0 and dsl.y1==0):
                log.fullinfo('No changes will be made to {}[*,{}], since '
                             'the data section matches the data shape'.format(
                             ad.filename,ext.hdr['EXTVER']))
                continue

            # Update logger with the section being kept
            log.fullinfo('For {}:{}, keeping the data from the section {}'.
                         format(ad.filename, ext.hdr['EXTVER'], datasecStr))

            # Trim SCI, VAR, DQ to new section
            ext.reset(ext.nddata[dsl.y1:dsl.y2,dsl.x1:dsl.x2])
            # And OBJMASK (if it exists)
            # TODO: should check more generally for any image extensions
            if hasattr(ext, 'OBJMASK'):
                ext.OBJMASK = ext.OBJMASK[dsl.y1:dsl.y2,dsl.x1:dsl.x2]

            # Update header keys to match new dimensions
            newDataSecStr = '[1:{},1:{}]'.format(dsl.x2-dsl.x1, dsl.y2-dsl.y1)
            ext.hdr.set('NAXIS1', dsl.x2-dsl.x1, keyword_comments['NAXIS1'])
            ext.hdr.set('NAXIS2', dsl.y2-dsl.y1, keyword_comments['NAXIS2'])
            ext.hdr.set(ds_kw, newDataSecStr, comment=keyword_comments[ds_kw])
            ext.hdr.set('TRIMSEC', datasecStr, comment=keyword_comments['TRIMSEC'])

            # Update WCS reference pixel coordinate
            try:
                crpix1 = ext.hdr['CRPIX1'] - dsl.x1
                crpix2 = ext.hdr['CRPIX2'] - dsl.y1
            except:
                log.warning("Could not access WCS keywords; using dummy "
                            "CRPIX1 and CRPIX2")
                crpix1 = 1
                crpix2 = 1
            ext.hdr.set('CRPIX1', crpix1, comment=keyword_comments["CRPIX1"])
            ext.hdr.set('CRPIX2', crpix2, comment=keyword_comments["CRPIX2"])
        adoutput_list.append(ad)

    return adoutput_list

def write_database(ad, database_name=None, input_name=None):
    """
    Write out IRAF database files containing a wavelength calibration

    Parameters
    ----------
    ad: AstroData
        image containing a WAVECAL extension
    database_name: str
        IRAF database directory
    input_name: str
        filename of the image associated with the WAVECAL table
        (if None, use the filename of the AD object)
    """
    if input_name is None:
        input_name = ad.filename

    basename = os.path.basename(input_name)
    basename,filetype = os.path.splitext(basename)

    for ext in ad:
        record_name = '{}_{:0.3d}'.format(basename, ext.EXTVER)
        db = at.SpectralDatabase(binary_table=ext.WAVECAL,
                                 record_name=record_name)
        db.write_to_disk(database_name=database_name)
    return


class ExposureGroup:
    """
    An ExposureGroup object maintains a record of AstroData instances that
    are spatially associated with the same general nod position or dither
    group, along with their co-ordinates and other properties of the group.

    This object can be interrogated as to whether a given set of co-ordinates
    are within the same field of view as the centroid of an existing group
    and therefore on the same source. It can then be instructed to incorporate
    a point into the group if appropriate, providing a simple agglomerative
    clustering algorithm.
    """

    # The reason this class isn't built on a more universal version that
    # doesn't require AstroData is that it's currently unclear what the API
    # at that level should look like -- it would probably be most useful to
    # pass around nddata instances rather than lists or dictionaries but
    # that's not even well defined within AstroPy yet.

    def __init__(self, adinput, pkg=None, frac_FOV=1.0):
        """
        :param adinputs: an exposure list from which to initialize the group
            (currently may not be empty)
        :type adinputs: list of AstroData instances

        :param pkg: Package name of the 

        :param frac_FOV: proportion by which to scale the area in which
            points are considered to be within the same field, for tweaking
            the results in borderline cases (eg. to avoid co-adding target
            positions right at the edge of the field).
        :type frac_FOV: float
        """
        if not isinstance(adinput, list):
            adinput = [adinput]
        # Make sure the field scaling is valid:
        if not isinstance(frac_FOV, numbers.Number) or frac_FOV < 0.:
            raise IOError('frac_FOV must be >= 0.')

        # Initialize members:
        self.members = {}
        self.package = pkg
        self._frac_FOV = frac_FOV
        self.group_cen = (0., 0.)
        self.add_members(adinput)

        # Use the first list element as a reference for getting the
        # instrument properties:
        ref_ad = adinput[0]

        # Here we used to define self._pointing_in_field, pointing to the
        # instrument-specific back-end function, but that's been moved to a
        # separate function in this module since it may be generally useful.

    def pointing_in_field(self, position):
        """
        Determine whether or not a point falls within the same field of view
        as the centre of the existing group (using the function of the same
        name).

        :param position: An AstroData instance to compare with the centroid
          of the set of existing group members.
        :type position: tuple, list, AstroData

        :returns: Whether or not the input point is within the field of view
            (adjusted by the frac_FOV specified when creating the group).
        :rtype: boolean
        """

        # Check co-ordinates WRT the group centre & field of view as
        # appropriate for the instrument:
        return pointing_in_field(position, self.package, self.group_cen,
                                 frac_FOV=self._frac_FOV)

    def __len__(self):
        return len(self.members)

    def __repr__(self):
        return str(self.list())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # A Python dictionary equality seems to take care of comparing
            # groups nicely, irrespective of ordering:
            return self.members == other.members
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def list(self):
        """
        List the AstroData instances associated with this group.

        :returns: Exposure list
        :rtype: list of AstroData instances
        """
        return self.members.keys()

    def add_members(self, adinput):
        """
        Add one or more new points to the group.

        :param adinputs: A list of AstroData instances to add to the group
            membership. 
        :type adinputs: AstroData, list of AstroData instances
        """
        if not isinstance(adinput, list):
            adinput = [adinput]
        # How many points were there previously and will there be now?
        ngroups = self.__len__()
        ntot = ngroups + len(adinput)

        # This will be replaced by a descriptor that looks up the RA/Dec
        # of the field centre:
        addict = get_offset_dict(adinput)

        # Complain sensibly if we didn't get valid co-ordinates:
        for ad in addict:
            for coord in addict[ad]:
                if not isinstance(coord, numbers.Number):
                    raise IOError('non-numeric co-ordinate %s ' \
                        % coord + 'from %s' % ad)

        # Add the new points to the group list:
        self.members.update(addict)

        # Update the group centroid to account for the new points:
        new_vals = addict.values()
        newsum = [sum(axvals) for axvals in zip(*new_vals)]
        self.group_cen = [(cval * ngroups + nval) / ntot \
          for cval, nval in zip(self.group_cen, newsum)]

def group_exposures(adinput, pkg=None, frac_FOV=1.0):

    """
    Sort a list of AstroData instances into dither groups around common
    nod positions, according to their WCS offsets.

    :param adinputs: A list of exposures to sort into groups.
    :type adinputs: list of AstroData instances

    :param pkg: Package name of the calling primitive. Used to determine
                correct package lookup tables. Passed through to
                ExposureGroup() call.
    :type pkg: <str>

    :param frac_FOV: proportion by which to scale the area in which
        points are considered to be within the same field, for tweaking
        the results in borderline cases (eg. to avoid co-adding target
        positions right at the edge of the field).
    :type frac_FOV: float

    :returns: One group of exposures per identified nod position.
    :rtype: tuple of ExposureGroup instances
    """

    # In principle divisive clustering algorithms have the best chance of
    # robust separation because of their top-down view of the problem.
    # However, an agglomerative algorithm is probably more practical to
    # implement here in the first instance, given that we can try to
    # constrain the problem using the known field size and remembering the
    # one-at-a-time case where the full list isn't available up front.

    # The FOV gives us a pretty good idea what's close enough to the base
    # position to be considered on-source, but it may not be enough on its
    # own to provide a threshold for intermediate nod distances for F2 given
    # IQ problems towards the edges of the field. OTOH intermediate offsets
    # are generally difficult to achieve anyway due to guide probe limits
    # when the instrumental FOV is comparable to that of the telescope.

    groups = []

    # Iterate over the input exposures:
    for ad in adinput:

        # Should this pointing be associated with an existing group?
        found = False
        for group in groups:
            if group.pointing_in_field(ad):
                group.add_members(ad)
                found = True
                break

        # If unassociated, start a new group:
        if not found:
            groups.append(ExposureGroup(ad, pkg, frac_FOV=frac_FOV))
            # if debug: print 'New group', groups[-1]

    # Here this simple algorithm could be made more robust for borderline
    # spacing (a bit smaller than the field size) by merging clusters that
    # have members within than the threshold after initial agglomeration.
    # This is an odd use case that's hard to classify automatically anyway
    # but the grouping info. might be more useful later, when stacking.

    # if debug: print 'Groups are', groups

    return tuple(groups)

def pointing_in_field(pos, package, refpos, frac_FOV=1.0, frac_slit=None):

    """
    Determine whether two telescope pointings fall within each other's
    instrumental field of view, such that point(s) at the centre(s) of one
    exposure's aperture(s) will still fall within the same aperture(s) of
    the other exposure or reference position.

    For direct imaging, this is the same question as "is point 1 within the
    field at pointing 2?" but for multi-object spectroscopy neither position
    at which the telescope pointing is defined will actually illuminate the
    detector unless it happens to coincide with a target aperture.

    This function has no knowledge of actual target position(s) within the
    field, providing accurate results for centred target(s) with the default
    frac_FOV=1.0. This should only be of concern in borderline cases where
    exposures are offset by approximately the width of the field.

    :param pos: Exposure defining the position & instrumental field of view
      to check.
    :type pos: AstroData instance

    :param refpos: Reference position/exposure (currently assumed to be
      Gemini p/q co-ordinates, but we intend to change that to RA/Dec).
    :type refpos: AstroData instance or tuple of floats

    :param frac_FOV: Proportion by which to scale the field size in order to
      adjust whether borderline points are considered within its boundaries.
    :type frac_FOV: float

    :param frac_slit: If defined, the maximum deviation from the slit centre
      that's still considered to be within the field, as a fraction of the
      slit's half-width. If None, the value of frac_FOV is used instead. For
      direct images this parameter is ignored.
    :type frac_slit: float

    :returns: Whether or not the pointing falls within the field (after
      adjusting for frac_FOV & frac_slit). 
    :rtype: boolean

    """
    log = logutils.get_logger(__name__)

    # This function needs an AstroData instance rather than just 2
    # co-ordinate tuples and a PA in order to look up the instrument for
    # which the field of view is defined and so that the back-end function
    # can distinguish between focal plane masks and access instrument-
    # specific MDF tables for MOS.

    # Use the first argument for looking up the instrument, since that's
    # the one that's always an AstroData instance because the reference point
    # doesn't always correspond to a single exposure.
    inst = pos.instrument().lower()

    # To keep the back-end functions simple, always pass them a dictionary
    # for the reference position (at least for now). All these checks add ~4%
    # in overhead for imaging.
    try:
        pointing = (refpos.phu['POFFSET'], refpos.phu['QOFFSET'])
    except AttributeError:
        if not isinstance(refpos, (list, tuple)) or \
           not all(isinstance(x, numbers.Number) for x in refpos):
            raise IOError('Parameter refpos should be a '
                'co-ordinate tuple or AstroData instance')
        # Currently the comparison is always 2D since we're explicitly
        # looking up POFFSET & QOFFSET:
        if len(refpos) != 2:
            raise IOError('Points to group must have the '
                    'same number of co-ords')
        pointing = refpos

    # Use a single scaling for slit length & width if latter unspecified:
    if frac_slit is None:
        frac_slit = frac_FOV

    # These values are cached in order to avoid the multiple-second overhead of
    # reading and evaling a look-up table when there are repeated queries for
    # the same instrument. Instead of private global variables we could use a
    # function-like class here, but that would make the API rather convoluted
    # in this simple case.
    global _FOV_lookup, _FOV_pointing_in_field

    # Look up the back-end implementation for the appropriate instrument,
    # or use the previously-cached one.
    # Build the lookup name to the instrument specific FOV module. In all 
    # likelihood this is 'Gemini'.
    FOV_mod = "geminidr.{}.lookups.FOV".format(inst)

    try:
        FOV = import_module(FOV_mod)
        _FOV_pointing_in_field = FOV.pointing_in_field
    except (ImportError, AttributeError):
        raise NameError("FOV.pointing_in_field() function not implemented for %s" % inst)

    # Execute it & return the results:
    return _FOV_pointing_in_field(pos, pointing, frac_FOV=frac_FOV,
                                  frac_slit=frac_slit)

# Since the following function will go away after redefining RA & Dec
# descriptors appropriately, I've put it here instead of in
# gemini_metadata_utils to avoid creating an import that's circular WRT
# existing imports and might later hang around causing trouble.
@handle_single_adinput
def get_offset_dict(adinput=None):
    """
    (To be deprecated)

    The get_offset_dict() function extracts a dictionary of co-ordinate offset
    tuples from a list of Gemini datasets, one per input AstroData instance.
    What's currently Gemini-specific is that POFFSET & QOFFSET header keywords
    are used; this could be abstracted via a descriptor once we decide how to
    do so generically (accounting for the need to know slit axes sometimes).
    
    :param adinputs: the AstroData objects
    :type adinput: list of AstroData
    
    :rtype: dictionary
    :return: a dictionary whose keys are the AstroData instances and whose
        values are tuples of (POFFSET, QOFFSET).

    """
    offsets = {}

    # Loop over AstroData instances:
    for ad in adinput:
        # Get the offsets from the primary header:
        poff = ad.phu['POFFSET']
        qoff = ad.phu['QOFFSET']
        # name = ad.filename
        name = ad  # store a direct reference
        offsets[name] = (poff, qoff)

    return offsets
