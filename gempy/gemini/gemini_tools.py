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
import json
import urllib2
import numbers

import pyfits as pf
import numpy  as np

from copy import deepcopy
from datetime import datetime
from importlib import import_module

from ..library import astrotools as at

from astropy.modeling import models, fitting
from astropy import stats

from astrodata import AstroData
from astrodata import __version__ as ad_version
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.ConfigSpace import lookup_path
from astrodata.utils.gemconstants import SCI, VAR, DQ
from astrodata.interface.slices import pixel_exts
from astrodata.interface.Descriptors import DescriptorValue
from astrodata.utils.Errors import DescriptorValueTypeError

#import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
def add_objcat(adinput=None, extver=1, replace=False, columns=None, sxdict=None):
    """
    Add OBJCAT table if it does not exist, update or replace it if it does.
    
    :param adinput: AD object(s) to add table to
    :type adinput: AstroData objects, either a single instance or a list
    
    :param extver: Extension number for the table (should match the science
                   extension).
    :type extver: int
    
    :param replace: Flag to determine if an existing OBJCAT should be
                    replaced or updated in place. If replace=False, the
                    length of all lists provided must match the number
                    of entries currently in OBJCAT.
    :type replace: boolean
    
    :param columns: Columns to add to table.  Columns named 'X_IMAGE',
                    'Y_IMAGE','X_WORLD','Y_WORLD' are required if making
                    new table.
    :type columns: dictionary of Pyfits Column objects with column names
                   as keys
    """
    
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)

    # ensure caller passes the sextractor default dictionary of parameters.
    try:
        assert isinstance(sxdict, dict) and sxdict.has_key('dq')
    except AssertionError:
        log.error("TypeError: A sextractor dictionary was not received.")
        raise TypeError("Require sextractor parameter dictionary.")
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinput)
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        
        # Parse sextractor parameters for the list of expected columns
        expected_columns = parse_sextractor_param(sxdict)

        # Append a few more that don't come from directly from sextractor
        expected_columns.extend(["REF_NUMBER","REF_MAG","REF_MAG_ERR",
                                 "PROFILE_FWHM","PROFILE_EE50"])
        
        # Loop over each input AstroData object in the input list
        for ad in adinput_list:
            
            # Check if OBJCAT already exists and just update if desired
            objcat = ad["OBJCAT", extver]
            if objcat and not replace:
                log.fullinfo("Table already exists; updating values.")
                for name in columns.keys():
                    objcat.data.field(name)[:] = columns[name].array
            else:
            
                # Make new table: x, y, ra, dec required
                x   = columns.get("X_IMAGE", None)
                y   = columns.get("Y_IMAGE", None)
                ra  = columns.get("X_WORLD", None)
                dec = columns.get("Y_WORLD", None)

                if x is None or y is None or ra is None or dec is None:
                    raise Errors.InputError("Columns X_IMAGE, Y_IMAGE, "\
                                            "X_WORLD, Y_WORLD must be present.")

                # Append columns in order of definition in sextractor params
                table_columns = []
                nlines = len(x.array)

                for name in expected_columns:
                    if name in ["NUMBER"]:
                        default = range(1, nlines+1)
                        format = "J"
                    elif name in ["FLAGS", "IMAFLAGS_ISO", "REF_NUMBER"]:
                        default = [-999] * nlines
                        format = "J"
                    else:
                        default = [-999] * nlines
                        format = "E"

                    # Get column from input if present, otherwise
                    # define a new Pyfits column with sensible placeholders
                    data = columns.get(name,
                                       pf.Column(name=name, format=format,
                                                 array=default))
                    table_columns.append(data)

                # Make new pyfits table
                col_def = pf.ColDefs(table_columns)
                tb_hdu  = pf.new_table(col_def)
                tb_ad   = AstroData(tb_hdu)
                tb_ad.rename_ext("OBJCAT", extver)
            
                # Replace old version or append new table to AD object
                if objcat:
                    log.fullinfo("Replacing existing OBJCAT in %s" % 
                                 ad.filename)
                    ad.remove(("OBJCAT", extver))
                ad.append(tb_ad)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


def array_information(adinput=None):
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinput)
    
    # Initialize the list of dictionaries of output array numbers
    # Keys will be (extname,extver)
    array_info_list = []

    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput_list:
            arrayinfo = {}
            # Get the number of science extensions
            nsciext = ad.count_exts(SCI)

            # Get the correct order of the extensions by sorting on
            # the first element in detector section
            # (raw ordering is whichever amps read out first)

            detsecs = ad.detector_section().as_list()

            if not isinstance(detsecs[0],list):
                detsecs = [detsecs]

            detx1      = [sec[0] for sec in detsecs]
            ampsorder  = range(1,nsciext+1)
            orderarray = np.array(zip(ampsorder,detx1), 
                                  dtype=[('ext', np.int), ('detx1', np.int)])
            orderarray.sort(order = 'detx1')

            if np.all(ampsorder == orderarray['ext']):
                in_order = True
            else:
                ampsorder = orderarray['ext']
                in_order = False
                
            # Get array sections for determining when
            # a new array is found
            arraysecs = ad.array_section().as_list()
            if not isinstance(arraysecs[0], list):
                arraysecs = [arraysecs]

            if len(arraysecs) != nsciext:
                arraysecs *= nsciext

            arrayx1 = [sec[0] for sec in arraysecs]

            # Initialize these so that first extension will always
            # start a new array
            last_detx1   = detx1[ampsorder[0]-1] - 1
            last_arrayx1 = arrayx1[ampsorder[0] - 1]

            arraynum  = {}
            num_array = 0
            amps_per_array = {}

            for i in ampsorder:
                sciext = ad[SCI,i]
                this_detx1 = detx1[i-1]
                this_arrayx1 = arrayx1[i-1]
                
                if (this_detx1 > last_detx1 and this_arrayx1 <= last_arrayx1):
                    # New array found
                    num_array += 1
                    amps_per_array[num_array] = 1
                else:
                    amps_per_array[num_array] += 1
                
                arraynum[(sciext.extname(), sciext.extver())] = num_array

            # Reference extension if tiling/mosaicing all data together
            try:
                refext = ampsorder[int((amps_per_array[2] + 1) / 2.0 - 1)
                                   + amps_per_array[1]]
            except KeyError:
                refext = None

            arrayinfo['array_number'] = arraynum
            arrayinfo['amps_order'] = ampsorder
            arrayinfo['amps_per_array'] = amps_per_array
            arrayinfo['reference_extension'] = refext

            # Append the output AstroData object to the list of output
            # AstroData objects
            array_info_list.append(arrayinfo)
        
        # Return the list of output AstroData objects
        return array_info_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


def calc_nbiascontam(adInputs=None, biassec=None):
    """
    This function will find the largest difference between the horizontal 
    component of every BIASSEC value and those of the biassec parameter. 
    The returned value will be that difference as an integer and it will be
    used as the value for the nbiascontam parameter used in the gireduce 
    call of the overscanSubtract primitive.
    
    :param adInputs: AstroData instance(s) to calculate the bias 
                     contamination 
    :type adInputs: AstroData instance in a list
    
    :param biassec: biassec parameter of format 
                    '[#:#,#:#],[#:#,#:#],[#:#,#:#]'
    :type biassec: string 
    
    """
    log = logutils.get_logger(__name__)
        
    try:
        # Prepare a stored value to be compared between the inputs
        retvalue=0
        # Loop through the inputs
        for ad in adInputs:
            # Split up the input triple list into three separate sections
            biassecStrList = biassec.split('],[')
            # Prepare the to-be list of lists
            biassecIntList = []
            for biassecStr in biassecStrList:
                # Use sectionStrToIntList function to convert 
                # each string version of the list into actual integer tuple 
                # and load it into the lists of lists
                # of form [y1, y2, x1, x2] 0-based and non-inclusive
                biassecIntList.append(sectionStrToIntList(biassecStr))
            
            # Setting the return value to be updated in the loop below    
            retvalue=0
            for ext in ad['SCI']:
                # Retrieving current BIASSEC value                    #  THIS WHERE THE 
                BIASSEC = ext.get_key_value('BIASSEC')                #  bias_section()
                # Converting the retrieved string into a integer list #  descriptor
                # of form [y1, y2, x1, x2] 0-based and non-inclusive  #  would be used!!!!
                BIASSEClist = sectionStrToIntList(BIASSEC)     #
                # Setting the lower case biassec list to the appropriate 
                # list in the lists of lists created above the loop
                biasseclist = biassecIntList[ext.extver() - 1]
                # Ensuring both biassec's have the same vertical coords
                if (biasseclist[0] == BIASSEClist[0]) and \
                (biasseclist[1] == BIASSEClist[1]):
                    # If overscan/bias section is on the left side of chip
                    if biasseclist[3] < 50: 
                        # Ensuring right X coord of both biassec's are equal
                        if biasseclist[2] == BIASSEClist[2]: 
                            # Set the number of contaminating columns to the 
                            # difference between the biassec's left X coords
                            nbiascontam = BIASSEClist[3] - biasseclist[3]
                        # If left X coords of biassec's don't match, set  
                        # number of contaminating columns to 4 and make a 
                        # error log message
                        else:
                            log.error('right horizontal components of '+
                                      'biassec and BIASSEC did not match, '+
                                      'so using default nbiascontam=4')
                            nbiascontam = 4
                    # If overscan/bias section is on the right side of chip
                    else: 
                        # Ensuring left X coord of both biassec's are equal
                        if biasseclist[3] == BIASSEClist[3]: 
                            # Set the number of contaminating columns to the 
                            # difference between the biassec's right X coords
                            nbiascontam = BIASSEClist[2] - biasseclist[2]
                        else:
                            log.error('left horizontal components of '+
                                      'biassec and BIASSEC did not match, '+
                                      'so using default nbiascontam=4') 
                            nbiascontam = 4
                # Overscan/bias section is not on left or right side of chip
                # , so set to number of contaminated columns to 4 and log 
                # error message
                else:
                    log.error('vertical components of biassec and BIASSEC '+
                              'parameters did not match, so using default '+
                              'nbiascontam=4')
                    nbiascontam = 4
                # Find the largest nbiascontam value throughout all chips  
                # and set it as the value to be returned  
                if nbiascontam > retvalue:  
                    retvalue = nbiascontam
        return retvalue
    # If all the above checks and attempts to calculate a new nbiascontam 
    # fail, make a error log message and return the value 4. so exiting 
    # 'gracefully'.        
    except:
        log.error('An error occurred while trying to calculate the '+
                  'nbiascontam, so using default value = 4')
        return 4 
             
  
def check_inputs_match(ad1=None, ad2=None, check_filter=True):
    """
    This function will check if the inputs match.  It will check the filter,
    binning and shape/size of the every SCI frames in the inputs.
    
    There must be a matching number of inputs for 1 and 2.
    
    :param ad1: input astrodata instance(s) to be check against ad2
    :type ad1: AstroData objects, either a single or a list of objects
                Note: inputs 1 and 2 must be matching length lists or single 
                objects
    
    :param ad2: input astrodata instance(s) to be check against ad1
    :type ad2: AstroData objects, either a single or a list of objects
                  Note: inputs 1 and 2 must be matching length lists or single 
                  objects
    """
    log = logutils.get_logger(__name__) 
    
    # Check inputs are both matching length lists or single objects
    if (ad1 is None) or (ad2 is None):
        log.error('Inputs ad1 and ad2 must not be None')
        raise Errors.ToolboxError('Either inputs ad1 or ad2 was None')
    if isinstance(ad1, list):
        if isinstance(ad2, list):
            if len(ad1) != len(ad2):
                log.error('Both ad1 and ad2 inputs must be lists of MATCHING'+
                          ' lengths.')
                raise Errors.ToolboxError('There were mismatched numbers ' \
                                          'of ad1 and ad2 inputs.')
    if isinstance(ad1, AstroData):
        if isinstance(ad2, AstroData):
            # casting both ad1 and ad2 inputs to lists for looping later
            ad1 = [ad1]
            ad2 = [ad2]
        else:
            log.error('Both ad1 and ad2 inputs must be lists of MATCHING'+
                      ' lengths.')
            raise Errors.ToolboxError('There were mismatched numbers of '+
                               'ad1 and ad2 inputs.')
    
    for count in range(0, len(ad1)):
        A = ad1[count]
        B = ad2[count]
        log.fullinfo('Checking inputs ' + A.filename+' and ' + B.filename)
        
        if A.count_exts('SCI') != B.count_exts('SCI'):
            log.error('Inputs have different numbers of SCI extensions.')
            raise Errors.ToolboxError('Mismatching number of SCI ' \
                                      'extensions in inputs')
        for sciA in A[SCI]:
            # grab matching SCI extensions from A's and B's
            extCount = sciA.extver()
            sciB = B[('SCI', extCount)]
            
            log.fullinfo('Checking SCI extension ' + str(extCount))
            
            # Check shape/size
            if sciA.data.shape != sciB.data.shape:
                log.error('Extensions have different shapes')
                raise Errors.ToolboxError('Extensions have different shape')
            
            # Check binning
            aX = sciA.detector_x_bin()
            aY = sciA.detector_y_bin()
            bX = sciB.detector_x_bin()
            bY = sciB.detector_y_bin()

            if (aX != bX) or (aY != bY):
                log.error('Extensions have different binning')
                raise Errors.ToolboxError('Extensions have different binning')
        
            # Check filter if desired
            if check_filter:
                if (sciA.filter_name().as_pytype() != 
                    sciB.filter_name().as_pytype()):
                    log.error('Extensions have different filters')
                    raise Errors.ToolboxError('Extensions have different ' +
                                              'filters')
        log.fullinfo('Inputs match')
    return


def matching_inst_config(ad1=None, ad2=None, check_exposure=False):
    """
    Compare two AstroData instances and report whether their instrument
    configurations are identical, including the exposure time (but excluding
    the telescope pointing). This function was added for NIR images but
    should probably be merged with check_inputs_match() above, after checking
    carefully that the changes needed won't break any GMOS processing.

    :param ad1: input AstroData instance to check against ad2
    :type ad1: AstroData
    
    :param ad2: input AstroData instance to check against ad1
    :type ad2: AstroData

    :param check_exposure: Require exposure times to match?
    :type check_exposure: bool
    """
    log = logutils.get_logger(__name__)

    log.debug('Comparing instrument config for AstroData instances:')

    if not isinstance(ad1, AstroData) or not isinstance(ad2, AstroData):
        raise Errors.ToolboxError('Inputs must be AstroData instances')

    # Some of the following repetition could probably be cleaned up using a
    # list of descriptors but there would still be exceptions and it's not
    # really all that cumbersome.

    result = True

    # Don't bother checking instrument(), as it has to match within an OBSID
    # and the following things couldn't all match anyway were it to differ.

    if ad1.count_exts('SCI') != ad2.count_exts('SCI'):
        result = False
        log.debug('  Number of SCI extensions differ')

    # I don't *think* reading the shape attribute causes the arrays to be
    # loaded into memory here -- in any case I can't see a way to unload an
    # array from an AstroData instance, as with "del ext['SCI'].data" in
    # PyFITS, so there's not much that can be done about it...
    for ext1, ext2 in zip(ad1['SCI'], ad2['SCI']):
        if ext1.data.shape != ext2.data.shape:
            result = False
            log.debug('  Array dimensions differ')
            break

    # Some of these checks seem a bit redundant, but I think some are used
    # for some instruments and others for others.

    # Descriptor comparisons fail for values of None, so use as_pytype(),
    # which seems to be harmless, if in any doubt.

    if ad1.data_section().as_pytype() != ad2.data_section().as_pytype():
        result = False
        log.debug('  Data sections differ')

    if ad1.detector_roi_setting().as_pytype() != \
       ad2.detector_roi_setting().as_pytype():
        result = False
        log.debug('  Detector regions of interest differ')

    if ad1.read_mode().as_pytype() != ad2.read_mode().as_pytype():
        result = False
        log.debug('  Read modes differ')

    if ad1.well_depth_setting().as_pytype() != \
       ad2.well_depth_setting().as_pytype():
        result = False
        log.debug('  Well depths differ')

    if ad1.gain_setting().as_pytype() != ad2.gain_setting().as_pytype():
        result = False
        log.debug('  Gain settings differ')

    if ad1.detector_x_bin() != ad2.detector_x_bin() or \
       ad1.detector_y_bin() != ad2.detector_y_bin():
        result = False
        log.debug('  Detector binnings differ')

    # It may or may not be critical for coadds to match, depending on
    # whether they are added or averaged for the instrument concerned,
    # but it's safest to require that they do:
    if ad1.coadds().as_pytype() != ad2.coadds().as_pytype():
        result = False
        log.debug('  Number of co-adds differ')

    if check_exposure:
        try:
            # This is the tolerance Paul used for NIRI in FITS store:
            if abs(ad1.exposure_time() - ad2.exposure_time()) > 0.01:
                result = False
                log.debug('  Exposure times differ')
        except (TypeError, DescriptorValueTypeError):
            log.error('Non-numeric type from exposure_time() descriptor')

    if ad1.camera().as_pytype() != ad2.camera().as_pytype():
        result = False
        log.debug('  Cameras differ')

    # This apparently works for GMOS so don't bother with grating() and
    # prism() as well. This may need checking for GNIRS XD but I would expect
    # it to incorporate the cross-disperser info. in that case.
    if ad1.disperser().as_pytype() != ad2.disperser().as_pytype():
        result = False
        log.debug('  Dispersers differ')

    # This is the tolerance Paul used for NIRI & F2 in FITS store:
    try:
        same = abs(ad1.central_wavelength() - ad2.central_wavelength()) < 0.001
    except (TypeError, DescriptorValueTypeError):
        same = ad1.central_wavelength().as_pytype() == \
               ad2.central_wavelength().as_pytype()
    if not same:
        result = False
        log.debug('  Central wavelengths differ')

    if ad1.focal_plane_mask().as_pytype() != \
       ad2.focal_plane_mask().as_pytype():
        result = False
        log.debug('  Focal plane masks differ')

    if ad1.filter_name().as_pytype() != ad2.filter_name().as_pytype():
        result = False
        log.debug('  Filters differ')
    
    if ad1.lyot_stop().as_pytype() != ad2.lyot_stop().as_pytype():
        result = False
        log.debug('  Lyot stop settings differ')

    if ad1.decker().as_pytype() != ad2.decker().as_pytype():
        result = False
        log.debug('  Decker settings differ')

    if ad1.pupil_mask().as_pytype() != ad2.pupil_mask().as_pytype():
        result = False
        log.debug('  Pupil mask settings differ')

    # Should beam splitter be added? I can't see a descriptor for it.

    if result:
        log.debug('  Configurations match')
    
    return result


def clip_auxiliary_data(adinput=None, aux=None, aux_type=None, 
                        return_dtype=None, keyword_comments=None):
    """
    This function clips auxiliary data like calibration files or BPMs
    to the size of the data section in the science. It will pad auxiliary
    data if required to match un-overscan-trimmed data, but otherwise
    requires that the auxiliary data contain the science data.
    
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)

    # ensure caller passes the sextractor default dictionary of parameters.
    try:
        assert isinstance(keyword_comments, dict)
    except AssertionError:
        log.error("TypeError: keyword comments dict was not received.")
        raise TypeError("keyword comments dict required")

    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinput)
    aux_list = validate_input(input=aux)
    
    # Create a dictionary that has the AstroData objects specified by adinput
    # as the key and the AstroData objects specified by aux as the value
    aux_dict = make_dict(key_list=adinput_list, value_list=aux_list)
    
    # Initialize the list of output AstroData objects
    aux_output_list = []
    
    try:
        # Check aux_type parameter for valid value
        if aux_type is None:
            raise Errors.InputError("The aux_type parameter must not be None")
        
        # If dealing with BPMs, relevant extensions are DQ; otherwise use SCI
        aux_type = aux_type.lower()
        if aux_type == "bpm":
            extname = DQ
        else:
            extname = SCI
        
        # Loop over each input AstroData object in the input list
        for ad in adinput_list:
            
            # Get the associated auxiliary file
            this_aux = aux_dict[ad]
            
            # Make a new blank auxiliary file for appending to
            new_aux = AstroData()
            new_aux.filename = this_aux.filename
            new_aux.phu = this_aux.phu
            
            # Get the detector section, data section, array section and the
            # binning of the x-axis and y-axis values for the science AstroData
            # object using the appropriate descriptors
            science_detector_section_dv = ad.detector_section()
            science_data_section_dv = ad.data_section()
            science_array_section_dv = ad.array_section()
            science_detector_x_bin_dv = ad.detector_x_bin()
            science_detector_y_bin_dv = ad.detector_y_bin()

            used_descriptors = [science_detector_section_dv,
                                science_data_section_dv,
                                science_array_section_dv,
                                science_detector_x_bin_dv,
                                science_detector_y_bin_dv]

            if any([used_descriptor.is_none() for used_descriptor in
                    used_descriptors]):
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                if hasattr(ad, "exception_info"):
                    raise ad.exception_info
            
            # Get the associated keyword for the detector section, data
            # section and array section from the DescriptorValue (DV) object
            detector_section_keyword = science_detector_section_dv.keyword
            data_section_keyword     = science_data_section_dv.keyword
            array_section_keyword    = science_array_section_dv.keyword
            
            # Get the detector section, data section and array section values
            # for the auxiliary AstroData object using the appropriate
            # descriptors
            log.fullinfo(" bpm BPM extname %s" %(extname))
            log.fullinfo(" bpm BPM this aux %s" %(this_aux))
            aux_detector_section_dv = this_aux[extname].detector_section()
            aux_data_section_dv     = this_aux[extname].data_section()
            aux_array_section_dv    = this_aux[extname].array_section()
            
            for sciext in ad[SCI]:
                # Retrieve the extension number for this extension
                science_extver = sciext.extver()
                
                # Get the section information for this extension from the DV
                science_detector_section = (
                  science_detector_section_dv.get_value(extver=science_extver))
                science_data_section = (
                  science_data_section_dv.get_value(extver=science_extver))
                science_array_section = (
                  science_array_section_dv.get_value(extver=science_extver))
                
                # Array section is unbinned; to use as indices for
                # extracting data, need to divide by the binning
                science_detector_x_bin = (
                  science_detector_x_bin_dv.get_value(extver=science_extver))
                science_detector_y_bin = (
                  science_detector_y_bin_dv.get_value(extver=science_extver))
                
                science_array_section = [
                  science_array_section[0] / science_detector_x_bin,
                  science_array_section[1] / science_detector_x_bin,
                  science_array_section[2] / science_detector_y_bin,
                  science_array_section[3] / science_detector_y_bin]
                
                # Check whether science data has been overscan-trimmed
                science_shape = sciext.data.shape[-2:]
                if (science_shape[1] == science_data_section[1] and
                    science_shape[0] == science_data_section[3] and
                    science_data_section[0] == 0 and
                    science_data_section[2] == 0):
                    
                    science_trimmed = True
                    science_offsets = [0,0,0,0]
                else:
                    science_trimmed = False
                    
                    # Offsets give overscan regions on either side of data:
                    # [left offset, right offset, bottom offset, top offset]
                    science_offsets = [
                      science_data_section[0],
                      science_shape[1] - science_data_section[1],
                      science_data_section[2],
                      science_shape[0] - science_data_section[3]]
                
                found = False
                for auxext in this_aux[extname]:
                    
                    # Retrieve the extension number for this extension
                    aux_extver = auxext.extver()
                    
                    # Get the section information for this extension from the
                    # DV 
                    aux_detector_section = (
                      aux_detector_section_dv.get_value(extver=aux_extver))
                    aux_data_section = (
                      aux_data_section_dv.get_value(extver=aux_extver))
                    aux_array_section = (
                      aux_array_section_dv.get_value(extver=aux_extver))
                    
                    # Array section is unbinned; to use as indices for
                    # extracting data, need to divide by the binning
                    aux_array_section = [
                      aux_array_section[0] / science_detector_x_bin,
                      aux_array_section[1] / science_detector_x_bin,
                      aux_array_section[2] / science_detector_y_bin,
                      aux_array_section[3] / science_detector_y_bin]
                    
                    # Check whether auxiliary detector section contains
                    # science detector section
                    if (aux_detector_section[0] <=
                        science_detector_section[0] and # x lower
                        aux_detector_section[1] >=
                        science_detector_section[1] and # x upper
                        aux_detector_section[2] <=
                        science_detector_section[2] and # y lower
                        aux_detector_section[3] >=
                        science_detector_section[3]):   # y upper
                        
                        # Auxiliary data contains or is equal to science data
                        found = True
                    else:
                        continue
                    
                    # Check whether auxiliary data has been overscan-trimmed
                    aux_shape = auxext.data.shape
                    if (aux_shape[1] == aux_data_section[1] and 
                        aux_shape[0] == aux_data_section[3] and
                        aux_data_section[0] == 0 and
                        aux_data_section[2] == 0):
                        
                        aux_trimmed = True
                        aux_offsets = [0,0,0,0]
                    else:
                        aux_trimmed = False
                        
                        # Offsets give overscan regions on either side of data:
                        # [left offset, right offset, bottom offset, top
                        # offset]
                        aux_offsets = [aux_data_section[0],
                                       aux_shape[1] - aux_data_section[1],
                                       aux_data_section[2],
                                       aux_shape[0] - aux_data_section[3]]
                    
                    # Define data extraction region corresponding to science
                    # data section (not including overscan)
                    x_translation = (
                      science_array_section[0] - science_data_section[0] -
                      aux_array_section[0] + aux_data_section[0])
                    y_translation = (
                      science_array_section[2] - science_data_section[2] -
                      aux_array_section[2] + aux_data_section[2])
                    region = [science_data_section[2] + y_translation,
                              science_data_section[3] + y_translation,
                              science_data_section[0] + x_translation,
                              science_data_section[1] + x_translation]
                    
                    # Deepcopy auxiliary SCI plane
                    # and auxiliary VAR/DQ planes if they exist
                    # (in the non-BPM case)
                    # This must be done here so that the same
                    # auxiliary extension can be used for a
                    # different science extension; without the
                    # deepcopy, the original auxiliary extension
                    # gets clipped
                    ext_to_clip = [deepcopy(auxext)]
                    if aux_type != "bpm":
                        varext = this_aux[VAR,aux_extver]
                        if varext is not None:
                            ext_to_clip.append(deepcopy(varext))
                            
                        dqext = this_aux[DQ,aux_extver]
                        if dqext is not None:
                            ext_to_clip.append(deepcopy(dqext))
                    
                    # Clip all relevant extensions
                    for ext in ext_to_clip:

                        # Pull out specified data region:
                        if science_trimmed or aux_trimmed:
                            clipped = ext.data[region[0]:region[1],
                                               region[2]:region[3]]

                        # Where no overscan is needed, just use the data region:
                        if science_trimmed:
                            ext.data = clipped

                        # Pad trimmed aux arrays with zeros to match untrimmed
                        # science data:
                        elif aux_trimmed:

                            # Science decision: trimmed calibrations can't be
                            # meaningfully matched to untrimmed science data
                            if aux_type != 'bpm':
                                raise Errors.ScienceError(
                                    "Auxiliary data %s is trimmed, but "
                                    "science data %s is untrimmed." %
                                    (auxext.filename, sciext.filename))

                            # Use duplicate iterators over the reversed
                            # science_offsets list to unpack its values in
                            # pairs and reverse them:
                            padding = tuple((bef, aft) for aft, bef in \
                                            zip(*[reversed(science_offsets)]*2))

                            # Replace the array with one that's padded with the
                            # appropriate number of zeros at each edge:
                            ext.data = np.pad(clipped, padding, 'constant',
                                              constant_values=0)

                        # If nothing is trimmed, just use the unmodified data
                        # after checking that the regions match (a condition
                        # preserved from r5564 without revisiting its logic):
                        elif not all(off1 == off2 for off1, off2 in \
                                     zip(aux_offsets, science_offsets)):
                            raise Errors.ScienceError(
                                "Overscan regions do not match in %s, %s" % \
                                (auxext.filename, sciext.filename))

                        # Convert the dtype if requested
                        if return_dtype is not None:
                            new_data = ext.data.astype(return_dtype)
                            ext.data = new_data
                            del new_data
                            
                        # Set the section keywords as appropriate
                        data_section_value = sciext.get_key_value(
                          data_section_keyword)
                        if data_section_value is not None:
                            ext.set_key_value(
                              data_section_keyword,
                              sciext.header[data_section_keyword],
                              keyword_comments[data_section_keyword])
                        
                        detector_section_value = sciext.get_key_value(
                          detector_section_keyword)
                        if detector_section_value is not None:
                            ext.set_key_value(
                              detector_section_keyword,
                              sciext.header[detector_section_keyword],
                              keyword_comments[detector_section_keyword])
                        
                        array_section_value = sciext.get_key_value(
                          array_section_keyword)
                        if array_section_value is not None:
                            ext.set_key_value(
                              array_section_keyword,
                              sciext.header[array_section_keyword],
                              keyword_comments[array_section_keyword])
                        
                        # Rename the auxext to the science extver
                        ext.rename_ext(name=ext.extname(),ver=science_extver)
                        new_aux.append(ext)
                
                if not found:
                    raise Errors.ScienceError(
                      "No auxiliary data in %s matches the detector section "
                      "%s in %s[%s,%d]" % (this_aux.filename,
                        science_detector_section_dv.get_value(science_extver),
                                           ad.filename, SCI, sciext.extver()))
            
            new_aux.refresh_types()
            log.stdinfo("Clipping {} to match science data.".
                        format(os.path.basename(new_aux.filename)))
            aux_output_list.append(new_aux)
        
        return aux_output_list
    
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def clip_auxiliary_data_GSAOI(adinput=None, aux=None, aux_type=None, 
                        return_dtype=None, keyword_comments=None):
    """
    This function clips auxiliary data like calibration files or BPMs
    to the size of the data section in the science. It will pad auxiliary
    data if required to match un-overscan-trimmed data, but otherwise
    requires that the auxiliary data contain the science data.
    
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)

    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinput)
    aux_list = validate_input(input=aux)
    
    # Create a dictionary that has the AstroData objects specified by adinput
    # as the key and the AstroData objects specified by aux as the value
    aux_dict = make_dict(key_list=adinput_list, value_list=aux_list)
    
    # Initialize the list of output AstroData objects
    aux_output_list = []
    
    try:
        # Check aux_type parameter for valid value
        if aux_type is None:
            raise Errors.InputError("The aux_type parameter must not be None")
        
        # If dealing with BPMs, relevant extensions are DQ; otherwise use SCI
        aux_type = aux_type.lower()
        if aux_type == "bpm":
            extname = DQ
        else:
            extname = SCI
        
        # Loop over each input AstroData object in the input list
        for ad in adinput_list:
            science_detector_section_dv = ad.detector_section()
            science_data_section_dv = ad.data_section()
            science_array_section_dv = ad.array_section()
            
            # Get the associated auxiliary file
            this_aux = aux_dict[ad]
            
            # Make a new blank auxiliary file for appending to
            new_aux = AstroData()
            new_aux.filename = this_aux.filename
            new_aux.phu = this_aux.phu
                       
            # Get the associated keyword for the detector section, data
            # section and array section from the DescriptorValue (DV) object
            detector_section_keyword = science_detector_section_dv.keyword
            data_section_keyword     = science_data_section_dv.keyword
            array_section_keyword    = science_array_section_dv.keyword

            aux_data_section_dv     = this_aux[extname].data_section()
            aux_array_section_dv    = this_aux[extname].array_section()

            for sciext in ad[SCI]:
                # Retrieve the extension number for this extension
                science_extver = sciext.extver()
                science_frameid = sciext.get_key_value("FRAMEID")
                
                science_data_section = (
                  science_data_section_dv.get_value(extver=science_extver))
                science_array_section = (
                  science_array_section_dv.get_value(extver=science_extver))
                
                # Check whether science data has been overscan-trimmed
                science_shape = sciext.data.shape
                if (science_shape[1] == science_data_section[1] and
                    science_shape[0] == science_data_section[3] and
                    science_data_section[0] == 0 and
                    science_data_section[2] == 0):
                    
                    science_trimmed = True
                    science_offsets = [0,0,0,0]
                else:
                    science_trimmed = False
                    # Offsets give overscan regions on either side of data:
                    # [left offset, right offset, bottom offset, top offset]
                    science_offsets = [
                      science_data_section[0],
                      science_shape[1] - science_data_section[1],
                      science_data_section[2],
                      science_shape[0] - science_data_section[3]]
                
                found = False
                for auxext in this_aux[extname]:
                    # Retrieve the extension number for this extension
                    aux_extver = auxext.extver()
                    aux_frameid = auxext.get_key_value("FRAMEID")
                    aux_shape = auxext.data.shape
                    
                    if (aux_frameid == science_frameid and
                        aux_shape[0] >= science_shape[0] and
                        aux_shape[1] >= science_shape[1]):
                    
                        # Auxiliary data is big enough as has right FRAMEID
                        found = True
                    else:
                        continue
                    
                    aux_data_section = (
                      aux_data_section_dv.get_value(extver=aux_extver))
                    aux_array_section = (
                      aux_array_section_dv.get_value(extver=aux_extver))

                    # Check whether auxiliary data has been overscan-trimmed
                    if (aux_shape[1] == aux_data_section[1] and 
                        aux_shape[0] == aux_data_section[3] and
                        aux_data_section[0] == 0 and
                        aux_data_section[2] == 0):
                        
                        aux_trimmed = True
                        aux_offsets = [0,0,0,0]
                    else:
                        aux_trimmed = False
                        
                        # Offsets give overscan regions on either side of data:
                        # [left offset, right offset, bottom offset, top
                        # offset]
                        aux_offsets = [aux_data_section[0],
                                       aux_shape[1] - aux_data_section[1],
                                       aux_data_section[2],
                                       aux_shape[0] - aux_data_section[3]]
                    
                    # Define data extraction region corresponding to science
                    # data section (not including overscan)
                    x_translation = (
                      science_array_section[0] - science_data_section[0] -
                      aux_array_section[0] + aux_data_section[0])
                    y_translation = (
                      science_array_section[2] - science_data_section[2] -
                      aux_array_section[2] + aux_data_section[2])
                    region = [science_data_section[2] + y_translation,
                              science_data_section[3] + y_translation,
                              science_data_section[0] + x_translation,
                              science_data_section[1] + x_translation]
                    
                    # Deepcopy auxiliary SCI plane
                    # and auxiliary VAR/DQ planes if they exist
                    # (in the non-BPM case)
                    # This must be done here so that the same
                    # auxiliary extension can be used for a
                    # different science extension; without the
                    # deepcopy, the original auxiliary extension
                    # gets clipped
                    ext_to_clip = [deepcopy(auxext)]
                    if aux_type != "bpm":
                        varext = this_aux[VAR,aux_extver]
                        if varext is not None:
                            ext_to_clip.append(deepcopy(varext))
                            
                        dqext = this_aux[DQ,aux_extver]
                        if dqext is not None:
                            ext_to_clip.append(deepcopy(dqext))
                    
                    # Clip all relevant extensions
                    for ext in ext_to_clip:

                        # Pull out specified data region:
                        if science_trimmed or aux_trimmed:
                            clipped = ext.data[region[0]:region[1],
                                               region[2]:region[3]]

                        # Where no overscan is needed, just use the data region:
                        if science_trimmed:
                            ext.data = clipped

                        # Pad trimmed aux arrays with zeros to match untrimmed
                        # science data:
                        # CJS: left over from standard clip_auxiliary_data()
                        # even though not used by GSAOI
                        elif aux_trimmed:

                            # Science decision: trimmed calibrations can't be
                            # meaningfully matched to untrimmed science data
                            if aux_type != 'bpm':
                                raise Errors.ScienceError(
                                    "Auxiliary data %s is trimmed, but "
                                    "science data %s is untrimmed." %
                                    (auxext.filename, sciext.filename))

                            # Use duplicate iterators over the reversed
                            # science_offsets list to unpack its values in
                            # pairs and reverse them:
                            padding = tuple((bef, aft) for aft, bef in \
                                            zip(*[reversed(science_offsets)]*2))

                            # Replace the array with one that's padded with the
                            # appropriate number of zeros at each edge:
                            # CJS: pad with ones if it's a BPM
                            ext.data = np.pad(clipped, padding, 'constant',
                                    constant_values=1 if aux_type=="bpm" else 0)

                        # If nothing is trimmed, just use the unmodified data
                        # after checking that the regions match (a condition
                        # preserved from r5564 without revisiting its logic):
                        elif not all(off1 == off2 for off1, off2 in \
                                     zip(aux_offsets, science_offsets)):
                            raise Errors.ScienceError(
                                "Overscan regions do not match in %s, %s" % \
                                (auxext.filename, sciext.filename))

                        # Convert the dtype if requested
                        if return_dtype is not None:
                            new_data = ext.data.astype(return_dtype)
                            ext.data = new_data
                            del new_data
                            
                        # Set the section keywords as appropriate
                        data_section_value = sciext.get_key_value(
                          data_section_keyword)
                        if data_section_value is not None:
                            ext.set_key_value(
                              data_section_keyword,
                              sciext.header[data_section_keyword],
                              keyword_comments[data_section_keyword])
                        
                        detector_section_value = sciext.get_key_value(
                          detector_section_keyword)
                        if detector_section_value is not None:
                            ext.set_key_value(
                              detector_section_keyword,
                              sciext.header[detector_section_keyword],
                              keyword_comments[detector_section_keyword])
                        
                        array_section_value = sciext.get_key_value(
                          array_section_keyword)
                        if array_section_value is not None:
                            ext.set_key_value(
                              array_section_keyword,
                              sciext.header[array_section_keyword],
                              keyword_comments[array_section_keyword])
                        
                        # Rename the auxext to the science extver
                        ext.rename_ext(name=ext.extname(),ver=science_extver)
                        new_aux.append(ext)
                
                if not found:
                    raise Errors.ScienceError(
                      "No auxiliary data in %s matches the detector section "
                      "%s in %s[%s,%d]" % (this_aux.filename,
                        science_detector_section_dv.get_value(science_extver),
                                           ad.filename, SCI, sciext.extver()))
            
            new_aux.refresh_types()
            log.stdinfo("Clipping {} to match science data.".
                        format(os.path.basename(new_aux.filename)))
            aux_output_list.append(new_aux)
        
        return aux_output_list
    
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def clip_sources(ad):
    """
    This function takes the source data from the OBJCAT and returns the best
    sources for IQ measurement.
    
    :param ad: input image
    :type ad: AstroData instance with OBJCAT attached
    """

    good_source = {}
    for sciext in ad[SCI]:
        extver = sciext.extver()

        objcat = ad["OBJCAT",extver]
        if objcat is None:
            continue
        if objcat.data is None:
            continue

        x = objcat.data.field("X_IMAGE")
        y = objcat.data.field("Y_IMAGE")

        fwhm_pix = objcat.data.field("PROFILE_FWHM")
        fwhm_arcsec = fwhm_pix * sciext.pixel_scale().as_pytype()

        isofwhm_pix = objcat.data.field("FWHM_IMAGE")
        isofwhm_arcsec = objcat.data.field("FWHM_WORLD")

        ee50d_pix = objcat.data.field("PROFILE_EE50")
        ee50d_arcsec = ee50d_pix * sciext.pixel_scale().as_pytype()

        ellip = objcat.data.field("ELLIPTICITY")
        pa = objcat.data.field("THETA_WORLD")

        sxflag = objcat.data.field("FLAGS")
        dqflag = objcat.data.field("IMAFLAGS_ISO")
        ndqflag = objcat.data.field("NIMAFLAGS_ISO")
        class_star = objcat.data.field("CLASS_STAR")
        area = objcat.data.field("ISOAREA_IMAGE")
        flux = objcat.data.field("FLUX_AUTO")
        fluxerr = objcat.data.field("FLUXERR_AUTO")
        flux_max = objcat.data.field("FLUX_MAX")
        flux_radius = objcat.data.field("FLUX_radius")
        semiminor_axis = objcat.data.field("B_IMAGE")
        background = objcat.data.field("BACKGROUND")

        # CJS: This should all be tidied up with a
        # good = np.logical_and.reduce() statement
        # Source is good if fwhm is defined
        fwflag = np.where(fwhm_pix==-999,1,0)

        # Source is good if ellipticity defined and <0.5
        eflag = np.where((ellip>0.5)|(ellip==-999),1,0)

        # Stellarity test
        is_ao = ad.is_ao().as_pytype()
        if is_ao:
            sflag = np.where(np.fabs(isofwhm_pix/1.08-fwhm_pix) >
                             0.2*isofwhm_pix,1,0)
        else:
            sflag = np.where(class_star<0.9,1,0)

        # Source is good if semi-minor axis is greater than 1.1 pixels
        # (less indicates a cosmic ray)
        smflag = np.where(semiminor_axis<1.1,1,0)

        flags = sxflag | fwflag | eflag | sflag | smflag

        # Source is good if greater than 20 connected pixels
        # Ignore criterion if all undefined (-999)
        if not np.all(area==-999):
            aflag = np.where(area<20,1,0)
            flags |= aflag

        # Source is good if signal to noise ratio > 50
        if not np.all(fluxerr==-999):
            snlimit = 25 if is_ao else 50
            snflag = np.where(flux < snlimit*fluxerr, 1, 0)
            flags |= snflag

        # Allow up to 2% BPM/non-linear bad pixels but no saturated pixels
        max_bad_pix = np.where(dqflag & 4, 0, 0.02*area)

        # Ignore source that hae too many flagged pixels in them
        if not np.all(ndqflag==-999):
            toobadflags = np.where(ndqflag > max_bad_pix, 1, 0)
            flags |= toobadflags
            
        flags |= np.where(ee50d_pix<fwhm_pix, 1, 0)

        # Use flags=0 to find good data
        good = (flags==0)

        rec = np.rec.fromarrays(
                [x[good], y[good], fwhm_pix[good],fwhm_arcsec[good],
                 isofwhm_pix[good], isofwhm_arcsec[good], ee50d_pix[good],
                 ee50d_arcsec[good], ellip[good], pa[good], flux[good],
                 flux_max[good], background[good], flux_radius[good]],
                names=["x", "y", "fwhm", "fwhm_arcsec",
                       "isofwhm", "isofwhm_arcsec", "ee50d",
                       "ee50d_arcsec", "ellipticity", "pa", "flux",
                       "flux_max", "background", "flux_radius"])

        # Clip outliers in FWHM - single 1-sigma clip if more than 3 sources.
        num_total = len(rec)
        if num_total>=3:

            data = rec["fwhm_arcsec"]
            mean = data.mean()
            sigma = data.std()
            rec = rec[(data<mean+sigma) & (data>mean-sigma)]

        # Store data
        good_source[(SCI,extver)] = rec

    return good_source


def convert_to_cal_header(adinput=None, caltype=None, keyword_comments=None):
    """
    This function replaces position, object, and program information 
    in the headers of processed calibration files that are generated
    from science frames, eg. fringe frames, maybe sky frames too.
    It is called, for example, from the storeProcessedFringe primitive.

    :param adinput: astrodata instance to perform header key updates on
    :type adinput: an AstroData instance

    :param caltype: type of calibration.  Accepted values are 'fringe',
                    'sky', or 'flat'
    :type caltype: string
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)

    try:
        assert isinstance(keyword_comments, dict)
    except AssertionError:
        log.error("TypeError: keyword comments dict was not received.")
        raise TypeError("keyword comments dict required")
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinput)
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:

        if caltype is None:
            raise Errors.InputError("Caltype should not be None")

        fitsfilenamecre = re.compile("^([NS])(20\d\d)([01]\d[0123]\d)(S)"\
                                     "(?P<fileno>\d\d\d\d)(.*)$")

        for ad in adinput_list:

            log.fullinfo("Setting OBSCLASS, OBSTYPE, GEMPRGID, OBSID, " +
                         "DATALAB, RELEASE, OBJECT, RA, DEC, CRVAL1, " +
                         "and CRVAL2 to generic defaults")

            # Do some date manipulation to get release date and 
            # fake program number

            # Get date from day data was taken if possible
            date_taken = ad.ut_date()
            if date_taken.collapse_value() is None:
                # Otherwise use current time
                import datetime
                date_taken = datetime.date.today()
            else:
                date_taken = date_taken.as_pytype()
            site = str(ad.telescope()).lower()
            release = date_taken.strftime("%Y-%m-%d")

            # Fake ID is G(N/S)-CALYYYYMMDD-900-fileno
            if "north" in site:
                prefix = "GN-CAL"
            elif "south" in site:
                prefix = "GS-CAL"
            prgid = "%s%s" % (prefix,date_taken.strftime("%Y%m%d"))
            obsid = "%s-%d" % (prgid, 900)

            m = fitsfilenamecre.match(ad.filename)
            if m:
                fileno = m.group("fileno")
                try:
                    fileno = int(fileno)
                except:
                    fileno = None
            else:
                fileno = None

            # Use a random number if the file doesn't have a
            # Gemini filename
            if fileno is None:
                import random
                fileno = random.randint(1,999)
            datalabel = "%s-%03d" % (obsid,fileno)

            # Set class, type, object to generic defaults
            ad.phu_set_key_value("OBSCLASS", "partnerCal",
                                 keyword_comments["OBSCLASS"])

            if "fringe" in caltype:
                ad.phu_set_key_value("OBSTYPE", "FRINGE",
                                     keyword_comments["OBSTYPE"])
                ad.phu_set_key_value("OBJECT", "Fringe Frame",
                                     keyword_comments["OBJECT"])
            elif "sky" in caltype:
                ad.phu_set_key_value("OBSTYPE", "SKY",
                                     keyword_comments["OBSTYPE"])
                ad.phu_set_key_value("OBJECT", "Sky Frame",
                                     keyword_comments["OBJECT"])
            elif "flat" in caltype:
                ad.phu_set_key_value("OBSTYPE", "FLAT",
                                     keyword_comments["OBSTYPE"])
                ad.phu_set_key_value("OBJECT", "Flat Frame",
                                     keyword_comments["OBJECT"])
            else:
                raise Errors.InputError("Caltype %s not supported" % caltype)
            
            # Blank out program information
            ad.phu_set_key_value("GEMPRGID", prgid,
                                 keyword_comments["GEMPRGID"])
            ad.phu_set_key_value("OBSID", obsid,
                                 keyword_comments["OBSID"])
            ad.phu_set_key_value("DATALAB", datalabel,
                                 keyword_comments["DATALAB"])

            # Set release date
            ad.phu_set_key_value("RELEASE", release,
                                 keyword_comments["RELEASE"])

            # Blank out positional information
            ad.phu_set_key_value("RA", 0.0, keyword_comments["RA"])
            ad.phu_set_key_value("DEC", 0.0, keyword_comments["DEC"])
            
            # Blank out RA/Dec in WCS information in PHU if present
            if ad.phu_get_key_value("CRVAL1") is not None:
                ad.phu_set_key_value("CRVAL1", 0.0, keyword_comments["CRVAL1"])
            if ad.phu_get_key_value("CRVAL2") is not None:
                ad.phu_set_key_value("CRVAL2", 0.0, keyword_comments["CRVAL2"])

            # Do the same for each SCI,VAR,DQ extension
            # as well as the object name
            for ext in ad:
                if ext.extname() not in [SCI, VAR, DQ]:
                    continue
                if ext.get_key_value("CRVAL1") is not None:
                    ext.set_key_value("CRVAL1", 0.0, keyword_comments["CRVAL1"])
                if ext.get_key_value("CRVAL2") is not None:
                    ext.set_key_value("CRVAL2", 0.0, keyword_comments["CRVAL2"])
                if ext.get_key_value("OBJECT") is not None:
                    if "fringe" in caltype:
                        ext.set_key_value("OBJECT", "Fringe Frame",
                                          keyword_comments["OBJECT"])
                    elif "sky" in caltype:
                        ext.set_key_value("OBJECT", "Sky Frame",
                                          keyword_comments["OBJECT"])
                    elif "flat" in caltype:
                        ext.set_key_value("OBJECT", "Flat Frame",
                                          keyword_comments["OBJECT"])

            adoutput_list.append(ad)

        return adoutput_list    

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def filename_updater(adinput=None, infilename='', suffix='', prefix='',
                    strip=False):
    """
    This function is for updating the file names of astrodata objects.
    It can be used in a few different ways.  For simple post/pre pending of
    the infilename string, there is no need to define adinput or strip. The 
    current filename for adinput will be used if infilename is not defined. 
    The examples below should make the main uses clear.
        
    Note: 
    1.if the input filename has a path, the returned value will have
    path stripped off of it.
    2. if strip is set to True, then adinput must be defined.
          
    :param adinput: input astrodata instance having its filename being updated
    :type adinput: astrodata object
    
    :param infilename: filename to be updated
    :type infilename: string
    
    :param suffix: string to put between end of current filename and the 
                   extension 
    :type suffix: string
    
    :param prefix: string to put at the beginning of a filename
    :type prefix: string
    
    :param strip: Boolean to signal that the original filename of the astrodata
                  object prior to processing should be used. adinput MUST be 
                  defined for this to work.
    :type strip: Boolean
    
    ::
    
     filename_updater(adinput=myAstrodataObject, suffix='_prepared', strip=True)
     result: 'N20020214S022_prepared.fits'
        
     filename_updater(infilename='N20020214S022_prepared.fits',
         suffix='_biasCorrected')
     result: 'N20020214S022_prepared_biasCorrected.fits'
        
     filename_updater(adinput=myAstrodataObject, prefix='testversion_')
     result: 'testversion_N20020214S022.fits'
    
    """
    log = logutils.get_logger(__name__) 

    # Check there is a name to update
    if infilename=='':
        # if both infilename and adinput are not passed in, then log critical msg
        if adinput==None:
            log.critical('A filename or an astrodata object must be passed '+
                         'into filename_updater, so it has a name to update')
        # adinput was passed in, so set infilename to that ad's filename
        else:
            infilename = adinput.filename
            
    # Strip off any path that the input file name might have
    basefilename = os.path.basename(infilename)

    # Split up the filename and the file type ie. the extension
    (name,filetype) = os.path.splitext(basefilename)
    
    if strip:
        # Grabbing the value of PHU key 'ORIGNAME'
        phuOrigFilename = adinput.phu_get_key_value('ORIGNAME') 
        # If key was 'None', ie. store_original_name() wasn't ran yet, then run
        # it now
        if phuOrigFilename is None:
            # Storing the original name of this astrodata object in the PHU
            phuOrigFilename = adinput.store_original_name()
            
        # Split up the filename and the file type ie. the extension
        (name,filetype) = os.path.splitext(phuOrigFilename)
        
    # Create output filename
   
    outFileName = prefix+name+suffix+filetype
    return outFileName


def finalise_adinput(adinput=None, timestamp_key=None, suffix=None,
                     allow_empty=False):

    if adinput is None or (not allow_empty and not adinput):
        raise Errors.InputError()
    elif not isinstance(adinput, list):
        adinput_list = [adinput]
    else:
        adinput_list = adinput
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    # Loop over each input AstroData object in the list
    for ad in adinput_list:
        
        # Add the appropriate time stamps to the PHU
        if timestamp_key is not None:
            mark_history(adinput=ad, keyword=timestamp_key)
        
        # Update the filename
        if suffix is not None:
            ad.filename = filename_updater(adinput=ad, suffix=suffix,
                                           strip=True) 
        # Append the output AstroData object to the list of output AstroData
        # objects 
        adoutput_list.append(ad)
    return adoutput_list


def fit_continuum(ad):
    """
    This function fits Gaussians to the spectral continuum centered around 
    the row with the greatest total flux.
    
    :param ad: input image
    :type ad: AstroData instance
    """
    log = logutils.get_logger(__name__)
    import warnings

    good_source = {}
    
    # Get the pixel scale
    pixel_scale = ad.pixel_scale().as_pytype()
    
    # Set full aperture to 4 arcsec
    ybox = int(2.0 / pixel_scale)

    # Average 512 unbinned columns together
    xbox = 256 / int(ad.detector_x_bin())

    # Average 16 unbinned background rows together
    bgbox = 8 / int(ad.detector_x_bin())

    # Initialize the Gaussian width to FWHM = 1.2 arcsec
    init_width = 1.2 / (pixel_scale * (2 * np.sqrt(2 * np.log(2))))

    # Ignore spectrum if not >1.5*background
    s2n_bg = 1.5

    # Ignore spectrum if mean not >.9*std
    s2n_self = 0.9

    for sciext in ad[SCI]:
        extver = sciext.extver()
        data = sciext.data

        if ad[DQ,extver] is not None:
            dqdata = ad[DQ,extver].data
        else:
            dqdata = None

        if ad[VAR,extver] is not None:
            vardata = ad[VAR,extver].data
        else:
            vardata = None

        ####here - dispersion axis
        # Taking the 95th percentile should remove CRs
        signal = np.percentile(np.where(dqdata==0, data, 0), 95, axis=1)
        acq_star_positions = ad.phu_get_key_value("ACQSLITS")
        if acq_star_positions is None:
            if 'MOS' in ad.types:
                log.warning("%s is MOS but has no acquisition slits. "
                            "Not trying to find spectra." % ad.filename)
                if ad['MDF'] is None:
                    log.warning("No MDF is attached. Did addMDF find one?\n") 
                continue

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
                log.warning("Image %s has unparseable ACQSLITS keyword"
                            % ad.filename)
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
                
                if 'GMOS_NODANDSHUFFLE' in ad.types:
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
                    if (g.amplitude_0>0.5*maxflux and
                        g.mean_0>1 and g.mean_0<len(col)-2):
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
        
        rec = np.rec.fromarrays([x_list,y_list,fwhm_pix,
                                 fwhm_arcsec,weight_list],
                    names=["x","y","fwhm","fwhm_arcsec","weight"])
        #plt.hist(rec["fwhm_arcsec"], 10)
        #plt.show()
        #print rec

        # Clip outliers in FWHM
        num_total = len(rec)
        if num_total>=3:
            data = rec["fwhm_arcsec"]
            #mean = data.mean()
            #sigma = data.std()
            mean, sigma = at.clipped_mean(data)
            rec = rec[(data<mean+2*sigma) & (data>mean-2*sigma)]
        # Store data
        
        good_source[(SCI,extver)] = rec
    return good_source


def fitsstore_report(ad, rc, metric, info_dict, calurl_dict):
    if metric not in ["iq","zp","sb","pe"]:
        raise Errors.InputError("Unknown metric %s" % metric )
    
    # Empty qareport dictionary to build into
    qareport = {}

    # Compose metadata
    import getpass
    import socket
    qareport["hostname"]   = socket.gethostname()
    qareport["userid"]     = getpass.getuser()
    qareport["processid"]  = os.getpid()
    qareport["executable"] = os.path.basename(sys.argv[0])

    # These may need revisiting.  There doesn't seem to be a
    # way to access a version name or number for the primitive
    # set generating this metric
    qareport["software"] = "QAP"
    qareport["software_version"] = ad_version
    qareport["context"] = rc.context
    
    qametric_list = []

    for sciext in ad[SCI]:
        key = ('SCI',sciext.extver())

        # Empty qametric dictionary to build into
        qametric = {}

        # Metadata for qametric
        qametric["filename"] = ad.filename
        try:
            qametric["datalabel"] = ad.data_label().as_pytype()
        except:
            qametric["datalabel"] = None
        try:
            qametric["detector"] = sciext.array_name().as_pytype()
        except:
            qametric["detector"] = None

        if metric=="iq":
            # Check to see if there are any data for this extension
            if info_dict.has_key(key):
                qametric["iq"] = info_dict[key]
                qametric_list.append(qametric)

        elif metric=="zp":
            # Check to see if there is any data for this extension
            if not info_dict.has_key(key):
                continue
            
            # Use the info_dict as the zp dict
            # Check the measureCC primitive to see if the values
            # compiled here are the right ones for fitsstore
            zp = info_dict[key]

            # Add catalog information
            # This is hard coded for now, as there does not seem
            # to be a way to look it up easily
            zp["photref"] = "SDSS8"

            qametric["zp"] = zp
            qametric_list.append(qametric)

        elif metric=="sb":
            # Check to see if there is any data for this extension
            if not info_dict.has_key(key):
                continue

            # Use the info_dict as the sb dict
            sb = info_dict[key]

            qametric["sb"] = sb
            qametric_list.append(qametric)

        elif metric=="pe":
            # Check to see if there is any data for this extension
            if not info_dict.has_key(key):
                continue

            # Use the info_dict as the pe dict
            pe = info_dict[key]

            # Add catalog information
            # This is hard coded for now, as there does not seem
            # to be a way to look it up easily
            pe["astref"] = "SDSS8"

            qametric["pe"] = pe
            qametric_list.append(qametric)


    # Add qametric dictionary into qareport
    qareport["qametric"] = qametric_list
    
    # rc returns a string, not a boolean when adcc makes the call to reduce
    # as triggered by the url /runreduce call.
    # I suspect that rc will return a boolean when reduce is called from the 
    # command line.
    if rc["upload_metrics"] == 'True' or rc["upload_metrics"] == True:
        send_fitsstore_report(qareport, calurl_dict)
    return qareport


def send_fitsstore_report(qareport, calurl_dict):
    qalist = [qareport]
    req = urllib2.Request(url=calurl_dict["QAMETRICURL"], data=json.dumps(qalist))
    f = urllib2.urlopen(req)
    # Should do some error checking here.
    f.close()
    return

def log_message(function=None, name=None, message_type=None):
    if message_type == 'calling':
        message = 'Calling the %s %s' % (function, name)
    if message_type == 'starting':
        message = 'Starting the %s %s' % (function, name)
    if message_type == 'finishing':
        message = 'Finishing the %s %s' % (function, name)
    if message_type == 'completed':
        message = 'The %s %s completed successfully' % (name, function)
    if message:
        return message
    else:
        return None

def make_dict(key_list=None, value_list=None):
    """
    The make_dict function creates a dictionary with the elements in 'key_list'
    as the key and the elements in 'value_list' as the value to create an
    association between the input science dataset (the 'key_list') and a, for
    example, dark that is needed to be subtracted from the input science
    dataset. This function also does some basic checks to ensure that the
    filters, exposure time etc are the same.

    :param key: List containing one or more AstroData objects
    :type key: AstroData

    :param value: List containing one or more AstroData objects
    :type value: AstroData
    """
    # Check the inputs have matching filters, binning and SCI shapes.
    ret_dict = {}
    if not isinstance(key_list, list):
        key_list = [key_list]
    if not isinstance(value_list, list):
        value_list = [value_list]
    if len(key_list) == 1 and len(value_list) == 1:
        # There is only one key and one value - create a single entry in the
        # dictionary
        ret_dict[key_list[0]] = value_list[0]
    elif len(key_list) > 1 and len(value_list) == 1:
        # There is only one value for the list of keys
        for i in range (0, len(key_list)):
            ret_dict[key_list[i]] = value_list[0]
    elif len(key_list) > 1 and len(value_list) > 1:
        # There is one value for each key. Check that the lists are the same
        # length
        if len(key_list) != len(value_list):
            msg = """Number of AstroData objects in key_list does not match
            with the number of AstroData objects in value_list. Please provide
            lists containing the same number of AstroData objects. Please
            supply either a single AstroData object in value_list to be applied
            to all AstroData objects in key_list OR the same number of
            AstroData objects in value_list as there are in key_list"""
            raise Errors.InputError(msg)
        for i in range (0, len(key_list)):
            ret_dict[key_list[i]] = value_list[i]
    
    return ret_dict

def mark_history(adinput=None, keyword=None, primname=None, comment=None):
    """
    Add or update a keyword with the UT time stamp as the value (in the form
    <YYYY>-<MM>-<DD>T<HH>:<MM>:<SS>) to the header of the PHU of the AstroData
    object to indicate when and what function was just performed on the
    AstroData object 
    
    :param adinput: The input AstroData object to add or update the time stamp
                    keyword
    :type adinput: AstroData or list of AstroData
    :param keyword: The keyword to add or update in the PHU in upper case. The
                    keyword should be less than or equal to 8 characters. If
                    keyword is None, only the 'GEM-TLM' keyword is added or
                    updated. 
    :type keyword: string
    :param comment: Comment for the time stamp keyword. If comment is None, the
                    primitive the keyword is associated with will be determined
                    from the timestamp_keywords.py module; a default comment of
                    'UT time stamp for <primitive>' will then be used. However,
                    if the timestamp_keywords.py module cannot be found, the
                    comment 'UT time stamp for <keyword>' will instead be used.
    :type comment: string
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)

    try:
        assert keyword
    except AssertionError:
        log.error("TypeError: A keyword was not received.")
        raise TypeError("argument 'keyword' required")

    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinput)
    
    # Get the current time to use for the time of last modification
    tlm = datetime.now().isoformat()[0:-7]
    
    # Construct the default comment
    if comment is None:
        if primname:
            comment_suffix = primname
        else:
            comment_suffix = keyword

        final_comment = "UT time stamp for %s" % comment_suffix
    else:
        final_comment = comment
    
    # The GEM-TLM keyword will always be added or updated
    keyword_dict = {"GEM-TLM":"UT last modification with GEMINI"}
    
    if keyword is not None:
        # Add or update the input keyword in addition to the GEM-TLM keyword
        keyword_dict.update({keyword:final_comment})
    
    # Loop over each input AstroData object in the input list
    for ad in adinput_list:
        for key, comm in keyword_dict.iteritems():
            update_key(adinput=ad, keyword=key, value=tlm, comment=comm,
                       extname="PHU")
    return

def measure_bg_from_objcat(ad, min_ok=5, value_only=False):
    """
    Return a dictionary, keyed by extver, of triples of background values,
    and their std deviations derived from the OBJCATs in ad, plus the number
    of objects used.
    If there are too few good BG measurements, then None is returned.
    If the input has SCI extensions, then the output dict contains one tuple
    per SCI extension, even if no OBJCAT is associated with that extension
    
    :param ad: an AstroData instance (NOT a list)
    :type ad: AstroData
    :param min_ok: minimum number of good values (after sigma_clipping)
                   or else None is returned
    :type min_ok: float
    :param value_only: return only the values, not stddevs or nsamples?
    :type value_only: bool
    """

    if ad['SCI'] is not None:
        # If there are SCI extensions, use the associated OBJCATs
        input_list = [ad['OBJCAT',ext.extver()] for ext in ad['SCI']]
    else:
        # Otherwise, use all the OBJCATs.
        # This is not the same, because a SCI extension might not have
        # an associated OBJCAT, but we want to return values for all SCIs
        input_list = [ext for ext in ad['OBJCAT']]

    output_dict = {}
    for objcat in input_list:
        bg = None
        bg_std = None
        nsamples = None
        if objcat is not None:
            bg_data = objcat.data['BACKGROUND']
            # Don't use objects with dodgy flags
            if len(bg_data)>0 and not np.all(bg_data==-999):
                flags = objcat.data['FLAGS']
                dqflags = objcat.data['IMAFLAGS_ISO']
                if not np.all(dqflags==-999):
                    # Non-linear/saturated pixels are OK
                    flags |= (dqflags & 65529)
                bg_data = bg_data[flags==0]
                # Sigma-clip, and only give results if enough objects are left
                if len(bg_data) > min_ok:
                    clipped_data = stats.sigma_clip(bg_data, 3.0)
                    if np.sum(~clipped_data.mask) > min_ok:
                        bg = np.mean(clipped_data)
                        bg_std = np.std(clipped_data)
                        nsamples = np.sum(~clipped_data.mask)
            if value_only:
                output_dict[objcat.extver()] = bg
            else:
                output_dict[objcat.extver()] = (bg, bg_std, nsamples)

    return output_dict

def measure_bg_from_image(ad, use_extver=None, sampling=10, value_only=False, gaussfit=True):
    """
    Return background value, and its std deviation
    as measured directly from pixels in the SCI image.
    DQ and OBJMASK planes are used (if they exist)
    If extver is set, return a double for that extension,
    otherwise return a dictionary of doubles, keyed by extver.
    
    :param ad: an AstroData instance (NOT a list)
    :type ad: AstroData
    :param extver: extension number to use
    :type extver: int (or None)
    :param sampling: 1-in-n sampling factor
    :type sampling: int
    :param value_only: return only the values, not stddevs?
    :type value_only: bool
    :param gaussfit: fit Gaussian to pixel values, instead of sigma-clipping?
    :type gaussfit: bool
    """
    
    if use_extver is None:
        input_list = [ext for ext in ad['SCI']]
    else:
        input_list = [ad['SCI',use_extver]]
    
    output_dict = {}
    for sciext in input_list:
        # This can only happen if use_extver is invalid
        if sciext is None:
            if value_only:
                return None
            else:
                return (None, None)

        extver = sciext.extver()
        # Use DQ and OBJMASK; don't create flags array if not needed
        dqext = ad['DQ',extver]
        maskext = ad['OBJMASK',extver]
        if dqext is not None:
            if maskext is not None:
                if maskext.data.shape == dqext.data.shape:
                    bg_data = sciext.data[dqext.data | maskext.data==0].flatten()
                else:
                    bg_data = sciext.data[dqext.data==0].flatten()
            else:
                bg_data = sciext.data[dqext.data==0].flatten()
        elif maskext is not None:
            if maskext.data.shape == sciext.data.shape:
                bg_data = sciext.data[maskext.data==0].flatten()
        else:
            bg_data = sciext.data.flatten()

        bg_data = bg_data[::sampling]
        if gaussfit:
            bg = np.median(bg_data)
            bg_std = np.std(bg_data)
            binsize = bg_std*0.1
            # Fit from -5 to +1 sigma
            bins = np.arange(bg-5*bg_std, bg+bg_std, binsize)
            histdata, _ = np.histogram(bg_data, bins)
            # bin centers
            x = bins[:-1] + 0.5*(bins[1]-bins[0])
            # Eliminate bins with no data (e.g., if data are quantized)
            x = x[histdata>0]
            histdata = histdata[histdata>0]
            g_init = models.Gaussian1D(amplitude=np.max(histdata),
                                       mean=bg, stddev=bg_std)
            fit_g = fitting.LevMarLSQFitter()
            g = fit_g(g_init, x, histdata)
            bg, bg_std = g.mean.value, abs(g.stddev.value)
        else:
            # Sigma-clipping will screw up the stats of course!
            clipped_data = stats.sigma_clip(bg_data, 2.0, iters=2)
            clipped_data = clipped_data.data[~clipped_data.mask]
            bg = np.median(clipped_data)
            bg_std = np.std(clipped_data)

        if value_only:
            output_dict[extver] = bg
        else:
            output_dict[extver] = (bg, bg_std)

    if use_extver is None:
        # This could be a dict of one tuple if only one SCI extn 
        return output_dict
    else:
        return output_dict[use_extver]

def obsmode_add(ad):
    """Add 'OBSMODE' keyword to input phu for IRAF routines in GMOS package

    :param ad: AstroData instance to find mode of
    :type ad: AstroData instance
    """
    types = ad.types
    if "GMOS" in types:
        if "PREPARED" in types:
            gprep = ad.phu_get_key_value("GPREPARE")
            if gprep is None:
                prepare_date = ad.phu_get_key_value("PREPARE")
                ad.phu_set_key_value("GPREPARE", prepare_date,
                                     "UT Time stamp for GPREPARE")
            else:
                ad.phu_set_key_value("PREPARE", gprep,
                                     "UT Time stamp for GPREPARE")
        if "PROCESSED_BIAS" in types:
            mark_history(adinput=ad, keyword="GBIAS",
                            comment="Temporary key for GIREDUCE")
        if "GMOS_LS_FLAT" in types:
            mark_history(adinput=ad, keyword="GSREDUCE",
                comment="Temporary key for GSFLAT")
        try:
            if "GMOS_IMAGE" in types:
                typeStr = "IMAGE"
            elif "GMOS_IFU" in types:
                typeStr = "IFU"
            elif "GMOS_MOS" in types:
                typeStr = "MOS"
            else:
                typeStr = "LONGSLIT"
            ad.phu_set_key_value("OBSMODE", typeStr ,
                    "Observing mode (IMAGE|IFU|MOS|LONGSLIT)")
        except:
            raise Errors.InputError("Input %s is not of type" +
                "GMOS_IMAGE/_IFU/_MOS or /LS. " % ad.filename)
    return ad

def obsmode_del(ad):
    """This is an internally used function to delete the 'OBSMODE' key from
       the outputs from IRAF routines in the GMOS package.
       
       :param ad: AstroData instance to find mode of
       :type ad: AstroData instance
    """
    if 'GMOS' in ad.types:
        if 'OBSMODE' in ad.phu.header.keys():
            del ad.phu.header['OBSMODE']
        if 'GPREPARE' in ad.phu.header.keys():
            del ad.phu.header['GPREPARE']
        if 'GBIAS' in ad.phu.header.keys():
            del ad.phu.header['GBIAS']
        if 'GSREDUCE' in ad.phu.header.keys():
            del ad.phu.header['GSREDUCE']
    return ad
    

def parse_sextractor_param(default_dict):
    # default_dict formerly made with a get_lookup_table() call.
    # Get path to default sextractor parameter files
    param_file = lookup_path(default_dict["dq"]["param"])
    if param_file.endswith(".py"):
        param_file = param_file[:-3]
    
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
    if database_name is None:
        raise Errors.InputError('No database name specified')
    if not os.path.isdir(database_name):
        raise Errors.InputError('Database directory %s does not exist' %
                                database_name)
    if input_name is None:
        input_name = ad.filename
    if output_name is None:
        output_name = ad.filename

    basename = os.path.basename(input_name)
    basename,filetype = os.path.splitext(basename)
    out_basename = os.path.basename(output_name)
    out_basename,filetype = os.path.splitext(out_basename)

    for sciext in ad[SCI]:
        extver = sciext.extver()

        record_name = basename + "_%0.3d" % extver
        db = at.SpectralDatabase(database_name,record_name)

        out_record_name = out_basename + "_%0.3d" % extver
        table = db.as_binary_table(record_name=out_record_name)

        table_ad = AstroData(table)
        table_ad.rename_ext("WAVECAL",extver)

        if ad["WAVECAL",extver] is not None:
            ad.remove(("WAVECAL",extver))
        ad.append(table_ad)
    return ad

def trim_to_data_section(adinput=None, keyword_comments=None):
    """
    This function trims the data in each SCI extension to the
    the section returned by its data_section descriptor.  VAR and DQ
    planes, if present, are trimmed to the same section as the
    corresponding SCI extension.
    This is intended for use in removing overscan sections, or other
    unused parts of the data array.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)

    try:
        assert isinstance(keyword_comments, dict)
    except AssertionError:
        log.error("TypeError: keyword comments dict was not received.")
        raise TypeError("keyword comments dict required")
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinput)

    # Initialize the list of output AstroData objects
    adoutput_list = []
 
    try:
        for ad in adinput_list:
            for sciext in ad[SCI]:
                # Get matching VAR, DQ, OBJMASK planes if present
                extver = sciext.extver()
                varext = ad[VAR,extver]
                dqext = ad[DQ,extver]
                objmask = ad["OBJMASK",extver]
                
                # Get the data section from the descriptor
                try:
                    # as a string for printing
                    datasecStr = str(sciext.data_section(pretty=True))

                    # as int list of form [x1,x2,y1,y2],
                    # 0-based and non-inclusive
                    dsl = sciext.data_section().as_pytype()

                    # Get the keyword associated with the data_section
                    # descriptor, for later updating.  This keyword 
                    # may be instrument specific.
                    ds_kw = sciext.data_section().keyword
                except:
                    raise Errors.ScienceError("No data section defined; " +
                                              "cannot trim to data section")

                # Check whether data needs to be trimmed
                sci_shape = sciext.data.shape
                if (sci_shape[1]==dsl[1] and 
                    sci_shape[0]==dsl[3] and
                    dsl[0]==0 and
                    dsl[2]==0):
                    sci_trimmed = True
                else:
                    sci_trimmed = False
               
                if sci_trimmed:
                    log.fullinfo("No changes will be made to %s[*,%i], since "\
                                 "the data section matches the data shape" %
                                 (ad.filename,sciext.extver()))
                    continue

                # Update logger with the section being kept
                log.fullinfo("For "+ad.filename+" extension "+
                             str(sciext.extver())+
                             ", keeping the data from the section "+
                             datasecStr,"science")
                
                # Trim the data section from input SCI array
                # and make it the new SCI data
                sciext.data=sciext.data[dsl[2]:dsl[3],dsl[0]:dsl[1]]
                
                # Update header keys to match new dimensions
                newDataSecStr = "[1:"+str(dsl[1]-dsl[0])+",1:"+\
                                str(dsl[3]-dsl[2])+"]" 
                sciext.set_key_value("NAXIS1",dsl[1]-dsl[0],
                                     comment=keyword_comments["NAXIS1"])
                sciext.set_key_value("NAXIS2",dsl[3]-dsl[2],
                                     comment=keyword_comments["NAXIS2"])
                sciext.set_key_value(ds_kw,newDataSecStr,
                                     comment=keyword_comments[ds_kw])
                sciext.set_key_value("TRIMSEC", datasecStr, 
                                     comment=keyword_comments["TRIMSEC"])
                
                # Update WCS reference pixel coordinate
                try:
                    crpix1 = sciext.get_key_value("CRPIX1") - dsl[0]
                    crpix2 = sciext.get_key_value("CRPIX2") - dsl[2]
                except:
                    log.warning("Could not access WCS keywords; using dummy " +
                                "CRPIX1 and CRPIX2")
                    crpix1 = 1
                    crpix2 = 1
                sciext.set_key_value("CRPIX1",crpix1,
                                     comment=keyword_comments["CRPIX1"])
                sciext.set_key_value("CRPIX2",crpix2,
                                     comment=keyword_comments["CRPIX2"])

                # If other planes are present, update them to match
                for ext in [dqext, varext, objmask]:
                    if ext is not None:
                        # Check that ext does not already match the science
                        # (eg. gireduce DQ planes)
                        if ext.data.shape!=sciext.data.shape:
                            # Trim the data
                            ext.data=ext.data[dsl[2]:dsl[3],dsl[0]:dsl[1]]
                            # Set NAXIS keywords
                            ext.set_key_value("NAXIS1",dsl[1]-dsl[0],
                                          comment=keyword_comments["NAXIS1"])
                            ext.set_key_value("NAXIS2",dsl[3]-dsl[2],
                                          comment=keyword_comments["NAXIS2"])
                            # Skip the rest for object masks
                            if ext.extname() in [VAR,DQ]:
                                # Set section keywords
                                ext.set_key_value(ds_kw,newDataSecStr,
                                            comment=keyword_comments[ds_kw])
                                ext.set_key_value("TRIMSEC", datasecStr, 
                                            comment=keyword_comments["TRIMSEC"])
                                # Set WCS keywords
                                ext.set_key_value("CRPIX1",crpix1,
                                            comment=keyword_comments["CRPIX1"])
                                ext.set_key_value("CRPIX2",crpix2,
                                            comment=keyword_comments["CRPIX2"])

            adoutput_list.append(ad)

        return adoutput_list

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def update_key(adinput=None, keyword=None, value=None, comment=None,
               extname=None, keyword_comments=None):
    """
    Add or update a keyword in the specified header of the AstroData object.
    
    :param adinput: The input AstroData object to add or update the keyword
    :type adinput: AstroData
    :param keyword: The keyword to add or update in the header in upper
                    case. The keyword should be less than or equal to 8
                    characters.
    :type keyword: string
    :param value: The value to add or update in the header.
    :param comment: Comment for the keyword. If comment is None, a default
                    comment, as defined in the keyword_comments.py module, will
                    be used. However, if the keyword_comments.py module cannot
                    be found, the comment will not be updated.
    :type comment: string
    :param extname: Name of the extension to add or update the keyword, e.g.,
                   'PHU', 'SCI', 'VAR', 'DQ'
    :type extname: string

    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = logutils.get_logger(__name__)

    if keyword_comments is None and comment is None:
        log.error("TypeError: One of comment or keyword_comments"
                       "must be passed.")
        raise TypeError("Missing parameter: comment or keyword_comments")
                            
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinput)
    
    if len(adinput_list) > 1:
        raise Errors.Error("Please provide only one AstroData object as input")
    else:
        ad = adinput_list[0]
    
    # Validate remaining input parameters
    if keyword is None:
        raise Errors.FITSError("No keyword provided")
    if value is None:
        raise Errors.FITSError("No value provided")
    if extname is None:
        raise Errors.FITSError("No extension name provided")
    if extname is not "PHU":
        if extname == "pixel_exts":
            extname = pixel_exts
        if not ad[extname]:
            raise Errors.FITSError("Extension %s does not exist in %s"
                               % (extname, ad.filename))
    
    # Get the comment for the keyword, if available
    if comment is None:
        if keyword in keyword_comments:
            comment = keyword_comments[keyword]
    
    if extname is "PHU":
        # Check to see whether the keyword is already in the PHU
        original_value = ad.phu_get_key_value(keyword)
        if original_value is not None:
            # The keyword exists
            log.debug("Keyword %s=%s already exists in the PHU" % (
              keyword, original_value))
            msg = "updated in"
        else:
            msg = "added to"
        
        if isinstance(value, DescriptorValue):
            value_for_phu = value.as_pytype()
        else:
            value_for_phu = value
        
        # Add or update the keyword value and comment
        ad.phu_set_key_value(keyword, value_for_phu, comment)
        log.fullinfo("PHU keyword %s=%s %s %s" % (keyword, value_for_phu, msg,
                                                  ad.filename))
    
    else:
        # Loop over each input AstroData object in the input list
        for ext in ad[extname]:
            
            # Retrieve the extension number for this extension
            extname = ext.extname()
            extver = ext.extver()
            
            if isinstance(value, dict):
                # The key of the dictionary could be an (EXTNAME, EXTVER) tuple
                # or an EXTVER integer
                if (extname, extver) in value:
                    value_for_ext = value[(extname, extver)]
                elif extver in value:
                    value_for_ext = value[extver]
                else:
                    raise Errors.FITSError( "The dictionary provided to the"
                                " 'value' parameter contains an unknown key")

            elif isinstance(value, DescriptorValue):
                value_for_ext = value.get_value(extver=extver)
            else:
                value_for_ext = value
            
            # Check to see whether the keyword is already in the specified
            # extension
            original_value = ext.get_key_value(keyword)
            if original_value is not None:
                # The keyword exists
                log.debug("Keyword %s=%s exists in ext %s,%s" % (keyword, 
                                                                 original_value, 
                                                                 extname, 
                                                                 extver))
                msg = "updated in"
            else:
                msg = "added to"
            
            # Add or update the keyword value and comment
            if value_for_ext is not None:
                ext.set_key_value(keyword, value_for_ext, comment)
                log.fullinfo("%s,%s keyword %s=%s %s %s" % (
                  extname, extver, keyword, value_for_ext, msg, ad.filename))

def update_key_from_descriptor(adinput=None, descriptor=None, keyword=None,
                               extname=None, keyword_comments=None):
    """
    Add or update a keyword in the specified header of the AstroData object
    with a value determined from the specified descriptor.
    
    :param adinput: The input AstroData object to add or update the keyword
    :type adinput: AstroData
    :param descriptor: Name of the descriptor used to obtain the value that
                       will be written to the header, e.g., 'gain()'
    :type descriptor: string 
    :param extname: Name of the extension to add or update the keyword, e.g.,
                   'PHU', 'SCI', 'VAR', 'DQ'
    :type extname: string
    """
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinput)
    
    if len(adinput_list) > 1:
        raise Errors.Error("Please provide only one AstroData object as input")
        
    # Validate remaining input parameters
    if descriptor is None:
        raise Errors.Error("No descriptor name provided")
    if extname is None:
        raise Errors.Error("No extension name provided")
    
    # Determine the value of the descriptor
    ad = adinput_list[0]
    dv = eval("ad.%s" % descriptor)
    
    if dv.is_none():
        raise Errors.Error("No value found for %s descriptor in %s" %
                           (descriptor, ad.filename))
    
    if keyword is None:
        # Use the default keyword stored in the DescriptorValue object
        key = dv.keyword
    else:
        key = keyword
    
    if key is None:
        raise Errors.Error("No keyword found for descriptor %s" % descriptor)
    
    update_key(adinput=ad, keyword=key, value=dv, extname=extname, 
               keyword_comments=keyword_comments)
    return

def validate_input(input=None, dtype=None):
    """
    The validate_input helper function is used to validate the input value to
    the input parameter.

    :param input: An input object or list of objects. If None, an exception
         is raised.
    :type input: any

    :param dtype: Acceptable type(s) for the input objects. If None, no check
         is made, otherwise an error is raised unless all input elements have
         one of the required types.
    :type dtype: type or tuple of types

    :returns: A list containing one or more inputs
    :rtype: list
    """
    # If the input is None, raise an exception
    if input is None:
        raise Errors.InputError("The input cannot be None")
    
    # If the input is a single input, put it in a list
    if not isinstance(input, list):
        input = [input]
    
    # If the input is an empty list, raise an exception
    if len(input) == 0:
        raise Errors.InputError("The input cannot be an empty list")

    # Complain if the list elements aren't all of a specified type:
    if dtype is not None:
        if not all(isinstance(element, dtype) for element in input):
            raise Errors.InputError("Input list elements should be of " + \
                "%s" % str(dtype))
    
    # Now, input is a list that contains one or more inputs
    return input

def write_database(ad, database_name=None, input_name=None):
    if input_name is None:
        input_name = ad.filename

    basename = os.path.basename(input_name)
    basename,filetype = os.path.splitext(basename)

    for sciext in ad[SCI]:
        record_name = basename + "_%0.3d" % sciext.extver()
        wavecal_table = ad["WAVECAL",sciext.extver()]
        if wavecal_table is None:
            raise Errors.InputError('WAVECAL extension must exist '\
                                       'to write spectroscopic database')
        db = at.SpectralDatabase(binary_table=wavecal_table,
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

    def __init__(self, adinputs, pkg, frac_FOV=1.0):

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

        # Make sure we have a non-empty input list of AstroData instances:
        adinputs = validate_input(input=adinputs, dtype=AstroData)

        # Make sure the field scaling is valid:
        if not isinstance(frac_FOV, numbers.Number) or frac_FOV < 0.:
            raise Errors.InputError('frac_FOV must be >= 0.')

        # Initialize members:
        self.members = {}
        self.package = pkg
        self._frac_FOV = frac_FOV
        self.group_cen = (0., 0.)
        self.add_members(adinputs)

        # Use the first list element as a reference for getting the
        # instrument properties:
        ref_ad = adinputs[0]

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

    def add_members(self, adinputs):

        """
        Add one or more new points to the group.

        :param adinputs: A list of AstroData instances to add to the group
            membership. 
        :type adinputs: AstroData, list of AstroData instances
        """

        # Make sure we have a non-empty input list of AstroData instances:
        adinputs = validate_input(input=adinputs, dtype=AstroData)

        # How many points were there previously and will there be now?
        ngroups = self.__len__()
        ntot = ngroups + len(adinputs)

        # This will be replaced by a descriptor that looks up the RA/Dec
        # of the field centre:
        addict = get_offset_dict(adinputs)

        # Complain sensibly if we didn't get valid co-ordinates:
        for ad in addict:
            for coord in addict[ad]:
                if not isinstance(coord, numbers.Number):
                    raise Errors.InputError('non-numeric co-ordinate %s ' \
                        % coord + 'from %s' % ad)

        # Add the new points to the group list:
        self.members.update(addict)

        # Update the group centroid to account for the new points:
        new_vals = addict.values()
        newsum = [sum(axvals) for axvals in zip(*new_vals)]
        self.group_cen = [(cval * ngroups + nval) / ntot \
          for cval, nval in zip(self.group_cen, newsum)]


def group_exposures(adinputs, pkg, frac_FOV=1.0):

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
    for ad in adinputs:

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
    inst = str(pos.instrument())

    # To keep the back-end functions simple, always pass them a dictionary
    # for the reference position (at least for now). All these checks add ~4%
    # in overhead for imaging.
    if isinstance(refpos, AstroData):
        pointing = (refpos.phu_get_key_value('POFFSET'),
                    refpos.phu_get_key_value('QOFFSET'))
    else:
        if not isinstance(refpos, (list, tuple)) or \
           not all(isinstance(x, numbers.Number) for x in refpos):
            raise Errors.InputError('Parameter refpos should be a ' + \
                'co-ordinate tuple or AstroData instance')
        # Currently the comparison is always 2D since we're explicitly
        # looking up POFFSET & QOFFSET:
        if len(refpos) != 2:
            raise Errors.InputError('Points to group must have the ' + \
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
    FOV_mod = "astrodata_{0}.ADCONFIG_{0}.lookups.{1}.FOV".format(package, inst)

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
def get_offset_dict(adinputs=None):
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

    # # Instantiate the log.
    # log = logutils.get_logger(__name__)

    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more inputs
    adinput_list = validate_input(input=adinputs, dtype=AstroData)

    # Loop over AstroData instances:
    for ad in adinputs:

        # Get the offsets from the primary header:
        poff = ad.phu_get_key_value('POFFSET')
        qoff = ad.phu_get_key_value('QOFFSET')
        # name = ad.filename
        name = ad  # store a direct reference

        offsets[name] = (poff, qoff)

    return offsets

