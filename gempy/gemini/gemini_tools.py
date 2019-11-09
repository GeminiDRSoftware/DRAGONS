from __future__ import division
#
#                                                                  gemini_python
#
#                                                                   gempy.gemini
#                                                                gemini_tools.py
# ------------------------------------------------------------------------------
from builtins import str
from builtins import zip
from builtins import range
from builtins import object
import os
import re
import numbers
import itertools
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

from ..library import astromodels, tracing, astrotools as at
from ..library.nddops import NDStacker
from ..utils import logutils

import astrodata

from collections import namedtuple
ArrayInfo = namedtuple("ArrayInfo", "detector_shape origins array_shapes "
                                    "extensions")

@models.custom_model
def CumGauss1D(x, mean=0.0, stddev=1.0):
    return 0.5*(1.0+erf((x-mean) / (1.414213562*stddev)))

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
        assert isinstance(sx_dict, dict) and ('dq', 'param') in sx_dict
    except AssertionError:
        log.error("TypeError: A SExtractor dictionary was not received.")
        raise TypeError("Require SExtractor parameter dictionary.")
    
    adoutput = []
    # Parse sextractor parameters for the list of expected columns
    expected_columns = parse_sextractor_param(sx_dict['dq', 'param'])
    # Append a few more that don't come from directly from sextractor
    expected_columns.extend(["REF_NUMBER","REF_MAG","REF_MAG_ERR",
                             "PROFILE_FWHM","PROFILE_EE50"])

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
                    default = list(range(1, nrows+1))
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
    and physical detectors. This works for any 2D array provided the
    array_section() and detector_section() descriptors return correct values.

    Parameters
    ----------
    adinput: list/AD
        single or list of AD objects

    Returns
    -------
    list of ArrayInfo namedtuples, each containing:
        detector_shape: doubleton indicating the arrangement of physical
                        detectors
        origins: list of doubletons indicating the bottom-left of each
                 physical detector in detector_section coordinates
        array_shapes: list of doubletons indicating the shape for tiling
                      the extensions to make each physical detector
        extensions: list of tuples, one for each physical detector containing
                    the indices of the extensions on that detector. Both the
                    list and the tuples are sorted based on detector_section
                    first in y, then x (so order is along the bottom, then
                    the next row up, and so on).

        An example for the standard 12-amp GMOS read is:
            detector_shape=(1, 3)
            origins=[(0, 0), (0, 2048), (0, 4096)]
            array_shapes=[(1, 4), (1, 4), (1, 4)]
            extensions=[(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
    """
    array_info_list = []
    for ad in adinput:
        det_corners = np.array([(sec.y1, sec.x1) for sec in ad.detector_section()])
        # If the array_section() descriptor returns None, then it's reasonable
        # to assume that each extension is a full detector...
        try:
            array_corners = np.array([(sec.y1, sec.x1) for sec in ad.array_section()])
        except AttributeError:
            array_corners = det_corners
        origins = det_corners - array_corners

        # Sort by y first, then x as a tiebreaker, keeping all extensions with
        # the same origin together
        ampsorder = np.lexsort(np.vstack([det_corners.T[::-1], origins.T[::-1]]))
        unique_origins = np.unique(origins, axis=0)
        detshape = tuple(len(set(orig_coords)) for orig_coords in unique_origins.T)
        sorted_origins = [tuple(unique_origins[i])
                          for i in np.lexsort(unique_origins.T[::-1])]
        arrays_list = [tuple(j for j in ampsorder if np.array_equal(det_corners[j],
                             array_corners[j]+origin)) for origin in sorted_origins]
        array_shapes = [tuple(len(set(coords)) for coords in det_corners[exts,:].T)
                        for exts in arrays_list]
        array_info_list.append(ArrayInfo(detshape, sorted_origins,
                                         array_shapes, arrays_list))
    return array_info_list

def check_inputs_match(adinput1=None, adinput2=None, check_filter=True,
                       check_units=False):
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

    check_units: bool
        if True, also check that both inputs are in electrons or ADUs
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

            # Check units if desired
            if check_units:
                if ext1.is_in_adu() != ext2.is_in_adu():
                    raise ValueError('Extensions have different units')

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
    else:
        for i in range(len(ad1)):
            if ad1.nddata[i].shape != ad2.nddata[i].shape:
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
            log.debug('  Descriptor failure for {}'.format(descriptor))

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
                        return_dtype=None):
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

    Returns
    -------
    list/AD:
        auxiliary file(s), appropriately clipped
    """
    log = logutils.get_logger(__name__)
    aux_output_list = []

    for ad, this_aux in zip(*make_lists(adinput, aux, force_ad=True)):
        clipped_this_ad = False

        # Make a new auxiliary file for appending to, starting with PHU
        new_aux = astrodata.create(this_aux.phu)
        new_aux.filename = this_aux.filename
        new_aux.update_filename(suffix="_clipped", strip=False)

        # Get the detector section, data section, array section and the
        # binning of the x-axis and y-axis values for the science AstroData
        # object using the appropriate descriptors
        sci_detsec = ad.detector_section()
        sci_datasec = ad.data_section()
        sci_arraysec = ad.array_section()
        sci_xbin = ad.detector_x_bin()
        sci_ybin = ad.detector_y_bin()

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
            arraysec = [arraysec[0] // sci_xbin, arraysec[1] // sci_xbin,
                        arraysec[2] // sci_ybin, arraysec[3] // sci_ybin]

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
                if not (auxext.detector_x_bin() == sci_xbin and
                        auxext.detector_y_bin() == sci_ybin):
                    continue

                # Array section is unbinned; to use as indices for
                # extracting data, need to divide by the binning
                aarraysec = [
                    aarraysec[0] // sci_xbin, aarraysec[1] // sci_xbin,
                    aarraysec[2] // sci_ybin, aarraysec[3] // sci_ybin]

                # Check whether auxiliary detector section contains
                # science detector section
                if (adetsec[0] <= detsec[0] and adetsec[1] >= detsec[1] and
                    adetsec[2] <= detsec[2] and adetsec[3] >= detsec[3]):
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
                    # We're not actually clipping!
                    if region != list(sum(zip([0,0], ext_to_clip[0].nddata.shape),())):
                        clipped_this_ad = True
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
                    raise ValueError("Overscan regions do not match in {}, {}".
                        format(auxext.filename, ext.filename))

                # Convert the dtype if requested (only SCI and VAR)
                if return_dtype is not None:
                    #ext_to_clip.operate(np.ndarray.astype, return_dtype)
                    ext_to_clip[0].data = ext_to_clip[0].data.astype(return_dtype)
                    if ext_to_clip[0].variance is not None:
                        ext_to_clip[0].variance = \
                            ext_to_clip[0].variance.astype(return_dtype)

                # Update keywords based on the science frame
                for descriptor in ('data_section', 'detector_section',
                                   'array_section'):
                    try:
                        kw = ext._keyword_for(descriptor)
                        ext_to_clip.hdr[kw] = (ext.hdr[kw],
                                               ext.hdr.comments[kw])
                    except (AttributeError, KeyError):
                        pass

                # Append the data to the AD object
                new_aux.append(ext_to_clip[0].nddata, reset_ver=True)

            if not found:
                raise IOError(
                  "No auxiliary data in {} matches the detector section "
                  "{} in {}[SCI,{}]".format(this_aux.filename, detsec,
                                       ad.filename, ext.hdr['EXTVER']))

        if clipped_this_ad:
            log.stdinfo("Clipping {} to match science data.".
                        format(os.path.basename(this_aux.filename)))
        aux_output_list.append(new_aux)

    return aux_output_list

@handle_single_adinput
def clip_auxiliary_data_GSAOI(adinput=None, aux=None, aux_type=None, 
                        return_dtype=None):
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

    Returns
    -------
    list/AD:
        auxiliary file(s), appropriately clipped
    """
    log = logutils.get_logger(__name__)

    if not isinstance(aux, list):
        aux = [aux]

    aux_output_list = []

    for ad, this_aux in zip(adinput, aux):
        # Make a new auxiliary file for appending to, starting with PHU
        new_aux = astrodata.create(this_aux.phu)

        # Get the detector section, data section, array section and the
        # binning of the x-axis and y-axis values for the science AstroData
        # object using the appropriate descriptors
        sci_detsec = ad.detector_section()
        sci_datasec = ad.data_section()
        sci_arraysec = ad.array_section()

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

                # Convert the dtype if requested (only SCI and VAR)
                if return_dtype is not None:
                    #ext_to_clip.operate(np.ndarray.astype, return_dtype)
                    ext_to_clip[0].data = ext_to_clip[0].data.astype(return_dtype)
                    if ext_to_clip[0].variance is not None:
                        ext_to_clip[0].variance = \
                            ext_to_clip[0].variance.astype(return_dtype)

                # Update keywords based on the science frame
                for descriptor in ('data_section', 'detector_section',
                                   'array_section'):
                    try:
                        kw = ext._keyword_for(descriptor)
                        ext_to_clip.hdr[kw] = (ext.hdr[kw],
                                               ext.hdr.comments[kw])
                    except (AttributeError, KeyError):
                        pass

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
    log = logutils.get_logger(__name__)

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

    # Produce warning but return what is expected
    if not any([hasattr(ext, 'OBJCAT') for ext in ad]):
        log.warning("No OBJCATs found on input. Has detectSources() been run?")
        return [Table()] * len(ad)

    good_sources = []
    for ext in ad:
        try:
            objcat = ext.OBJCAT
        except AttributeError:
            good_sources.append(Table())
            continue

        stellar = np.fabs((objcat['FWHM_IMAGE'] / 1.08) - objcat['PROFILE_FWHM']
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
        for new_name, old_name in column_mapping.items():
            table[new_name] = objcat[old_name][good]
        pixscale = ext.pixel_scale()
        table['fwhm_arcsec'] *= pixscale
        table['ee50d_arcsec'] *= pixscale
        table['isofwhm_arcsec'] *= 3600  # degrees -> arcseconds

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

    fitsfilenamecre = re.compile(r"^([NS])(20\d\d)([01]\d[0123]\d)(S)"
                                 r"(?P<fileno>\d\d\d\d)(.*)$")

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
        if "arc" in caltype:
            ad.phu.set("OBSTYPE", "ARC", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "Arc spectrum", keyword_comments["OBJECT"])
        elif "bias" in caltype:
            ad.phu.set("OBSTYPE", "BIAS", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "Bias Frame", keyword_comments["OBJECT"])
        elif "dark" in caltype:
            ad.phu.set("OBSTYPE", "DARK", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "Dark Frame", keyword_comments["OBJECT"])
        elif "fringe" in caltype:
            ad.phu.set("OBSTYPE", "FRINGE", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "Fringe Frame", keyword_comments["OBJECT"])
        elif "sky" in caltype:
            ad.phu.set("OBSTYPE", "SKY", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "Sky Frame", keyword_comments["OBJECT"])
        elif "flat" in caltype:
            ad.phu.set("OBSTYPE", "FLAT", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "Flat Frame", keyword_comments["OBJECT"])
        elif "bpm" in caltype:
            ad.phu.set("BPMASK", True, "Bad pixel mask")
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
    
    for ad in adinput:
        # Add the appropriate time stamps to the PHU
        if timestamp_key is not None:
            mark_history(adinput=ad, keyword=timestamp_key)
        # Update the filename
        if suffix is not None:
            ad.update_filename(suffix=suffix, strip=True)
        adoutput_list.append(ad)
    return adoutput_list


def fit_continuum(ad):
    """
    This function fits Gaussians to the spatial profile of a 2D image,
    collapsed in the dispersion direction. It works both for dispersed 2D
    spectral images (tagged as "SPECT") and through-slit images ("IMAGE").
    Depending on the type of data and its processed state, there may be
    information about the locations of sources suitable for profile
    measurement.

    For MOS data, the acquisition slits are used as these are known to
    contain stellar objects; if this information is missing from the
    header (where it is placed by the findAcqusitionSlits primitive) then
    no further processing is performed.

    If an extension has an APERTURE table, then source positions are taken
    from this, otherwise the brightest pixel is used as an initial guess.
    
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
    
    pixel_scale = ad.pixel_scale()
    spatial_box = int(5.0 / pixel_scale)

    # Initialize the Gaussian width to FWHM = 1.2 arcsec
    init_width = 1.2 / (pixel_scale * (2 * np.sqrt(2 * np.log(2))))

    tags = ad.tags
    acq_star_positions = ad.phu.get("ACQSLITS")

    for ext in ad:
        fwhm_list = []
        x_list, y_list = [], []
        weight_list = []

        dispaxis = 2 - ext.dispersion_axis()  # python sense
        dispersed_length = ext.shape[dispaxis]
        binnings = (ext.detector_x_bin(), ext.detector_y_bin())
        spatial_bin, spectral_bin = binnings[1-dispaxis], binnings[dispaxis]

        # Determine regions for collapsing into 1D spatial profiles
        if 'IMAGE' in tags:
            # A through-slit image: extract a 2-arcsecond wide region about
            # the center. This should be OK irrespective of the actual slit
            # width and avoids having to work out the width from instrument-
            # dependent header information.
            centers = [dispersed_length // 2]
            hwidth = int(1.0 / pixel_scale + 0.5)
        else:
            # This is a dispersed 2D spectral image, chop it into 512-pixel
            # (unbinned) sections. This is one GMOS amp, and most detectors
            # are a multiple of 512 pixels. We do this because the data are
            # unlikely to have gone through traceApertures() so we can't
            # account for curvature in the spectral trace.
            hwidth = 512 // spectral_bin
            centers = [i*hwidth for i in range(1, dispersed_length // hwidth, 1)]
        spectral_slices = [slice(center-hwidth, center+hwidth) for center in centers]

        # Try to get an idea of where peaks might be from header information
        # We start with the entire slit length (for N&S, only use the part
        # with both the +ve and -ve beams)
        if 'NODANDSHUFFLE' in tags:
            shuffle = ad.shuffle_pixels() // spatial_bin
            spatial_slice = slice(shuffle, shuffle * 2)
        else:
            spatial_slice = slice(0, ext.shape[1 - dispaxis])

        # First, if there are acquisition slits, we use those positions
        spatial_slices = []
        if acq_star_positions is None:
            if 'MOS' in tags:
                log.warning("{} is MOS but has no acquisition slits. "
                            "Not trying to find spectra.".format(ad.filename))
                if not hasattr(ad, 'MDF'):
                    log.warning("No MDF is attached. Did addMDF find one?")
                    good_sources.append(Table())
                continue
            try:
                # No acquisition slits, maybe we've already found apertures?
                aptable = ext.APERTURES
                for row in aptable:
                    model_dict = dict(zip(aptable.colnames, row))
                    trace_model = astromodels.dict_to_chebyshev(model_dict)
                    aperture = tracing.Aperture(trace_model,
                                                aper_lower=model_dict['aper_lower'],
                                                aper_upper=model_dict['aper_upper'])
                    spatial_slices.append(aperture)
            except AttributeError:
                # No apertures, so find sources now in the region already defined
                # Taking the 95th percentile should remove CRs
                if ext.mask is None:
                    profile = np.percentile(ext.data, 95, axis=dispaxis)
                else:
                    profile = np.percentile(np.where(ext.mask==0, ext.data, 0), 95, axis=dispaxis)
                center = np.argmax(profile[spatial_slice]) + spatial_slice.start
                spatial_slices = [slice(max(center-spatial_box, 0),
                                        min(center+spatial_box, ext.shape[1-dispaxis]))]
        else:
            for slit in acq_star_positions.split():
                c, w = [int(x) for x in slit.split(':')]
                spatial_slices.append(slice(c-w, c+w))

        from matplotlib import pyplot as plt
        for spectral_slice in spectral_slices:
            coord = 0.5 * (spectral_slice.start + spectral_slice.stop)

            for spatial_slice in spatial_slices:
                try:
                    length = spatial_slice.stop - spatial_slice.start
                except AttributeError:
                    # It's an Aperture, so convert to a regular slice object
                    limits = spatial_slice.model(coord) + np.array([spatial_slice.lower,
                                                                    spatial_slice.upper])
                    spatial_slice = slice(*limits)
                    length = spatial_slice.stop - spatial_slice.start

                # These are all in terms of the full unsliced extension
                pixels = np.arange(spatial_slice.start, spatial_slice.stop)

                # Cut the rectangular region and collapse it
                if dispaxis == 0:
                    full_slice = (spectral_slice, spatial_slice)
                    ndd = ext.nddata[full_slice]
                else:
                    full_slice = (spatial_slice, spectral_slice)
                    ndd = ext.nddata[full_slice].T

                data, mask, var = NDStacker.mean(ndd)
                maxflux = np.max(abs(data))
                
                # Crude SNR test; is target bright enough in this wavelength range?
                #if np.percentile(col,90)*xbox < np.mean(databox[dqbox==0]) \
                #            + 10*np.sqrt(np.median(varbox)):
                #    continue

                # Check that the spectrum looks like a continuum source.
                # This is needed to avoid cases where another slit is shuffled onto
                # the acquisition slit, resulting in an edge that the code tries
                # to fit with a Gaussian. That's not good, but if source is very
                # bright and dominates spectrum, then it doesn't matter.
                # All non-N&S data should pass this step, which checks whether 80%
                # of the spectrum has SNR>2
                #spectrum = databox[np.argmax(col),:]
                #if np.percentile(spectrum,20) < 2*np.sqrt(np.median(varbox)):
                #    continue

                # Set one profile to be +ve, one -ve, and widths equal
                m_init = models.Gaussian1D(amplitude=maxflux, mean=pixels[np.argmax(data)],
                            stddev=init_width) + models.Gaussian1D(amplitude=-maxflux,
                                mean=pixels[np.argmin(data)], stddev=init_width) + models.Const1D(0.)
                m_init.amplitude_0.min = 0.
                m_init.amplitude_1.max = 0.
                m_init.stddev_1.tied = lambda f: f.stddev_0

                if 'NODANDSHUFFLE' in tags and shuffle < length:
                    # Fix background=0 if slit is in region where sky-subtraction will occur
                    if spatial_slice.start >= shuffle and spatial_slice.stop <= shuffle * 2:
                            m_init.amplitude_2.fixed = True
                else:
                    # force -ve profile to have zero amplitude (since there won't be one)
                    m_init.amplitude_1 = 0.
                    m_init.amplitude_1.fixed = True

                fit_it = fitting.LevMarLSQFitter()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m_final = fit_it(m_init, pixels, data)

                if fit_it.fit_info['ierr'] < 5:
                    # This is kind of ugly and empirical; philosophy is that peak should
                    # be away from the edge and should be similar to the maximum
                    if (m_final.amplitude_0 > 0.5*maxflux and
                        pixels.min()+1 < m_final.mean_0 < pixels.max()-1):
                        fwhm = abs(2 * np.sqrt(2*np.log(2)) * m_final.stddev_0)
                        fwhm_list.append(fwhm)
                        if dispaxis == 0:
                            x_list.append(m_final.mean_0.value)
                            y_list.append(coord)
                        else:
                            x_list.append(coord)
                            y_list.append(m_final.mean_0.value)
                        # Add a "weight" based on distance from edge
                        # If only one spectrum, all weights will be basically the same
                        if m_final.mean_1 > m_final.mean_0:
                            weight_list.append(max(m_final.mean_0 - pixels.min(),
                                                   pixels.max() - m_final.mean_1))
                        else:
                            weight_list.append(max(m_final.mean_1 - pixels.min(),
                                                   pixels.max() - m_final.mean_0))


        # Now do something with the list of measurements
        fwhm_pix = np.array(fwhm_list)
        fwhm_arcsec = pixel_scale * fwhm_pix

        table = Table([x_list, y_list, fwhm_pix, fwhm_arcsec, weight_list],
                    names=("x", "y", "fwhm", "fwhm_arcsec", "weight"))

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

def make_lists(*args, **kwargs):
    """
    The make_lists function attaches auxiliary things to an input key_list
    of (normally) AD objects. Each key gets exactly one auxilary thing from
    each other list -- these lists can be as long as the key_list, or have
    only one item in (in which case they don't have to be lists at all).

    Parameters
    ----------
    args: lists of str/AD (or single str/AD)
        key_list and auxiliary things to be matched to each AD
    kwargs["force_ad"]: bool
        coerce strings into AD objects?

    Returns
    -------
    tuple of lists
        the lists made from the keys and values
    """
    force_ad = kwargs.pop("force_ad", False)
    if kwargs:
        raise TypeError("make_lists() got unexpected keyword arguments "
                        "{}".format(kwargs.keys()))

    ret_value = [arg if isinstance(arg, (list, tuple)) else [arg]
                 for arg in args]

    # We allow only one value that can be assigned to multiple keys
    len_list = len(ret_value[0])
    if len_list > 1:
        for i in range(1, len(ret_value)):
            if len(ret_value[i]) == 1:
                ret_value[i] *= len_list

    if force_ad:
        # We only want to open as many AD objects as there are unique entries,
        # so collapse all items in lists to a set and multiple keys with the
        # same value will be assigned references to the same open AD object
        ad_map_dict = {}
        for x in set(itertools.chain(*ret_value)):
            try:
                ad_map_dict.update({x: x if isinstance(x, astrodata.AstroData)
                                        or x is None else astrodata.open(x)})
            except:
                ad_map_dict.update({x: None})
        ret_value = [[ad_map_dict[x] for x in List] for List in ret_value]

    return ret_value

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
        for key, comm in keyword_dict.items():
            ad.phu.set(key, tlm, comm)

    return


def measure_bg_from_image(ad, sampling=10, value_only=False, gaussfit=True,
                          separate_ext=True):
    """
    Return background value, and its std deviation, as measured directly
    from pixels in the SCI image. DQ plane are used (if they exist)
    If extver is set, return a double for that extension only,
    otherwise return a list of doubles, as long as the number of extensions.

    Parameters
    ----------
    ad: AstroData
    sampling: int
        1-in-n sampling factor
    value_only: bool
        if True, return only background values, not the standard deviations
    gaussfit: bool
        if True, fit a Gaussian to the pixel values, instead of sigma-clipping?
    separate_ext: bool
        return information for each extension, rather than the whole AD?

    Returns
    -------
    list/value/tuple
        if ad is single extension, or separate_ext==False, returns a bg value
        or (bg, std, number of samples) tuple; otherwise returns a list of
        such things
    """
    # Handle NDData objects (or anything with .data and .mask attributes
    try:
        single = ad.is_single
    except AttributeError:
        single = True
    input_list = [ad] if single or not separate_ext else [ext for ext in ad]

    output_list = []
    for ext in input_list:
        # Use DQ and OBJMASK to flag pixels
        if not single and not separate_ext:
            bg_data = np.array([ext.data for ext in ad]).ravel()
            flags = np.array([ext.mask | getattr(ext, 'OBJMASK', 0)
                              if ext.mask is not None
                else getattr(ext, 'OBJMASK', np.empty_like(ext.data, dtype=bool))
                              for ext in ad]).ravel()
        else:
            flags = ext.mask | getattr(ext, 'OBJMASK', 0) if ext.mask is not None \
                else getattr(ext, 'OBJMASK', None)
            bg_data = ext.data.ravel()

        if flags is None or np.sum(flags==0) == 0:
            bg_data = bg_data[::sampling]
        else:
            bg_data = bg_data[flags.ravel()==0][::sampling]

        if len(bg_data) > 0:
            if gaussfit:
                # An ogive fit is more robust than a histogram fit
                bg_data = np.sort(bg_data)
                bg = np.median(bg_data)
                bg_std = 0.5*(np.percentile(bg_data, 84.13) -
                              np.percentile(bg_data, 15.87))
                g_init = CumGauss1D(bg, bg_std)
                fit_g = fitting.LevMarLSQFitter()
                g = fit_g(g_init, bg_data, np.linspace(0.,1.,len(bg_data)+1)[1:])
                bg, bg_std = g.mean.value, abs(g.stddev.value)
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

    return output_list[0] if single or not separate_ext else output_list


def measure_bg_from_objcat(ad, min_ok=5, value_only=False, separate_ext=True):
    """
    Return a list of triples of background values, (and their std deviations
    and the number of objects used, if requested) from the OBJCATs in an AD.
    If there are too few good BG measurements, None is returned.
    If the input has SCI extensions, then the output list contains one tuple
    per SCI extension, even if no OBJCAT is associated with that extension.
    If a single extension/OBJCAT is sent, or separate_ext==False, then
    return only the value or triple, not a list.

    Parameters
    ----------
    ad: AstroData/Table
        image with OBJCATs to measure background from, or OBJCAT
    min_ok: int
        return None if fewer measurements than this (after sigma-clipping)
    value_only: bool
        if True, only return the background values, not the other stuff
    separate_ext: bool
        return information for each extension, rather than the whole AD?

    Returns
    -------
    list/value/triple
        as described above
    """
    # Populate input list with either the OBJCAT or a list
    single = True
    if isinstance(ad, Table):
        input_list = [ad]
    elif ad.is_single:
        input_list = [getattr(ad, 'OBJCAT', None)]
    elif separate_ext:
        input_list = [getattr(ext, 'OBJCAT', None) for ext in ad]
        single = False
    else:
        input_list = [vstack([getattr(ext, 'OBJCAT', Table()) for ext in ad],
                             metadata_conflicts='silent')]

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
    return output_list[0] if single else output_list

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
    for ext in adoutput:
        outextver = ext.hdr['EXTVER']
        output_wcs = WCS(ext.hdr)
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

    adoutput_list = []

    for ad in adinput:
        # Get the keyword associated with the data_section descriptor
        datasec_kw = ad._keyword_for('data_section')
        oversec_kw = ad._keyword_for('overscan_section')

        for ext in ad:
            # Get data section as string and as a tuple
            datasecStr = ext.data_section(pretty=True)
            datasec = ext.data_section()

            # Check whether data need to be trimmed
            sci_shape = ext.data.shape
            if (sci_shape[0]==datasec.y2 and sci_shape[1]==datasec.x2 and
                datasec.x1==0 and datasec.y1==0):
                log.fullinfo('No changes will be made to {}[*,{}], since '
                             'the data section matches the data shape'.format(
                             ad.filename,ext.hdr['EXTVER']))
                continue

            # Update logger with the section being kept
            log.fullinfo('For {}:{}, keeping the data from the section {}'.
                         format(ad.filename, ext.hdr['EXTVER'], datasecStr))

            # Trim SCI, VAR, DQ to new section
            ext.reset(ext.nddata[datasec.y1:datasec.y2,datasec.x1:datasec.x2])
            # And OBJMASK (if it exists)
            # TODO: should check more generally for any image extensions
            if hasattr(ext, 'OBJMASK'):
                ext.OBJMASK = ext.OBJMASK[datasec.y1:datasec.y2,
                              datasec.x1:datasec.x2]

            # Update header keys to match new dimensions
            newDataSecStr = '[1:{},1:{}]'.format(datasec.x2-datasec.x1,
                                                 datasec.y2-datasec.y1)
            ext.hdr.set('NAXIS1', datasec.x2-datasec.x1, keyword_comments['NAXIS1'])
            ext.hdr.set('NAXIS2', datasec.y2-datasec.y1, keyword_comments['NAXIS2'])
            ext.hdr.set(datasec_kw, newDataSecStr, comment=keyword_comments[datasec_kw])
            ext.hdr.set('TRIMSEC', datasecStr, comment=keyword_comments['TRIMSEC'])
            if oversec_kw in ext.hdr:
                del ext.hdr[oversec_kw]


            # Update WCS reference pixel coordinate
            try:
                crpix1 = ext.hdr['CRPIX1'] - datasec.x1
                crpix2 = ext.hdr['CRPIX2'] - datasec.y1
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


class ExposureGroup(object):
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
        return list(self.members.keys())

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
        new_vals = list(addict.values())
        newsum = [sum(axvals) for axvals in zip(*new_vals)]
        self.group_cen = [(cval * ngroups + nval) / ntot \
          for cval, nval in zip(self.group_cen, newsum)]

def group_exposures(adinput, pkg=None, frac_FOV=1.0):

    """
    Sort a list of AstroData instances into dither groups around common
    nod positions, according to their WCS offsets.

    :param adinputs: A list of exposures to sort into groups.
    :type adinputs: list of AstroData instances

    :param pkg: package containing instrument lookups. Passed through
                to ExposureGroup() call.
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

    try:
        FOV = import_module('{}.FOV'.format(package))
        _FOV_pointing_in_field = FOV.pointing_in_field
    except (ImportError, AttributeError):
        raise NameError("FOV.pointing_in_field() function not found in {}".
                        format(package))

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
