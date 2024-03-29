#
#                                                                  gemini_python
#
#                                                                   gempy.gemini
#                                                                gemini_tools.py
# ------------------------------------------------------------------------------
import os
import re
import numbers

from copy import deepcopy
from datetime import datetime
from functools import wraps
from collections import namedtuple

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy.table import vstack, Table, Column

from scipy.ndimage import distance_transform_edt
from scipy.special import erf

from ..library import astromodels, tracing, astrotools as at
from ..library.nddops import NDStacker
from ..utils import logutils

import astrodata
from astrodata import Section

from gempy.utils.errors import ConvergenceError

ArrayInfo = namedtuple("ArrayInfo", "detector_shape origins array_shapes "
                                    "extensions")

@models.custom_model
def Ogive(x, mean=0.0, stddev=1.0, lsigma=3.0, hsigma=3.0):
    """A cumulative frequency curve for the Normal distribution between
    -lsigma and hsigma standard deviations, normalized to number of points"""
    lfrac = 0.5 * (1 - erf(lsigma / 1.414213562))
    hfrac = 0.5 * (1 + erf(hsigma / 1.414213562))
    return (0.5 * (1.0 + erf((x - mean) / (1.414213562 * stddev))) - lfrac) / (hfrac - lfrac)

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
def add_objcat(adinput=None, index=0, replace=False, table=None, sx_dict=None):
    """
    Add OBJCAT table if it does not exist, update or replace it if it does.

    Parameters
    ----------
    adinput : AstroData
        AD object(s) to add table to

    index : int
        Extension index for the table

    replace : bool
        replace (overwrite) with new OBJCAT? If False, the new table must
        have the same length as the existing OBJCAT

    table : Table
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
    expected_columns.extend(["REF_NUMBER", "REF_MAG", "REF_MAG_ERR",
                             "PROFILE_FWHM", "PROFILE_EE50"])

    for ad in adinput:
        ext = ad[index]
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
                    if 'MAG' in name:
                        unit = 'mag'
                    elif 'PROFILE' in name:
                        unit = 'pix'
                    else:
                        unit = None
                # Use input table column if given, otherwise the placeholder
                new_objcat.add_column(
                    table[name] if name in table.columns else
                    Column(data=default, name=name, dtype=dtype, unit=unit))

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
                       check_shape=True, check_units=False):
    """
    This function will check if the inputs match.  It will check the filter,
    binning and shape/size of the every SCI frames in the inputs.

    There must be a matching number of inputs for 1 and 2.

    Parameters
    ----------
    adinput1 : list/AD
    adinput2 : list/AD
        single AstroData instances or length-matched lists

    check_filter : bool
        if True, also check the filter name of each pair

    check_shape : bool
        If True, also check the data shape for each pair.

    check_units : bool
        if True, also check that both inputs are in electrons or ADUs
    """
    log = logutils.get_logger(__name__)

    # Turn inputs into lists for ease of manipulation later
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
            log.fullinfo(f'Checking extension {ext1.id}')

            # Check shape/size
            if check_shape and ext1.data.shape != ext2.data.shape:
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
        xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()

        if this_aux.detector_x_bin() != xbin or this_aux.detector_y_bin() != ybin:
            raise OSError("Auxiliary data {} has different binning to {}".
                          format(os.path.basename(this_aux.filename), ad.filename))

        # Make a new auxiliary file for appending to, starting with PHU
        new_aux = astrodata.create(this_aux.phu)
        new_aux.filename = this_aux.filename
        new_aux.update_filename(suffix="_clipped", strip=False)

        sci_detsec = ad.detector_section()
        sci_datasec = ad.data_section()
        sci_arraysec = ad.array_section()
        clipped_this_ad = False

        for num_ext, (ext, detsec, datasec, arrsec) in enumerate(zip(ad,
                      sci_detsec, sci_datasec, sci_arraysec), start=1):
            datasec, new_datasec = map_data_sections_to_trimmed_data(datasec)
            #new_datasec = new_datasec.view((int, 4))
            ext_shape = ext.shape[-2:]
            #sci_trimmed = np.array_equal(new_datasec, [[0, ext_shape[1], 0, ext_shape[0]]])
            sci_trimmed = new_datasec[0] == Section.from_shape(ext_shape)
            sci_multi_amp = isinstance(arrsec, list)

            for auxext in this_aux:
                aux_detsec = auxext.detector_section()
                #if not at.section_contains(aux_detsec, detsec):
                if not aux_detsec.contains(detsec):
                    continue

                # If this is a perfect match, just do it and carry on
                if auxext.shape == ext_shape:
                    new_aux.append(auxext.nddata)
                    continue

                aux_arrsec = auxext.array_section()
                aux_datasec, aux_new_datasec = map_data_sections_to_trimmed_data(auxext.data_section())
                aux_trimmed = aux_new_datasec[0] == Section.from_shape(auxext.shape)
                #aux_trimmed = np.array_equal(aux_new_datasec, [[0, auxext.shape[1], 0, auxext.shape[0]]])
                aux_multi_amp = isinstance(aux_arrsec, list)

                # Either both are multi-amp or neither is. If they both are,
                # the science can't have more amps (but the aux can).
                if ((sci_multi_amp ^ aux_multi_amp) or
                        (sci_multi_amp and len(arrsec) > len(aux_arrsec))):
                    raise OSError("Problem with array sections for "
                                  f"{ext.filename} and {auxext.filename}")

                # Assumption that the sections returned by array_section()
                # are contiguous so we don't have to do an amp-by-amp match
                if sci_multi_amp:
                    x1 = min(sec.x1 for sec in arrsec)
                    y1 = min(sec.y1 for sec in arrsec)
                    ax1 = min(sec.x1 for sec in aux_arrsec)
                    ay1 = min(sec.y1 for sec in aux_arrsec)
                else:
                    x1, y1 = arrsec.x1, arrsec.y1
                    ax1, ay1 = aux_arrsec.x1, aux_arrsec.y1

                xshift = (x1 - ax1) // xbin
                yshift = (y1 - ay1) // ybin
                #aux_new_datasec = (aux_new_datasec.view((int, 4)) -
                #                   np.array([xshift, xshift, yshift, yshift]))
                aux_new_datasec = [ands.shift(-xshift, -yshift)
                                   for ands in aux_new_datasec]

                shifts = [(ads[0]-ds[0]-ands[0]+nds[0], ads[2]-ds[2]-ands[2]+nds[2])
                          for ds, nds in zip(datasec, new_datasec)
                          for ads, ands in zip(aux_datasec, aux_new_datasec)
                          if ands.contains(nds)]

                # This means we can cut a single section from the aux data
                if len(set(shifts)) == 1:
                    #x1, y1 = shifts[0]
                    #x2, y2 = x1 + ext_shape[1], y1 + ext_shape[0]
                    #if at.section_contains((0, auxext.shape[1], 0, auxext.shape[0]),
                    #                       (x1, x2, y1, y2)):
                    #    new_aux.append(auxext.nddata[y1:y2, x1:x2])
                    cut_sec = Section.from_shape(ext_shape).shift(*shifts[0])
                    if Section.from_shape(auxext.shape).contains(cut_sec):
                        new_aux.append(auxext.nddata[cut_sec.asslice()])
                        clipped_this_ad = True
                        continue

                # So either we need to glue some stuff together, or pad the aux
                # Science decision that may need revisiting
                if aux_trimmed and not sci_trimmed and aux_type != 'bpm':
                    raise OSError(f"Auxiliary data {auxext.filename} is trimmed, but "
                                  f"science data {ext.filename} is untrimmed.")
                nd = astrodata.NDAstroData(np.zeros(ext_shape, dtype=auxext.data.dtype if return_dtype is None else return_dtype),
                                           mask=None if auxext.mask is None else np.zeros(ext_shape, dtype=auxext.mask.dtype),
                                           meta=auxext.nddata.meta)
                if auxext.variance is not None:
                    nd.variance = np.zeros_like(nd.data)
                # Find all overlaps between the "new" data_sections
                for ds, nds in zip(datasec, new_datasec):
                    for ads, ands in zip(aux_datasec, aux_new_datasec):
                        overlap = ands.overlap(nds)
                        if overlap:
                            in_vertices = [a+b-c for a, b, c in zip(overlap, ads, ands)]
                            input_section = (slice(*in_vertices[2:]), slice(*in_vertices[:2]))
                            out_vertices = [a+b-c for a, b, c in zip(overlap, ds, nds)]
                            output_section = (slice(*out_vertices[2:]), slice(*out_vertices[:2]))
                            log.debug("Copying aux section [{}:{},{}:{}] to [{}:{},{}:{}]".
                                      format(in_vertices[0]+1, in_vertices[1], in_vertices[2]+1, in_vertices[3],
                                             out_vertices[0]+1, out_vertices[1], out_vertices[2]+1, out_vertices[3]))
                            nd.set_section(output_section, auxext.nddata[input_section])
                            new_aux.append(nd)

            if len(new_aux) < num_ext:
                raise OSError(f"No auxiliary data in {this_aux.filename} "
                              f"matches the detector section {detsec} in "
                              f"{ad.filename} extension {ext.id}")

        # Coerce the return datatype if required
        if return_dtype:
            for auxext in new_aux:
                if auxext.data.dtype != return_dtype:
                    auxext.data = auxext.data.astype(return_dtype)
                if auxext.variance is not None and auxext.variance.dtype != return_dtype:
                    auxext.variance = auxext.variance.astype(return_dtype)

        if clipped_this_ad:
            log.stdinfo("Clipping {} to match science data.".
                        format(os.path.basename(this_aux.filename)))
        aux_output_list.append(new_aux)

    return aux_output_list

@handle_single_adinput
def clip_auxiliary_data_old(adinput=None, aux=None, aux_type=None,
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
        aux_detsec = this_aux.detector_section()
        aux_datasec = this_aux.data_section()
        aux_arraysec = this_aux.array_section()

        for ext, detsec, datasec, arraysec in zip(ad, sci_detsec, sci_datasec,
                                                  sci_arraysec):

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
                            raise OSError(
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
                new_aux.append(ext_to_clip[0].nddata)

            if not found:
                raise OSError(
                    f"No auxiliary data in {this_aux.filename} matches the "
                    f"detector section {detsec} in {ad.filename} extension "
                    f"{ext.id}"
                )

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
                            raise OSError(
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
                new_aux.append(ext_to_clip[0].nddata)

            if not found:
                raise OSError(
                    f"No auxiliary data in {this_aux.filename} matches the "
                    f"detector section {detsec} in {ad.filename} extension "
                    f"{ext.id}"
                )

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
    single = ad.is_single
    ad_iterable = [ad] if single else ad

    # Produce warning but return what is expected
    if not any([hasattr(ext, 'OBJCAT') for ext in ad_iterable]):
        input = f"{ad.filename} extension {ad.id}" if single else ad.filename
        log.warning(f"No OBJCAT(s) found on {input}. Has detectSources() been run?")
        return Table() if single else [Table()] * len(ad)

    good_sources = []
    for ext in ad_iterable:
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
            table = table[~sigma_clip(table['fwhm_arcsec'], sigma=2, maxiters=1).mask]

        good_sources.append(table)

    return good_sources[0] if single else good_sources

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
            # NIRI cal assoc requires ad.gcal_lamp() to return suitable value
            if ad.instrument() == 'NIRI':
                ad.phu.set("GCALLAMP", "QH", "For calibration association")
                ad.phu.set("GCALSHUT", "OPEN", "For calibration association")

        elif "bpm" in caltype:
            ad.phu.set("BPMASK", True, "Bad pixel mask")
            ad.phu.set("OBSTYPE", "BPM", keyword_comments["OBSTYPE"])
            ad.phu.set("OBJECT", "BPM", keyword_comments["OBJECT"])
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

        # The CRVALi keywords in the extension headers come from the gWCS
        # object, so that needs to be modified
        for ext in ad:
            for m in ext.wcs.forward_transform:
                if isinstance(m, models.RotateNative2Celestial):
                    m.lon = m.lat = 0
                    break
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
def cut_to_match_auxiliary_data(adinput=None, aux=None, aux_type=None,
                        return_dtype=None):
    """
    This function cuts sections of the science frame into extensions to match
    the auxiliary data like calibration files or BPMs based on the value of the
    `detector_section()` keyword in the auxiliary data.

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
        science frame file(s), appropriately cut into multiple extensions
    """
    log = logutils.get_logger(__name__)
    ad_output_list = []

    for ad, this_aux in zip(*make_lists(adinput, aux, force_ad=True)):
        xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()

        if this_aux.detector_x_bin() != xbin or this_aux.detector_y_bin() != ybin:
            raise OSError("Auxiliary data {} has different binning to {}".
                          format(os.path.basename(this_aux.filename), ad.filename))
        cut_this_ad = False

        # Since this is cutting a full frame, should be only one extension.
        for ext in ad:
            # Make a new science file for appending to, starting with PHU
            new_ad = astrodata.create(ad.phu)
            for auxext, detsec in zip(this_aux, this_aux.detector_section()):
                new_ad.append(ext.nddata[detsec.asslice()])
                new_ad[-1].SLITEDGE = auxext.SLITEDGE
                new_ad[-1].hdr[ad._keyword_for('detector_section')] =\
                    detsec.asIRAFsection(binning=(xbin, ybin))
                new_ad[-1].hdr['SPECORDR'] = auxext.hdr['SPECORDR']

                # By default, when a section is cut out of an array the WCS is
                # updated with Shift models corresponding to the number of rows/
                # columns that were removed, so that the WCS of the cut-out
                # section remains the same. For this, however, we want to change
                # the WCS so that in each slit the sky position points at the
                # center of the slit.

                # Get the forward and backward transforms from the input and
                # auxiliary files so we can update the spatial shifts simultaneously
                f_transform = new_ad[-1].wcs.get_transform('pixels', 'world')
                aux_f_transform = auxext.wcs.get_transform('rectified', 'world')

                b_transform = new_ad[-1].wcs.get_transform('world', 'pixels')
                aux_b_transform = auxext.wcs.get_transform('world', 'rectified')

                # In the first or last two submodels of the forward and backward
                # (respectively) transforms will be one or more Shift models.
                # Set these Shifts to 0. It's not part of the WAVE & SKY paradigm
                # so they will be dropped later anyway, but it's good to be
                # consistent at each step.
                # TODO: This works assuming a single Shift in the spatial direction.
                # If there's also a shift in the spectral direction (instead of
                # an Identity, this will need updating.
                for m in range(0, 2, 1):
                    if isinstance(f_transform[m], models.Shift):
                        f_transform[m].offset = 0
                        break

                for n in range(-2, 0, 1):
                    if isinstance(b_transform[n], models.Shift):
                        b_transform[n].offset = 0
                        break

                forward_shift, backward_shift = False, False
                # Now find the offset that sets the middle of the slit to the WCS
                # sky position from the auxiliary file and copy it over.
                for j in range(f_transform.n_submodels):
                    if (isinstance(f_transform[j], models.Shift) and
                        isinstance(f_transform[j+1], models.Const1D)):

                        f_transform[j].offset = aux_f_transform[j].offset
                        forward_shift = True
                        break

                for k in range(b_transform.n_submodels):
                    if (isinstance(b_transform[k], models.Shift) and
                        isinstance(b_transform[k+1], models.Shift)):

                        b_transform[k].offset = aux_b_transform[k].offset
                        backward_shift = True
                        break

            if not (forward_shift or backward_shift):
                raise RuntimeError("No forward or backward shift found in WCS")
            cut_this_ad = True

        if return_dtype:
            for adext in new_ad:
                if adext.data.dtype != return_dtype:
                    adext.data = adext.data.astype(return_dtype)
                if adext.variance is not None and adext.variance.dtype != return_dtype:
                    adext.variance = adext.variance.astype(return_dtype)

        ad_output_list.append(new_ad)
        if cut_this_ad:
            log.stdinfo(f"Cutting {os.path.basename(ad.filename)} to match "
                        "auxiliary data")

    return ad_output_list

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

    find_sources_while_iterating = False
    good_sources = []

    pixel_scale = ad.pixel_scale()
    spatial_box = int(5.0 / pixel_scale)
    MIN_APERTURE_WIDTH = 10  # pixels

    # Initialize the Gaussian width to FWHM = 1.2 arcsec
    init_width = 1.2 / (pixel_scale * (2 * np.sqrt(2 * np.log(2))))

    tags = ad.tags
    acq_star_positions = ad.phu.get("ACQSLITS")

    single = ad.is_single
    ad_iterable = [ad] if single else ad

    for ext in ad_iterable:
        fwhm_list = []
        x_list, y_list = [], []
        weight_list = []

        dispaxis = 2 - ext.dispersion_axis()  # python sense
        dispersed_length = ext.shape[dispaxis]
        spatial_length = ext.shape[1-dispaxis]
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
            hwidth = min(512 // spectral_bin, dispersed_length // 2)
            centers = [i*hwidth for i in range(1, dispersed_length // hwidth, 1)]
        spectral_slices = [slice(center-hwidth, center+hwidth) for center in centers]

        # Try to get an idea of where peaks might be from header information
        # We start with the entire slit length (for N&S, only use the part
        # with both the +ve and -ve beams)
        if 'NODANDSHUFFLE' in tags:
            shuffle = ad.shuffle_pixels() // spatial_bin
            spatial_slice = slice(shuffle, shuffle * 2)
        else:
            spatial_slice = slice(0, spatial_length)

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
                aptable = ext.APERTURE
                for row in aptable:
                    trace_model = astromodels.table_to_model(row)
                    aperture = tracing.Aperture(trace_model,
                                                aper_lower=row['aper_lower'],
                                                aper_upper=row['aper_upper'])
                    spatial_slices.append(aperture)
            except AttributeError:
                # No apertures, so defer source-finding until we iterate
                # over the spectral_slices
                find_sources_while_iterating = True
        else:
            for slit in acq_star_positions.split():
                c, w = [int(x) for x in slit.split(':')]
                spatial_slices.append(slice(c-w, c+w))

        for spectral_slice in spectral_slices:
            coord = 0.5 * (spectral_slice.start + spectral_slice.stop)

            if find_sources_while_iterating:
                _slice = ((spectral_slice, slice(None)) if dispaxis == 0
                          else (slice(None), spectral_slice))
                ndd = ext.nddata[_slice]
                if ext.mask is None:
                    profile = np.percentile(ndd.data, 95, axis=dispaxis)
                else:
                    profile = np.percentile(np.where(ndd.mask == 0, ndd.data, -np.inf),
                                            95, axis=dispaxis)
                center = np.argmax(profile[spatial_slice]) + spatial_slice.start
                spatial_slices = [slice(max(center - spatial_box, 0),
                                        min(center + spatial_box, spatial_length))]

            for spatial_slice in spatial_slices:
                if isinstance(spatial_slice, tracing.Aperture):
                    # convert to a regular slice object
                    limits = spatial_slice.model(coord) + np.array([spatial_slice.aper_lower,
                                                                    spatial_slice.aper_upper])
                    spatial_slice = slice(max(0, int(np.floor(limits[0]))),
                                          min(spatial_length, int(np.ceil(limits[1]))))
                length = spatial_slice.stop - spatial_slice.start

                # General sanity requirement (will reject bad Apertures)
                if length < MIN_APERTURE_WIDTH:
                    continue

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
                if mask is not None:
                    mask = (mask == 0)
                    if mask.sum() < MIN_APERTURE_WIDTH:
                        continue
                try:
                    maxflux = np.max(abs(data[mask]))
                except ValueError:
                    continue

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
                    try:
                        m_final = fit_it(m_init, pixels[mask], data[mask])
                    except:  # anything that goes wrong
                        continue

                if fit_it.fit_info['ierr'] < 5:
                    # This is kind of ugly and empirical; philosophy is that peak should
                    # be away from the edge and should be similar to the maximum
                    if (m_final.amplitude_0 > 0.5*(maxflux-m_final.amplitude_2) and
                        pixels.min()+1 < m_final.mean_0 < pixels.max()-1):
                        fwhm = abs(2 * np.sqrt(2*np.log(2)) * m_final.stddev_0)
                        if fwhm < 1.5:
                            continue
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
            ret_value = at.weighted_sigma_clip(
                table['fwhm_arcsec'].data, weights=table['weight'].data,
                sigma_lower=2, sigma_upper=1.5, maxiters=3)
            table = table[~ret_value.mask]
        good_sources.append(table)

    return good_sources[0] if single else good_sources


def array_from_descriptor_value(ext, descriptor):
    """
    This function tries to return an array or value for a descriptor that
    is suitable for combining with an NDAstroData object. In the simple case
    that the descriptor returns a single value applicable to all pixels in
    the NDAstroData object, that value is returned. However, in a more
    complex case the descriptor will return a list of values corresponding to
    different regions of the image, with the array_section() descriptor
    indicating which region each value corresponds to. In this case, a 2D
    array is constructed where each such region -- after mapping from the
    array_section() to an appropriate region in data_section() -- has the
    appropriate value.

    In the case of a 3D or higher-dimension array, this 2D array is repeated
    along the third (and, if necessary, higher) dimension(s). Furthermore, if
    there are parts of each 2D slice that are not part of data_section(),
    these are assigned values corresponding to the closest pixel.

    Parameters
    ----------
    ext : single-slice AstroData object
        the extension requiring application of a descriptor value
    descriptor : str
        name of descriptor

    Returns
    -------
    float/ndarray : either a single value (if applicable to all pixels) or
                    an ndarray of the same shape as ext.nddata, with
                    appropriate values in each element
    """
    log = logutils.get_logger(__name__)
    if not ext.is_single:
        raise ValueError("Not a single-slice AstroData object")

    desc_value = getattr(ext, descriptor)()
    arrsec = ext.array_section()
    extid = f"{ext.filename}:{ext.id}"

    # Can return a single value if that's all the descriptor gives us, or if
    # array_section() doesn't tell us how to divide the pixel plane
    if hasattr(arrsec, "x1") or not isinstance(desc_value, list):
        if desc_value is None:
            log.warning(f"  {descriptor} for {extid} = None. Setting to zero")
            desc_value = 0.0
        else:  # "electrons" here because this is only used for gain/read_noise
            log.fullinfo(f"  {descriptor} for {extid} = {desc_value} electrons")
        return desc_value

    if len(arrsec) != len(desc_value):
        raise ValueError(f"The number of array sections and {descriptor} "
                         f"values do not match in {extid}")

    data_sections = map_array_sections(ext)
    ret_arr = np.full(ext.shape[-2:], np.nan, dtype=np.float32)

    for datsec, value in zip(data_sections, desc_value):
        ret_arr[datsec.asslice()] = value

    # Pad regions not defined by the array_section() with nearest real value
    indices = distance_transform_edt(np.isnan(ret_arr), return_distances=False,
                                     return_indices=True)

    # Grow along third (and higher) dimension(s) if required
    return np.tile(ret_arr[indices[0], indices[1]], ext.shape[:-2] + (1, 1))


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
    of (normally) AD objects. Each key gets exactly one auxiliary thing from
    each other list -- these lists can be as long as the key_list, or have
    only one item in (in which case they don't have to be lists at all).

    Parameters
    ----------
    args: lists of str/AD (or single str/AD)
        key_list and auxiliary things to be matched to each AD
    kwargs["force_ad"]: bool/tuple
        coerce strings into AD objects? If True, coerce all objects, if a
        tuple/list, then convert only objects in that list (0-indexed)

    Returns
    -------
    tuple of lists
        the lists made from the keys and values
    """
    log = logutils.get_logger(__name__)

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

    if not force_ad:
        return ret_value

    # We only want to open as many AD objects as there are unique entries,
    # so collapse all items in lists to a set and multiple keys with the
    # same value will be assigned references to the same open AD object
    ad_map_dict = {}
    ret_lists = []
    for i, _list in enumerate(ret_value):
        if force_ad is True or i in force_ad:
            for x in set(_list):
                if x not in ad_map_dict:
                    try:
                        ad_map_dict.update({x: astrodata.open(x)
                                            if isinstance(x, str) else x})
                    except OSError:
                        ad_map_dict.update({x: None})
                        log.warning(f"Cannot open file {x}")
            ret_lists.append([ad_map_dict[x] for x in _list])
        else:
            ret_lists.append(_list)
    return ret_lists


def map_array_sections(ext):
    """
    This function determines the actual pixel locations in a single-slice
    AstroData object that the array_section corresponds to. Or, in the case
    of a multi-amp (list) array_section descriptor, the locations for each
    individual Section.

    Parameters
    ----------
    ext : single-slice AstroData object
        the slice whose sections need to be mapped

    Returns
    -------
    Section / list of Sections
        the region(s) in the data that the array_section() corresponds to
    """
    xbin, ybin = ext.detector_x_bin(), ext.detector_y_bin()

    # These return lists, which is correct (it's ad.XXXX_section()
    # that's wrong; it should return a list containing this list)
    datasec = ext.data_section()
    arrsec = ext.array_section(pretty=False)  # pretty required by code

    datsec, new_datsec = map_data_sections_to_trimmed_data(datasec)

    arrsec_is_list = isinstance(arrsec, list)
    sections = []
    xmin = min(asec.x1 for asec in arrsec) if arrsec_is_list else arrsec.x1
    ymin = min(asec.y1 for asec in arrsec) if arrsec_is_list else arrsec.y1
    for asec in (arrsec if arrsec_is_list else [arrsec]):
        sec = Section((asec.x1 - xmin) // xbin, (asec.x2 - xmin) // xbin,
                      (asec.y1 - ymin) // ybin, (asec.y2 - ymin) // ybin)
        for dsec, new_dsec in zip(datsec, new_datsec):
            if new_dsec.contains(sec):
                sections.append(Section(*[a - b + c for a, b, c in
                                          zip(sec, new_dsec, dsec)]))
                break

    return sections if arrsec_is_list else sections[0]


def map_data_sections_to_trimmed_data(datasec):
    """
    This function describes where individual data sections end up in a
    final image after all non-data regions have been removed. It was
    originally intended to work with SCORPIO data when the plan was to have
    overscan regions interior to the four quadrants.

    Parameters
    ----------
    datasec: list of Section tuples
        regions in the pixel plane containing data

    Returns
    -------
    sections: list of Sections
        array of data sections in datasec
    new_sections: list of Sections
        contiguous locations in pixel plane after trimming
    """
    if not isinstance(datasec, list):
        datasec = [datasec]
    sections = np.array(datasec, dtype=[('x1', int), ('x2', int),
                                        ('y1', int), ('y2', int)])
    args = np.argsort(sections, order=('y1', 'x1'))
    new_sections = np.zeros_like(sections)

    for arg in args:
        x1, y1 = sections[arg]['x1'], sections[arg]['y1']
        new_x1 = max(filter(lambda x: x <= x1, new_sections['x2']))
        xshift = x1 - new_x1
        new_y1 = max(filter(lambda y: y <= y1, new_sections['y2']))
        yshift = y1 - new_y1
        new_sections[arg]['x1'] = x1 - xshift
        new_sections[arg]['x2'] = sections[arg]['x2'] - xshift
        new_sections[arg]['y1'] = y1 - yshift
        new_sections[arg]['y2'] = sections[arg]['y2'] - yshift

    return ([Section(*sec) for sec in sections],
            [Section(*sec) for sec in new_sections])


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
    tlm = datetime.utcnow().isoformat()

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
                          separate_ext=True, ignore_mask=False, section=None):
    """
    Return background value, and its std deviation, as measured directly
    from pixels in the SCI image. DQ plane are used (if they exist)
    If extver is set, return a double for that extension only,
    otherwise return a list of doubles, as long as the number of extensions.

    Parameters
    ----------
    ad : AstroData
    sampling : int
        1-in-n sampling factor
    value_only : bool
        if True, return only background values, not the standard deviations
    gaussfit : bool
        if True, fit a Gaussian to the pixel values, instead of sigma-clipping?
    separate_ext : bool
        return information for each extension, rather than the whole AD?
    ignore_mask : bool
        if True, ignore the mask and OBJMASK
    section: slice/None
        region to use for statistics

    Returns
    -------
    list/value/tuple
        if ad is single extension, or separate_ext==False, returns a bg value
        or (bg, std, number of samples) tuple; otherwise returns a list of
        such things
    """
    # Handle NDData objects (or anything with .data and .mask attributes
    maxiter = 10
    try:
        single = ad.is_single
    except AttributeError:
        single = True
    input_list = [ad] if single or not separate_ext else [ext for ext in ad]

    output_list = []
    for ext in input_list:
        flags = None
        # Use DQ and OBJMASK to flag pixels
        if not single and not separate_ext:
            bg_data = np.array([ext.data[section] for ext in ad]).ravel()
            if not ignore_mask:
                flags = np.array([ext.mask | getattr(ext, 'OBJMASK', 0)
                                  if ext.mask is not None
                    else getattr(ext, 'OBJMASK', np.zeros_like(ext.data, dtype=bool))
                                  for ext in ad])[section].ravel()
        else:
            if not ignore_mask:
                flags = ((ext.mask | getattr(ext, 'OBJMASK', 0))[section]
                    if ext.mask is not None else
                         ext.OBJMASK[section] if hasattr(ext, 'OBJMASK') else None)
            bg_data = ext.data[section].ravel()

        if flags is None:
            bg_data = bg_data[::sampling]
        else:
            bg_data = bg_data[flags.ravel() == 0][::sampling]

        if len(bg_data) > 1:
            if gaussfit:
                lsigma, hsigma = 3, 1
                bg, bg_std = sigma_clipped_stats(bg_data, sigma=5, maxiters=5)[1:]
                niter = 0
                while True:
                    iter_data = np.sort(bg_data[np.logical_and(bg-lsigma*bg_std < bg_data,
                                                               bg+hsigma*bg_std > bg_data)][::sampling])
                    oldbg, oldbg_std = bg, bg_std
                    g_init = Ogive(bg, bg_std, lsigma, hsigma)
                    g_init.lsigma.fixed = True
                    g_init.hsigma.fixed = True
                    fit_g = fitting.LevMarLSQFitter()
                    g = fit_g(g_init, iter_data, np.linspace(0., 1., iter_data.size + 1)[1:])
                    bg, bg_std = g.mean.value, abs(g.stddev.value)
                    if abs(bg - oldbg) < 0.001 * bg_std or niter > maxiter:
                        break
                    niter += 1
            else:
                # Sigma-clipping will screw up the stats of course!
                bg_data = sigma_clip(bg_data[::sampling], sigma=2.0, maxiters=2)
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
                    clipped_data = sigma_clip(bg_data, sigma=3.0, maxiters=1)
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


def offsets_relative_to_slit(ext1, ext2):
    """
    Determine the spatial offsets between the pointings of two AstroData
    objects, expressed parallel and perpendicular to the slit axis.
    The issue here is that the world_to_pixel transformation is insensitive
    to coordinate movement perpendicular to the slit. The calculation is
    performed by determining the pointing of one AD from its center of
    projection and the pointing of the other from its WCS evaluated at the
    same pixel location. The position angle of the slit is calculated from
    the sky displacement when offsetting 500 pixels (arbitrary) along the
    slit axis.

    Parameters
    ----------
    ext1, ext2: single-slice AstroData instances
        the two AD objects (assumed to be of the same subclass)

    Returns
    -------
    dist_para: float
        separation (in arcsec) parallel to the slit
    dist_perp: float
        separation (in arcsec) perpendicular to the slit
    """
    wcs1 = ext1.wcs
    try:
        ra1, dec1 = at.get_center_of_projection(wcs1)
    except TypeError:
        raise ValueError(f"Cannot get center of projection for {ext1.filename}")
    dispaxis = 2 - ext1.dispersion_axis()  # python sense
    cenwave, *_ = wcs1(*(0.5 * np.asarray(ext1.shape)[::-1]))
    x, y = wcs1.invert(cenwave, ra1, dec1)

    # Get PA of slit by finding coordinates along the slit
    coord1 = SkyCoord(ra1, dec1, unit='deg')
    ra2, dec2 = wcs1(x, y+500)[-2:] if dispaxis == 1 else wcs1(x+500, y)[-2:]
    pa = coord1.position_angle(SkyCoord(ra2, dec2, unit='deg')).deg

    # Calculate PA and angular distance between sky coords of the same pixel
    # on the two input ADs
    ra2, dec2 = ext2.wcs(x, y)[-2:]
    coord2 = SkyCoord(ra2, dec2, unit='deg')
    return at.spherical_offsets_by_pa(coord1, coord2, pa)


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
    regexp = re.compile(r'(.*)\(\d+\)')
    columns = []
    fp = open(param_file)
    for line in fp:
        fields = line.split()
        if len(fields) == 0:
            continue
        if fields[0].startswith("#"):
            continue
        try:  # Turn FLUX_APER(n) -> FLUX_APER
            name = regexp.match(fields[0]).group(1)
        except AttributeError:
            name = fields[0]
        columns.append(name)
    return columns


# FIXME: unused ?
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
        raise OSError('No database name specified')
    if not os.path.isdir(database_name):
        raise OSError('Database directory {} does not exist'.format(
            database_name))
    if input_name is None:
        input_name = ad.filename
    if output_name is None:
        output_name = ad.filename

    basename = os.path.basename(input_name)
    basename, filetype = os.path.splitext(basename)
    out_basename = os.path.basename(output_name)
    out_basename, filetype = os.path.splitext(out_basename)

    for ext in ad:
        record_name = '{}_{:0.3d}'.format(basename, ext.id)
        db = at.SpectralDatabase(database_name, record_name)
        out_record_name = '{}_{:0.3d}'.format(out_basename, ext.id)
        table = db.as_binary_table(record_name=out_record_name)

        ext.WAVECAL = table
    return ad


def sky_factor(nd1, nd2, skyfunc, multiplicative=False, threshold=0.001):
    """
    This function determines the corrective factor (either additive or
    multiplicative) to apply to a sky frame so that, when subtracted from a
    science frame, the resulting background level is zero. The science
    NDAstroData object is modified, the sky NDAstroData object is returned
    to its original state. A multiplicative correction requires an iterative
    method, which converges once the changes are less than a given fraction
    of the original sky frame.

    Parameters
    ----------
    nd1 : NDAstroData
        the "science" frame
    nd2 : NDAstroData
        the "sky" frame
    skyfunc : callable
        function to determine sky level
    multiplicative : bool
        compute multiplicative rather than additive factor?
    threshold : float
        accuracy of sky subtraction relative to original sky level

    Returns
    -------
    float : factor to apply to "sky" to match "science"
    """
    log = logutils.get_logger(__name__)

    factor = 0
    if multiplicative:
        current_sky = 1.
        # A subtlety here: deepcopy-ing an AD slice will create a full AD
        # object, and so skyfunc() will return a list instead of the single
        # float value we want. So make sure the copy is a single slice too
        if isinstance(nd1, astrodata.AstroData) and nd1.is_single:
            ndcopy = deepcopy(nd1)[0]
        else:
            ndcopy = deepcopy(nd1)
        iter = 1
        max_iter = 100   # normally converges in < 10 iterations
        while abs(current_sky) > threshold and iter <= max_iter:
            f = skyfunc(ndcopy) / skyfunc(nd2)
            ndcopy.subtract(nd2.multiply(f))
            current_sky *= f
            factor += current_sky
            iter += 1
        #print('iter upon exit: ', iter)
        if iter > max_iter:
            log.warning(f"Failed to converge.\n"
                        f"   Reached: {abs(current_sky)} while threshold = {threshold}\n"
                        f"   Final factor = {factor}")
            nd2.divide(current_sky)  # reset to original value
            raise(ConvergenceError)

        nd1.subtract(nd2.multiply(factor / current_sky))
        nd2.divide(factor)  # reset to original value
    else:
        factor = skyfunc(nd1.subtract(nd2))
        nd1.subtract(factor)
    return factor

# FIXME: unused ?
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
        output_wcs = ext.wcs
        indices = [i for i in range(len(ext_mapping))
                   if ext_mapping[i] == ext.id]
        inp_objcats = [adinput[i].OBJCAT for i in indices if
                       hasattr(adinput[i], 'OBJCAT')]

        if inp_objcats:
            out_objcat = vstack(inp_objcats, metadata_conflicts='silent')

            # Get new pixel coords for objects from RA/Dec and the output WCS
            ra = out_objcat["X_WORLD"]
            dec = out_objcat["Y_WORLD"]
            newx, newy = output_wcs(ra, dec)
            out_objcat["X_IMAGE"] = newx + 1
            out_objcat["Y_IMAGE"] = newy + 1

            # Remove the NUMBER column so add_objcat renumbers
            out_objcat.remove_column('NUMBER')

            adoutput = add_objcat(adinput=adoutput, index=ext.id - 1,
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

    for ad in adinput:
        # Get the keyword associated with the data_section descriptor
        datasec_kw = ad._keyword_for('data_section')
        oversec_kw = ad._keyword_for('overscan_section')
        all_datasecs = ad.data_section()

        for ext, datasec in zip(ad, all_datasecs):
            if isinstance(datasec, list):
                # Starting with the bottom-leftmost sections, we shift all
                # the sections as far as they can do down and to the left
                # which should end up producing a contiguous region
                sections, new_sections = map_data_sections_to_trimmed_data(datasec)

                #x1, y1 = sections['x1'].min(), sections['y1'].min()
                #nxpix = new_sections['x2'].max() - new_sections['x1'].min()
                #nypix = new_sections['y2'].max() - new_sections['y1'].min()
                x1 = min(s.x1 for s in sections)
                y1 = min(s.y1 for s in sections)
                nxpix = max(s.x2 for s in new_sections) - min(s.x1 for s in new_sections)
                nypix = max(s.y2 for s in new_sections) - min(s.y1 for s in new_sections)

                old_ext = deepcopy(ext)[0]
                # Trim SCI, VAR, DQ to new section, aligned at bottom-left.
                # This slicing will update the gWCS properly too.
                ext.reset(ext.nddata[y1:y1+nypix, x1:x1+nxpix])
                has_objmask = hasattr(old_ext, 'OBJMASK')
                if has_objmask:
                    ext.OBJMASK = old_ext.OBJMASK[y1:y1+nypix, x1:x1+nxpix]

                for i, (oldsec, newsec) in enumerate(zip(sections, new_sections), start=1):
                    #oldsec, newsec = Section(*oldsec), Section(*newsec)
                    datasecStr = oldsec.asIRAFsection()
                    log.fullinfo(f'For {ad.filename} extension {ext.id}, '
                                 f'keeping the data from the section {datasecStr}')
                    newslice = newsec.asslice()
                    oldslice = oldsec.asslice()
                    ext.nddata.set_section(newslice, old_ext.nddata[oldslice])
                    if has_objmask:
                        ext.OBJMASK[newslice] = old_ext.OBJMASK[oldslice]
                    ext.hdr.set(f'TRIMSEC{i}', datasecStr, comment=keyword_comments['TRIMSEC'])
                del ext.hdr[datasec_kw+'*']

            else:
                # Check whether data need to be trimmed
                x1, x2 = datasec.x1, datasec.x2
                y1, y2 = datasec.y1, datasec.y2
                sci_shape = ext.data.shape
                if x1 == 0 and y1 == 0 and sci_shape == (y2, x2):
                    log.fullinfo(f'No changes will be made to {ad.filename} '
                                 f'extension {ext.id}, since the data section '
                                 f'matches the data shape')
                    continue

                # Update logger with the section being kept
                datasecStr = datasec.asIRAFsection()
                log.fullinfo(f'For {ad.filename} extension {ext.id}, keeping '
                             f'the data from the section {datasecStr}')

                # Trim SCI, VAR, DQ to new section
                ext.reset(ext.nddata[datasec.asslice()])
                # And OBJMASK (if it exists)
                # TODO: should check more generally for any image extensions
                if hasattr(ext, 'OBJMASK'):
                    ext.OBJMASK = ext.OBJMASK[datasec.asslice()]

                # We can't do this unless the data section was contiguous
                ext.hdr.set('TRIMSEC', datasecStr, comment=keyword_comments['TRIMSEC'])

            # Update header keys to match new dimensions
            nypix, nxpix = ext.shape
            newDataSecStr = f'[1:{nxpix},1:{nypix}]'
            ext.hdr.set(datasec_kw, newDataSecStr, comment=keyword_comments.get(datasec_kw))
            if oversec_kw in ext.hdr:
                del ext.hdr[oversec_kw]

    return adinput

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
    basename, filetype = os.path.splitext(basename)

    for ext in ad:
        record_name = '{}_{:0.3d}'.format(basename, ext.id)
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

    def __init__(self, adinputs, fields_overlap=None, frac_FOV=1.0):
        """
        Parameters
        ----------
        adinputs: AstroData/list of AD objects
            exposure list from which to initialize the group
            (currently may not be empty)

        :param pkg: Package name of the

        :param frac_FOV: proportion by which to scale the area in which
            points are considered to be within the same field, for tweaking
            the results in borderline cases (eg. to avoid co-adding target
            positions right at the edge of the field).
        :type frac_FOV: float
        """
        if not isinstance(adinputs, list):
            adinputs = [adinputs]
        # Make sure the field scaling is valid:
        if not isinstance(frac_FOV, numbers.Number) or frac_FOV < 0.:
            raise ValueError('frac_FOV must be >= 0.')
        if not callable(fields_overlap):
            raise ValueError('fields_overlap must be callable')

        # Initialize members:
        self.members = []
        self._frac_FOV = frac_FOV
        self.group_center = None
        self.add_members(adinputs)
        self._fields_overlap = fields_overlap

    def fields_overlap(self, ad):
        """
        Determine whether or not a new pointing falls within this group.
        The check can be done against either the group center, or against
        all members of the group.

        Parameters
        ----------
        ad: AstroData instance
            the AD object to be tested
        fast: bool
            check against the field center only? (or else, check for overlap
            with all group members)

        Returns
        -------
        bool: whether or not the input point is within the field of view
            (adjusted by the frac_FOV specified when creating the group).
        """
        for ad_in_group in self.members:
            if self._fields_overlap(ad, ad_in_group, frac_FOV=self._frac_FOV):
                return True
        return False

    def __len__(self):
        return len(self.members)

    def __repr__(self):
        return str(self.list())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # A Python dictionary equality seems to take care of comparing
            # groups nicely, irrespective of ordering:
            return self.members == other.members
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, item):
        return item in self.members

    def list(self):
        return self.members

    def add_members(self, adinputs):
        """
        Add one or more new points to the group.

        :param adinputs: A list of AstroData instances to add to the group
            membership.
        :type adinputs: AstroData, list of AstroData instances
        """
        if not isinstance(adinputs, list):
            adinputs = [adinputs]
        # How many points were there previously and will there be now?
        for ad in adinputs:
            if ad not in self.members:
                self.members.append(ad)
                ad_coord = SkyCoord(ad.ra(), ad.dec(), unit='deg')
                if self.group_center:
                    separation = self.group_center.separation(ad_coord)
                    pa = self.group_center.position_angle(ad_coord)
                    # We move the group center fractionally towards the new
                    # position
                    self.group_center = self.group_center.directional_offset_by(
                        pa, separation / len(self))
                else:
                    self.group_center = ad_coord

def group_exposures(adinputs, fields_overlap=None, frac_FOV=1.0):
    """
    Sort a list of AstroData instances into dither groups around common
    nod positions, according to their pointing offsets.

    In principle divisive clustering algorithms have the best chance of
    robust separation because of their top-down view of the problem.
    However, an agglomerative algorithm is probably more practical to
    implement here in the first instance, given that we can try to
    constrain the problem using the known field size and remembering the
    one-at-a-time case where the full list isn't available up front.

    The FOV gives us a pretty good idea what's close enough to the base
    position to be considered on-source, but it may not be enough on its
    own to provide a threshold for intermediate nod distances for F2 given
    IQ problems towards the edges of the field. OTOH intermediate offsets
    are generally difficult to achieve anyway due to guide probe limits
    when the instrumental FOV is comparable to that of the telescope.

    The algorithm has been improved (but made slightly slower) to be
    independent of the order in which files arrive. If a pointing overlaps
    with two (or more) existing groups, it will unify these groups.

    Parameters
    ----------
    adinputs: list of AD instances
        Exposures to sort into groups

    pkg: str
        Name of the module containing the instrument FOV lookup.
        Passed through to ExposureGroup() call.

    frac_FOV: float
        proportion by which to scale the area in which points are considered
        to be within the same field, for tweaking the results in borderline
        cases (eg. to avoid co-adding target positions right at the edge of
        the field). frac_FOV=1.0 means *any* overlap is OK.

    Returns
    -------
    tuple: of ExposureGroup instances, one per distinct pointing position
    """
    groups = []

    for ad in adinputs:
        found = None
        for i, group in reversed(list(enumerate(groups))):
            if group.fields_overlap(ad):
                if found is None:
                    group.add_members(ad)
                else:
                    group.add_members(groups[found].list())
                    del groups[found]
                found = i
        if found is None:
            groups.append(ExposureGroup(ad, fields_overlap=fields_overlap,
                                        frac_FOV=frac_FOV))

    # Here this simple algorithm could be made more robust for borderline
    # spacing (a bit smaller than the field size) by merging clusters that
    # have members within than the threshold after initial agglomeration.
    # This is an odd use case that's hard to classify automatically anyway
    # but the grouping info. might be more useful later, when stacking.
    return tuple(groups)
