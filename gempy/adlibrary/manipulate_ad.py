import numpy as np

import astrodata
from astrodata import wcs as adwcs
from gempy.utils import logutils

from geminidr.gemini.lookups import DQ_definitions as DQ


def reassemble_ad(adinput, shape=None):
    """
    This takes an AstroData obejct with multiple extensions and reassembles
    the data into a single extension, using the array_section descriptors to
    place the data in the correct location. Pixels masked as either NO_DATA
    or UNILLUMINATED are not added to the final image and so will be left as
    zero.

    The header is copied from the first extension, but the various section
    keywords will be incorrect. This is not considered to be a major issue
    since the calling code will know that a full image has been created.

    Parameters
    ----------
    adinput: AstroData
        input AD object with multiple extensions
    shape: tuple/None
        shape of the output array, if None, the shape is determined from the
        array_section descriptors

    Returns
    -------
        AstroData: object of same subclass as input, with one extension
    """
    array_sections = adinput.array_section()

    # Regions not covered by any of the input extensions will be masked
    # as unilluminated.
    covered_shape = (max(arrsec.y2 for arrsec in array_sections),
                     max(arrsec.x2 for arrsec in array_sections))
    if shape is None:
        shape = covered_shape

    adout = astrodata.create(adinput.phu)
    data = np.zeros(shape, dtype=adinput[0].data.dtype)

    # Always create a mask because there may be regions between the input
    # extensions that need to be flagged as unilluminated.
    mask = np.full(shape, DQ.unilluminated)
    try:
        variance = np.zeros(shape, dtype=adinput[0].variance.dtype)
    except AttributeError:
        variance = None

    # Rather than create len(adinput) fullsize images and collapse them,
    # we implement a two-pass approach which requires us to go back and
    # unmasked regions that will have been masked by subsequent extensions.
    for ext, arrsec in zip(adinput, array_sections):
        _slice = arrsec.asslice()
        if ext.mask is None:
            data[_slice] += ext.data
            if variance is not None:
                variance[_slice] += ext.variance
            mask[_slice] = DQ.good
        else:
            illuminated = ext.mask & (DQ.no_data | DQ.unilluminated) == 0
            data[_slice][illuminated] += ext.data[illuminated]
            mask[_slice] |= ext.mask
            if variance is not None:
                variance[_slice][illuminated] += ext.variance[illuminated]

    for ext, arrsec in zip(adinput, array_sections):
        if ext.mask is not None:
            _slice = arrsec.asslice()
            illuminated = ext.mask & (DQ.no_data | DQ.unilluminated) == 0
            mask[_slice][illuminated] = ext.mask[illuminated]

    adout.append(adinput[0].nddata.__class__(data=data, mask=mask, variance=variance,
                                             meta=adinput[0].nddata.meta))
    return adout


def rebin_data(adinput, xbin=1, ybin=1, patch_binning_descriptors=True):
    """
    The pixel data in each extension of the input AstroData object are
    rebinned according to requirements. The "data_section" keyword is
    updated, and the "detector_x_bin" and "detector_y_bin" descriptors
    are modified to return the new binning but otherwise, headers and
    the WCS are left intact. This is done in-place.

    Parameters
    ----------
    adinput: AstroData
        input AD object
    xbin: int
        new binning in x-direction
    ybin: int
        new binning in y-direction
    patch_binning_descriptors: bool
        override the descriptor functions to return the new binning?
        (if False, the calling function should probably set the header keywords)

    Returns
    -------
        AstroData: modified object of same subclass as input
    """
    log = logutils.get_logger(__name__)

    core_attributes = ['data', 'mask', 'variance']

    xrebin = xbin // adinput.detector_x_bin()
    yrebin = ybin // adinput.detector_y_bin()
    if xrebin < 1 or yrebin < 1:
        raise ValueError('Attempt to rebin to less coarse binning')
    elif xrebin * yrebin == 1:
        log.stdinfo(f"{adinput.filename} does not need rebinning")
        return adinput
    elif (xbin % adinput.detector_x_bin() > 0 or
          ybin % adinput.detector_y_bin() > 0):
        raise ValueError('New binning must be a multiple of the current binning')

    log.stdinfo(f"Rebinning {adinput.filename}")
    for ext, datsec in zip(adinput, adinput.data_section()):
        ext_shape = ext.shape
        extid = f"{adinput.filename}:{ext.id}"
        if len(ext_shape) != 2:
            log.warning(f"Cannot rebin {extid} with {len(ext_shape)} dimensions")
            continue

        if ext.shape[1] % yrebin > 0 or ext.shape[0] % xrebin > 0:
            raise ValueError(f"Cannot rebin {extid} of shape {ext_shape} to "
                             f"{xbin}x{ybin}")

        for attr in (core_attributes + list(ext.nddata.meta['other'].keys())):
            data = getattr(ext, attr, None)
            try:
                rebin_this = data.shape == ext_shape
            except AttributeError:
                log.debug(f"Not rebinning {extid} {attr} as it has a "
                          "different shape")
                continue
            if rebin_this:
                binned_data = data.reshape(ext_shape[0] // yrebin, yrebin,
                                           ext_shape[1] // xrebin, xrebin)
                # We don't do this based on datatype as raw or lightly-
                # processed data could still be of integer type
                if "mask" in attr.lower() or 'BPM' in adinput.tags:
                    binned_data = np.bitwise_or.reduce(
                        np.bitwise_or.reduce(binned_data, axis=1), axis=2)
                else:
                    binned_data = binned_data.sum(axis=1).sum(axis=2)
            setattr(ext, attr, binned_data)

        # Update keyword for data_section()
        ext.hdr[adinput._keyword_for('data_section')] = \
            (f"[{datsec.x1 // xrebin + 1}:{datsec.x2 // xrebin},"
             f"{datsec.y1 // yrebin + 1}:{datsec.y2 // yrebin}]")

    # Ensure binning descriptors return the correct values
    if patch_binning_descriptors:
        setattr(adinput, 'detector_x_bin', lambda: xbin)
        setattr(adinput, 'detector_y_bin', lambda: ybin)
        # gt.check_inputs_match() calls the binning descriptors on slices
        # and we can't fix that here, so patch CCDSUM (if it exists).
        # Another solution will need to be found for other ways of getting
        # the binning
        for ext in adinput:
            if 'CCDSUM' in ext.hdr:
                ext.hdr['CCDSUM'] = f"{xbin} {ybin}"

    return adinput


def remove_single_length_dimension(adinput):
    """
    If there is only one single length dimension in the pixel data, the
    remove_single_length_dimension function will remove the single length
    dimension. In addition, this function removes any keywords associated with
    that dimension. Used ONLY by the standardizeStructure primitive in
    primitives_F2.py.

    Parameters
    ----------
    adinput : AstroData
        input AD object

    Returns
    -------
        AstroData: modified object of same subclass as input
    """

    log = logutils.get_logger(__name__)

    for ext in adinput:
        # Ensure that there is only one single length dimension in the pixel
        # data
        shape = ext.shape
        extid = f"{adinput.filename}:{ext.id}"
        if shape.count(1) == 1:
            axis = shape.index(1)

            # Dimension in user-friendly format (1=x, 2=y, etc.)
            dimension = ext.data.ndim - axis

            log.debug(f"Removing dimension {dimension} from {extid}")
            _slice = tuple(0 if i == axis else slice(None)
                           for i, _ in enumerate(shape))
            ext.reset(ext.nddata[_slice])
            try:
                adwcs.remove_unused_world_axis(ext)
            except:
                log.debug(f"Cannot remove world axis from {extid}")
        else:
            log.warning("No dimension of length 1 in extension pixel data."
                        f"No changes will be made to {extid}")

    return adinput
