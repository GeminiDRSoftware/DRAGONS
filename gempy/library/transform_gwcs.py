"""
This is a temporary module with utility functions to deal with the manipulation
of AstroData objects in accordance with their GWCS attributes.

Initially, we are aiming to leave processing of IMAGEs alone so this will focus
on SPECT observations and it is cleaner to separate these, but ultimately this
will merge with the relevant parts of the current transform.py

Some parts may migrate elsewhere in the codebase
"""
import astrodata
import numpy as np

from astropy.modeling import models, Model
from astropy import table, units as u

from gwcs.coordinate_frames import Frame2D
from gwcs.wcs import WCS as gWCS

from functools import reduce

from geminidr.gemini.lookups import DQ_definitions as DQ

from gempy.gemini import gemini_tools as gt
from ..utils import logutils

from .transform import Block, Transform, DataGroup

log= logutils.get_logger(__name__)

# Table attribute names that should be modified to represent the
# coordinates in the Block, not their individual arrays.
# NB. Standard python ordering!
catalog_coordinate_columns = {'OBJCAT': (['Y_IMAGE'], ['X_IMAGE'])}

#-----------------------------------------------------------------------------

def add_mosaic_wcs(ad, geotable):
    """


    Parameters
    ----------
    ad: AstroData
        the astrodata instance

    Returns
    -------
    AstroData: the modified input AD, with WCS attributes
    """
    array_info = gt.array_information(ad)
    offsets = [ad[exts[0]].array_section()
               for exts in array_info.extensions]

    detname = ad.detector_name()
    xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
    geometry = geotable.geometry[detname]
    default_shape = geometry.get('default_shape')

    for indices, origin, offset in zip(array_info.extensions, array_info.origins, offsets):
        # Origins are in (x, y) order in LUT
        block_geom = geometry[origin[::-1]]
        nx, ny = block_geom.get('shape', default_shape)
        nx /= xbin
        ny /= ybin
        shift = block_geom.get('shift', (0, 0))
        rot = block_geom.get('rotation', 0.)
        mag = block_geom.get('magnification', (1, 1))

        model_list = []

        # Shift the Block's coordinates based on its location within
        # the full array, to ensure any rotation takes place around
        # the true centre.
        if offset.x1 != 0 or offset.y1 != 0:
            model_list.append(models.Shift(offset.x1 / xbin) &
                              models.Shift(offset.y1 / ybin))

        if rot != 0 or mag != (1, 1):
            # Shift to centre, do whatever, and then shift back
            model_list.append(models.Shift(-0.5 * (nx-1)) &
                              models.Shift(-0.5 * (ny-1)))
            if rot != 0:
                # Cope with non-square pixels by scaling in one
                # direction to make them square before applying the
                # rotation, and then reversing that.
                if xbin != ybin:
                    model_list.append(models.Identity(1) & models.Scale(ybin / xbin))
                model_list.append(models.Rotation2D(rot))
                if xbin != ybin:
                    model_list.append(models.Identity(1) & models.Scale(xbin / ybin))
            if mag != (1, 1):
                model_list.append(models.Scale(mag[0]) &
                                  models.Scale(mag[1]))
            model_list.append(models.Shift(0.5 * (nx-1)) &
                              models.Shift(0.5 * (ny-1)))
        model_list.append(models.Shift(shift[0] / xbin) &
                          models.Shift(shift[1] / ybin))
        mosaic_model = reduce(Model.__or__, model_list)

        first_section = ad[indices[0]].array_section()
        in_frame = Frame2D(name="pixels")
        tiled_frame = Frame2D(name="tile")
        mos_frame = Frame2D(name="mosaic")
        for i in indices:
            arrsec = ad[i].array_section()
            datsec = ad[i].data_section()
            if len(indices) > 1:
                #ext_shift = (models.Shift(((arrsec.x1 - first_section.x1) // xbin - datsec.x1)) &
                #             models.Shift(((arrsec.y1 - first_section.y1) // ybin - datsec.y1)))
                ext_shift = (models.Shift((arrsec.x1 // xbin - datsec.x1)) &
                             models.Shift((arrsec.y1 // ybin - datsec.y1)))
                pipeline = [(in_frame, ext_shift),
                            (tiled_frame, mosaic_model),
                            (mos_frame, None)]
            else:
                pipeline = [(in_frame, mosaic_model),
                            (mos_frame, None)]
            ad[i].wcs = gWCS(pipeline)

    return ad

def add_spectroscopic_wcs(ad):
    """
    Attach a GWCS object to all extensions of an AstroData objects,
    representing the approximate spectroscopic WCS, as returned by
    the descriptors.

    Parameters
    ----------
    ad: AstroData
        the astrodata instance

    Returns
    -------
    AstroData: the modified input AD, with WCS attributes
    """
    if 'SPECT' not in ad.tags:
        raise ValueError("Image {} not of type SPECT".format(ad.filename))

    cenwave = ad.central_wavelength(asNanometers=True)
    dw = np.mean(ad.dispersion(asNanometers=True))
    pixscale = ad.pixel_scale() * u.arcsec / u.pix

    linear_wavelength_model = models.Multiply(dw * u.nm / u.pix) | models.Shift(cenwave * u.nm)
    in_frame = Frame2D(name="pixels", axes_names=("x", "y"),
                       unit=(u.pix, u.pix))

    if len(ad) == 1:
        ext = ad[0]
        recenter_model = models.Shift(0.5*(1-ext.shape[1]) * u.pix) & models.Shift(0.5*(1-ext.shape[0]) * u.pix)
        dispaxis = 2 - ext.dispersion_axis()  # python sense
        if dispaxis == 0:
            center_to_world_model = models.Multiply(pixscale) & linear_wavelength_model
            out_frame = Frame2D(name="world", axes_names=("slit", "lambda"),
                                unit=(u.arcsec, u.nm))
        else:
            center_to_world_model = linear_wavelength_model & models.Multiply(pixscale)
            out_frame = Frame2D(name="world", axes_names=("lambda", "slit"),
                                unit=(u.nm, u.arcsec))
        ext.wcs = gWCS([(in_frame, recenter_model | center_to_world_model),
                        (out_frame, None)])
        return ad

    return ad



def resample_from_wcs(ad, frame_name, attributes=None, order=1, subsample=1,
                      threshold=0.01, conserve=False, parallel=False,
                      process_objcat=False):
    """
    This takes a single AstroData object with WCS objects attached to the
    extensions and applies some part of the WCS to resample into a
    different pixel coordinate frame.

    This effectively replaces AstroDataGroup since now the transforms are
    part of the AstroData object.

    Parameters
    ----------
    ad: AstroData
        the input image that is going to be resampled/mosaicked
    frame_name: str
        name of the frame to resample to
    attributes: list/None
        list of attributes to resample (None => all standard ones that exist)
    order: int
        order of interpolation
    subsample: int
        if >1, will transform onto finer pixel grid and block-average down
    threshold: float
        for transforming the DQ plane, output pixels larger than this value
        will be flagged as "bad"
    conserve: bool
        conserve flux rather than interpolate?
    parallel: bool
        use parallel processing to speed up operation?
    process_objcat: bool
        merge input OBJCATs into output AD instance?

    Returns
    -------
    AstroData: single-extension AD with suitable WCS
    """
    array_attributes = ['data', 'mask', 'variance', 'OBJMASK']

    # It's not clear how much checking we should do here but at a minimum
    # we should probably confirm that each extension is purely data. It's
    # up to a primitve to catch this, call trim_to_data_section(), and try again
    if not all(((datsec.x1, datsec.y1) == (0, 0) and (datsec.y2, datsec.x2) == shape)
               for datsec, shape in zip(ad.data_section(), ad.shape)):
        raise ValueError("Not all data sections agree with shapes")

    # Create the blocks (individual physical detectors)
    array_info = gt.array_information(ad)
    blocks = [Block(ad[arrays], shape=shape) for arrays, shape in
              zip(array_info.extensions, array_info.array_shapes)]

    dg = DataGroup()
    dg.no_data['mask'] = DQ.no_data

    for block in blocks:
        wcs = block.wcs
        # Do some checks that, e.g., this is a pixel->pixel mapping
        try:  # Create more informative exceptions
            frame_index = wcs.available_frames.index(frame_name)
        except AttributeError:
            raise TypeError("WCS attribute is not a WCS object on {}"
                            "".format(ad.filename))
        except ValueError:
            raise ValueError("Frame {} is not in WCS for {}"
                             "".format(frame_name, ad.filename))

        frame = getattr(wcs, frame_name)
        if not all(au == u.pix for au in frame.unit):
            raise ValueError("Requested output frame is not a pixel frame")

        transform = Transform(wcs.get_transform(wcs.input_frame, frame))
        dg.append(block, transform)

    if attributes is None:
        attributes = [attr for attr in array_attributes
                      if all(getattr(ext, attr, None) is not None for ext in ad)]
    if 'data' not in attributes:
        log.warning("The 'data' attribute is not specified. Adding to list.")
        attributes += ['data']
    log.fullinfo("Processing the following array attributes: "
                 "{}".format(', '.join(attributes)))
    dg.transform(attributes=attributes, order=order, subsample=subsample,
                 threshold=threshold, conserve=conserve, parallel=parallel)

    ad_out = astrodata.create(ad.phu)
    ad_out.orig_filename = ad.orig_filename

    # TODO: May need to revisit "reference extension"
    ad_out.append(dg.output_dict['data'], header=ad[0].hdr.copy())
    for key, value in dg.output_dict.items():
        if key != 'data':  # already done this
            setattr(ad_out[0], key, value)

    # Create a new gWCS object describing the remaining transformation.
    ndim = len(ad[0].shape)
    new_pipeline = wcs.pipeline[frame_index:]
    if len(new_pipeline) == 1:
        new_wcs = None
    else:
        new_wcs = gWCS(new_pipeline)
    ad[0].wcs = new_wcs

    # Update and delete keywords from extension (_update_headers)
    if ndim != 2:
        log.warning("The updating of header keywords has only been "
                    "fully tested for 2D data.")
    header = ad[0].hdr
    keywords = {sec: ad._keyword_for('{}_section'.format(sec))
                for sec in ('array', 'data', 'detector')}
    # Data section probably has meaning even if ndim!=2
    ad.hdr[keywords['data']] = '['+','.join('1:{}'.format(length)
                                for length in ad[0].shape[::-1])+']'
    # These descriptor returns are unclear in non-2D data
    if ndim == 2:
        # If detector_section returned something, set an appropriate value
        all_detsec = np.array([ext.detector_section() for ext in ad]).T
        ad.hdr[keywords['detector']] = '['+','.join('{}:{}'.format(min(c1)+1, max(c2))
                            for c1, c2 in zip(all_detsec[::2], all_detsec[1::2]))+']'
        # array_section only has meaning now if the inputs were from a
        # single physical array
        if len(blocks) == 1:
            all_arrsec = np.array([ext.array_section() for ext in ad]).T
            ad.hdr[keywords['array']] = '[' + ','.join('{}:{}'.format(min(c1) + 1, max(c2))
                                for c1, c2 in zip(all_arrsec[::2], all_arrsec[1::2])) + ']'
        else:
            del ad.hdr[keywords['array']]
    if 'CCDNAME' in ad[0].hdr:
        ad.hdr['CCDNAME'] = ad.detector_name()
    # Finally, delete any keywords that no longer make sense
    for kw in ('AMPNAME', 'FRMNAME', 'FRAMEID', 'CCDSIZE', 'BIASSEC',
               'DATATYP', 'OVERSEC', 'TRIMSEC', 'OVERSCAN', 'OVERRMS'):
        if kw in header:
            del ad.hdr[kw]

    # Now let's worry about the tables. Top-level ones first
    for table_name in ad.tables:
        setattr(ad_out, table_name, getattr(ad, table_name).copy())
        log.fullinfo("Copying {}".format(table_name))

    # And now the OBJCATs. This code is partially set-up for other types of
    # tables, but it's likely each will need some specific tweaks.
    if 'IMAGE' in ad.tags and process_objcat:
        for table_name, coord_columns in catalog_coordinate_columns.items():
            tables = []
            for block in blocks:
                # This returns a single Table
                cat_table = getattr(block, table_name, None)
                if cat_table is None:
                    continue
                if 'header' in cat_table.meta:
                    del cat_table.meta['header']
                for ycolumn, xcolumn in zip(*coord_columns):
                    # OBJCAT coordinates are 1-indexed
                    newx, newy = transform(cat_table[xcolumn]-1, cat_table[ycolumn]-1)
                    cat_table[xcolumn] = newx + 1
                    cat_table[ycolumn] = newy + 1
                if new_wcs is not None:
                    ra, dec = new_wcs(cat_table['X_IMAGE']-1, cat_table['Y_IMAGE']-1)
                cat_table["X_WORLD"][:] = ra
                cat_table["Y_WORLD"][:] = dec
                tables.append(cat_table)

        if tables:
            log.stdinfo("Processing {}s".format(table_name))
            objcat = table.vstack(tables, metadata_conflicts='silent')
            objcat['NUMBER'] = np.arange(len(objcat)) + 1
            setattr(ad_out[0], table_name, cat_table)

    return ad_out