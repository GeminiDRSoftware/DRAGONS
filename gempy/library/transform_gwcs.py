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
from astropy import units as u

from gwcs.coordinate_frames import Frame2D
from gwcs.wcs import WCS as gWCS

from functools import reduce

from gempy.gemini import gemini_tools as gt
from ..utils import logutils

from . import transform
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
            model_list.append(models.Shift(float(offset.x1) / xbin * u.pix) &
                              models.Shift(float(offset.y1) / ybin * u.pix))

        if rot != 0 or mag != (1, 1):
            # Shift to centre, do whatever, and then shift back
            model_list.append(models.Shift(-0.5 * (nx-1) * u.pix) &
                              models.Shift(-0.5 * (ny-1) * u.pix))
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
            model_list.append(models.Shift(0.5 * (nx-1) * u.pix) &
                              models.Shift(0.5 * (ny-1) * u.pix))
        model_list.append(models.Shift(float(shift[0]) / xbin * u.pix) &
                          models.Shift(float(shift[1]) / ybin * u.pix))

        first_section = ad[indices[0]].array_section()
        in_frame = Frame2D(name="pixels", axes_names=("x", "y"),
                           unit=(u.pix, u.pix))
        out_frame = Frame2D(name="mosaic", axes_names=("x", "y"),
                            unit=(u.pix, u.pix))
        for i in indices:
            arrsec = ad[i].array_section()
            datsec = ad[i].data_section()
            ext_shift = (models.Shift(((arrsec.x1 - first_section.x1) // xbin - datsec.x1) * u.pix) &
                         models.Shift(((arrsec.y1 - first_section.y1) // ybin - datsec.y1) * u.pix))
            mosaic_model = reduce(Model.__or__, [ext_shift] + model_list)
            ad[i].wcs = gWCS([(in_frame, mosaic_model),
                              (out_frame, None)])

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


def _remove_pixel_units(model):
    """

    Parameters
    ----------
    model: astropy.modeling.Model instance (may be a CompoundModel)

    Returns
    -------
    Model: the input model with all Quantity objects converted to scalars
    """
    for param in model.param_names:
        getattr(model, param)._unit = None
    if isinstance(model.inverse, Model):
        for param in model.inverse.param_names:
            getattr(model.inverse, param)._unit = None
    return model




def resample_from_wcs(inputs, frame):
    """

    Parameters
    ----------
    inputs: iterable of NDData/AstroData/Block-like objects
        the arrays that are being transformed onto a single new pixel plane
    frame: str
        name of the frame to resample to

    Returns
    -------
    AstroData: single AD with suitable WCS
    """
    # Let's check this is a valid frame before we do anything
    try:
        indices = [[step[0].name for step in item.wcs.pipeline].index(frame)
                   for item in inputs]
    except ValueError:
        raise ValueError("Cannot find {} in all inputs".format(frame))

    adg = transform.AstroDataGroup()

    for item, index in zip(inputs, indices):
        t = transform.Transform([_remove_pixel_units(step[1])
                                 for step in item.wcs.pipeline[:index]])
        adg.append(item, t)

    return adg