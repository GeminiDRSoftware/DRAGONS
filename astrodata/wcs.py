import numpy as np

from functools import reduce
from gwcs import coordinate_frames as cf
from gwcs.utils import make_fitswcs_transform
from astropy import units as u, coordinates as coord
from astropy.modeling import models, projections, Model

from gwcs.wcs import WCS as gWCS

from collections import namedtuple
AffineMatrices = namedtuple("AffineMatrices", "matrix offset")

#-----------------------------------------------------------------------------
# FITS-WCS -> gWCS
#-----------------------------------------------------------------------------

def fitswcs_to_gwcs(header):
    """
    Create and return a gWCS object from a FITS header.
    Currently the gWCS function here only supports imaging frames,
    but that's OK for now because Gemini raw data always has an imaging WCS
    """
    in_frame = cf.Frame2D(name="pixels")

    # Check whether the CTYPE keywords indicate it's an image
    if [ctype[:4] for ctype in sorted(header['CTYPE*'].values())] == ['DEC-', 'RA--']:
        frame_name = header.get('RADESYS') or header.get('RADECSYS')  # FK5, etc.
        try:
            ref_frame = getattr(coord, frame_name)()
            # TODO? Work out how to stuff EQUINOX and OBS-TIME into the frame
        except (AttributeError, TypeError):
            ref_frame = None
        out_frame = cf.CelestialFrame(reference_frame=ref_frame, name="world")
        transform = make_fitswcs_transform(header)
        return gWCS([(in_frame, transform),
                     (out_frame, None)])
    else:
        units = tuple(header['CUNIT*'].values())
        crpix_shift = reduce(Model.__and__, [models.Shift(1-crpix)
                                             for crpix in tuple(header['CRPIX*'].values())])

#-----------------------------------------------------------------------------
# gWCS -> FITS-WCS
#-----------------------------------------------------------------------------


def gwcs_to_fits(ndd, hdr):
    if ndd.wcs.output_frame.unit == (u.deg, u.deg):
        wcs_dict = gwcs_to_fits_image(ndd)
    else:
        wcs_dict = gwcs_to_fits_spect(ndd, hdr.get('CENTWAVE'))
    return wcs_dict

def gwcs_to_fits_image(ndd):
    """
    Convert an imaging gWCS object to a series of FITS WCS keywords that can
    be inserted into the header so other software can understand the WCS. If
    this can't be done, a ValueError should be raised. If the FITS WCS is
    only approximate, this should be indicated with a dict entry
    {'FITS-WCS': 'APPROXIMATE'}

    Parameters
    ----------
    ndd: NDAstroData object

    Returns
    -------
    dict: values to insert into the FITS header to express this WCS
    """
    wcs_dict = {'RADESYS': 'FK5'}  # Need a default
    wcs = ndd.wcs
    wcs_model = wcs.forward_transform
    if (isinstance(wcs_model[-1], models.RotateNative2Celestial) and
            isinstance(wcs_model[-2], models.Pix2SkyProjection)):
        # Determine which sort of projection this is
        for projcode in projections.projcodes:
            if isinstance(wcs_model[-2], getattr(models, 'Pix2Sky_{}'.format(projcode))):
                break
        else:
            raise ValueError("Unknown projection class: {}".format(wcs_model[-2].__class__.__name__))

        wcs_dict['CTYPE1'] = 'RA---{}'.format(projcode)
        wcs_dict['CTYPE2'] = 'DEC--{}'.format(projcode)
        crval = (wcs_model[-1].lon.value, wcs_model[-1].lat.value)
        wcs_dict['CRVAL1'], wcs_dict['CRVAL2'] = crval

        # Now the pre-projection part. There's an issue here in the way
        # astropy.modeling allows models to be sliced: each step in the gWCS
        # pipeline is a CompoundModel, and we can't slice the entire pipeline
        # in a way that also slices one of these steps. So we need to slice
        # the final step and then append it to any previous steps.
        # In the worst case scenario, where this slicing fails, we construct
        # the affine part of the transformation by inverting the projection
        # and CelestialRotation steps and appending them to the end.
        # This produces errors < 1 nanoarcsecond.
        penultimate_frame = wcs.available_frames[-2]
        pre_model = wcs.get_transform(wcs.input_frame, penultimate_frame)
        if pre_model is None:
            try:
                affine_model = wcs_model[:-2]
            except IndexError:
                affine_model = wcs_model | wcs_model[-1].inverse | wcs_model[-2].inverse
        else:
            try:
                affine_model = wcs.get_transform(penultimate_frame, wcs.output_frame)[:-2]
            except IndexError:
                affine_model = wcs.get_transform(penultimate_frame, wcs.output_frame)
                affine_model |= (affine_model[-1].inverse | affine_model[-2].inverse)
            affine_model = pre_model | affine_model
        affine = calculate_affine_matrices(affine_model, ndd.shape)
        wcs_dict.update({f'CD{i}_{j}': affine.matrix[j-1, i-1] for i in (1, 2) for j in (1, 2)})

        wcs_dict['CRPIX1'], wcs_dict['CRPIX2'] = np.array(wcs.backward_transform(*crval)) + 1

        ref_frame = wcs.output_frame.reference_frame
        if ref_frame is not None:
            wcs_dict['RADESYS'] = ref_frame.name.upper()
    else:
        # Do something if we this can't be specified as a normal FITS WCS?
        pass

    return wcs_dict

def gwcs_to_fits_spect(ndd, cenwave=None):
    """
    Convert a spectroscopic gWCS object to a series of FITS WCS keywords that
    can be inserted into the header so other software can understand the WCS.
    If this can't be done, a ValueError should be raised. If the FITS WCS is
    only approximate, this should be indicated with a dict entry
    {'FITS-WCS': 'APPROXIMATE'}

    Parameters
    ----------
    ndd: NDAstroData object

    Returns
    -------
    dict: values to insert into the FITS header to express this WCS
    """
    ctype_mapping = {'SPATIAL': 'SPATIAL', 'SPECTRAL': 'WAVE'}

    wcs = ndd.wcs
    wcs_dict = {}
    frame = wcs.output_frame

    wcs_dict.update({f'CTYPE{i}': ctype_mapping[axis_type]
                     for i, axis_type in enumerate(frame.axes_type, start=1)})
    wcs_dict.update({f'CUNIT{i}': unit.name
                     for i, unit in enumerate(frame.unit, start=1)})

    naxes = frame.naxes
    affine = calculate_affine_matrices(wcs, ndd.shape)
    wcs_dict.update({'CD{}_{}'.format(i+1, j+1): affine.matrix[j, i] for i in range(naxes) for j in range(naxes)})

    if cenwave is None:
        cenwave = affine.offset[frame.axes_type.index('SPECTRAL')]
    crval_mapping = {'SPATIAL': 0, 'SPECTRAL': cenwave}
    crval = tuple(crval_mapping[axis_type] for axis_type in frame.axes_type)
    wcs_dict.update({f'CRVAL{i}': cval for i, cval in enumerate(crval, start=1)})
    wcs_dict.update({f'CRPIX{i}': cpix+1 for i, cpix in enumerate(wcs.backward_transform(*crval), start=1)})

    # Flag this but still return the dict
    # TODO? Polynomial wavelength solutions
    if not model_is_affine(wcs.forward_transform):
        wcs_dict['FITS-WCS'] = ('APPROXIMATE', 'FITS WCS is approximate')

    return wcs_dict

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------

def model_is_affine(model):
    """"
    Test a Model for affinity. This is currently done by checking the
    name of its class (or the class names of all its submodels)

    TODO: Is this the right thing to do? We could compute the affine
    matrices *assuming* affinity, and then check that a number of random
    points behave as expected. Is that better?
    """
    try:
        return np.logical_and.reduce([model_is_affine(m)
                                      for m in model])
    except TypeError:
        return model.__class__.__name__[:5] in ('Affin', 'Rotat', 'Scale',
                                                'Shift', 'Ident')

def calculate_affine_matrices(func, shape):
    """
    Compute the matrix and offset necessary to turn a Transform into an
    affine transformation. This is done by computing the linear matrix
    along all axes extending from the centre of the region, and then
    calculating the offset such that the transformation is accurate at
    the centre of the region.

    Parameters
    ----------
    func: callable
        function that maps input->output coordinates
    shape: sequence
        shape to use for fiducial points

    Returns
    -------
        AffineMatrices(array, array): affine matrix and offset
    """
    ndim = len(shape)
    halfsize = [0.5 * length for length in shape]
    points = np.array([halfsize] * (2 * ndim + 1)).T
    points[:, 1:ndim + 1] += np.eye(ndim) * points[:, 0]
    points[:, ndim + 1:] -= np.eye(ndim) * points[:, 0]
    if ndim > 1:
        transformed = np.array(list(zip(*func(*points))))
    else:
        transformed = np.array([func(*points)]).T
    matrix = np.array([[0.5 * (transformed[j + 1, i] - transformed[ndim + j + 1, i]) / halfsize[j]
                        for j in range(ndim)] for i in range(ndim)])
    offset = transformed[0] - np.dot(matrix, halfsize)
    return AffineMatrices(matrix.T, offset[::-1])
