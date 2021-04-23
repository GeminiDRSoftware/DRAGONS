import functools
import re
from collections import namedtuple

import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import fits
from astropy.modeling import core, models, projections
from gwcs import coordinate_frames as cf
from gwcs import utils as gwutils
from gwcs.utils import sky_pairs, specsystems
from gwcs.wcs import WCS as gWCS

AffineMatrices = namedtuple("AffineMatrices", "matrix offset")

FrameMapping = namedtuple("FrameMapping", "cls description")

# Type of CoordinateFrame to construct for a FITS keyword and
# readable name so user knows what's going on
frame_mapping = {'WAVE': FrameMapping(cf.SpectralFrame, "Wavelength in vacuo"),
                 'AWAV': FrameMapping(cf.SpectralFrame, "Wavelength in air")}

#-----------------------------------------------------------------------------
# FITS-WCS -> gWCS
#-----------------------------------------------------------------------------


def fitswcs_to_gwcs(hdr):
    """
    Create and return a gWCS object from a FITS header. If it can't
    construct one, it should quietly return None.
    """
    # coordinate names for CelestialFrame
    coordinate_outputs = {'alpha_C', 'delta_C'}

    # transform = gw.make_fitswcs_transform(hdr)
    try:
        transform = make_fitswcs_transform(hdr)
    except Exception as e:
        return None
    outputs = transform.outputs
    wcs_info = read_wcs_from_header(hdr)

    naxes = transform.n_inputs
    axes_names = ('x', 'y', 'z', 'u', 'v', 'w')[:naxes]
    in_frame = cf.CoordinateFrame(naxes=naxes, axes_type=['SPATIAL'] * naxes,
                                  axes_order=tuple(range(naxes)), name="pixels",
                                  axes_names=axes_names, unit=[u.pix] * naxes)

    out_frames = []
    for i, output in enumerate(outputs):
        unit_name = wcs_info["CUNIT"][i]
        try:
            unit = u.Unit(unit_name)
        except TypeError:
            unit = None
        try:
            frame_type = output[:4].upper()
            frame_info = frame_mapping[frame_type]
        except KeyError:
            if output in coordinate_outputs:
                continue
            frame = cf.CoordinateFrame(naxes=1, axes_type=("SPATIAL",),
                                       axes_order=(i,), unit=unit,
                                       axes_names=(output,), name=output)
        else:
            frame = frame_info.cls(axes_order=(i,), unit=unit,
                                   axes_names=(frame_type,),
                                   name=frame_info.description)

        out_frames.append(frame)

    if coordinate_outputs.issubset(outputs):
        frame_name = wcs_info["RADESYS"]  # FK5, etc.
        axes_names = None
        try:
            ref_frame = getattr(coord, frame_name)()
            # TODO? Work out how to stuff EQUINOX and OBS-TIME into the frame
        except (AttributeError, TypeError):
            # TODO: Replace quick fix as gWCS doesn't recognize GAPPT
            if frame_name == "GAPPT":
                ref_frame = coord.FK5()
            else:
                ref_frame = None
                axes_names = ('lon', 'lat')
        axes_order = (outputs.index('alpha_C'), outputs.index('delta_C'))

        # Call it 'world' if there are no other axes, otherwise 'sky'
        name = 'SKY' if len(outputs) > 2 else 'world'
        cel_frame = cf.CelestialFrame(reference_frame=ref_frame, name=name,
                                      axes_names=axes_names, axes_order=axes_order)
        out_frames.append(cel_frame)

    out_frame = (out_frames[0] if len(out_frames) == 1
                 else cf.CompositeFrame(out_frames, name='world'))
    return gWCS([(in_frame, transform),
                 (out_frame, None)])


# -----------------------------------------------------------------------------
# gWCS -> FITS-WCS
# -----------------------------------------------------------------------------

def gwcs_to_fits(ndd, hdr=None):
    """
    Convert a gWCS object to a collection of FITS WCS keyword/value pairs,
    if possible. If the FITS WCS is only approximate, this should be indicated
    with a dict entry {'FITS-WCS': 'APPROXIMATE'}. If there is no suitable
    FITS representation, then a ValueError or NotImplementedError can be
    raised.

    Parameters
    ----------
    ndd : `astropy.nddata.NDData`
        The NDData whose wcs attribute we want converted
    hdr : `astropy.io.fits.Header`
        A Header object that may contain some useful keywords

    Returns
    -------
    dict
        values to insert into the FITS header to express this WCS

    """
    if hdr is None:
        hdr = {}

    wcs = ndd.wcs
    transform = wcs.forward_transform
    world_axes = list(wcs.output_frame.axes_names)
    nworld_axes = len(world_axes)
    wcs_dict = {'WCSAXES': nworld_axes,
                'WCSDIM': nworld_axes}
    wcs_dict.update({f'CD{i+1}_{j+1}': 0. for j in range(nworld_axes)
                     for i in range(nworld_axes)})
    pix_center = [0.5 * (length - 1) for length in ndd.shape[::-1]]
    wcs_center = transform(*pix_center)

    # Find and process the sky projection first
    if {'lon', 'lat'}.issubset(world_axes):
        if isinstance(wcs.output_frame, cf.CelestialFrame):
            cel_frame = wcs.output_frame
        elif isinstance(wcs.output_frame, cf.CompositeFrame):
            for frame in wcs.output_frame.frames:
                if isinstance(frame, cf.CelestialFrame):
                    cel_frame = frame

        # TODO: Non-ecliptic coordinate frames
        cel_ref_frame = cel_frame.reference_frame
        if not isinstance(cel_ref_frame, coord.builtin_frames.BaseRADecFrame):
            raise NotImplementedError("Cannot write non-ecliptic frames yet")
        wcs_dict['RADESYS'] = cel_ref_frame.name.upper()

        for m in transform:
            if isinstance(m, models.RotateNative2Celestial):
                nat2cel = m
            if isinstance(m, models.Pix2SkyProjection):
                m.name = 'pix2sky'
                # Determine which sort of projection this is
                for projcode in projections.projcodes:
                    if isinstance(m, getattr(models, f'Pix2Sky_{projcode}')):
                        break
                else:
                    raise ValueError("Unknown projection class: {}".
                                     format(m.__class__.__name__))

        lon_axis = world_axes.index('lon')
        lat_axis = world_axes.index('lat')
        world_axes[lon_axis] = f'RA---{projcode}'
        world_axes[lat_axis] = f'DEC--{projcode}'
        wcs_dict[f'CRVAL{lon_axis+1}'] = nat2cel.lon.value
        wcs_dict[f'CRVAL{lat_axis+1}'] = nat2cel.lat.value

        # Remove projection parts so we can calculate the CD matrix
        if projcode:
            nat2cel.name = 'nat2cel'
            transform = transform.replace_submodel('pix2sky', models.Identity(2))
            transform = transform.replace_submodel('nat2cel', models.Identity(2))

    # Deal with other axes
    # TODO: AD should refactor to allow the descriptor to be used here
    for i, axis_type in enumerate(wcs.output_frame.axes_type, start=1):
        if f'CRVAL{i}' in wcs_dict:
            continue
        if axis_type == "SPECTRAL":
            wcs_dict[f'CRVAL{i}'] = hdr.get('CENTWAVE', wcs_center[i-1] if nworld_axes > 1 else wcs_center)
            wcs_dict[f'CTYPE{i}'] = wcs.output_frame.axes_names[i-1]  # AWAV/WAVE
        else:  # Just something
            wcs_dict[f'CRVAL{i}'] = 0

    # Flag if we can't construct a perfect CD matrix
    if not model_is_affine(transform):
        wcs_dict['FITS-WCS'] = ('APPROXIMATE', 'FITS WCS is approximate')

    affine = calculate_affine_matrices(transform, ndd.shape)
    # Convert to x-first order
    affine_matrix = affine.matrix[::-1, ::-1]
    # Require an inverse to write out
    if np.linalg.det(affine_matrix) == 0:
        affine_matrix[-1, -1] = 1.
    wcs_dict.update({f'CD{i+1}_{j+1}': affine_matrix[i, j]
                     for j, _ in enumerate(world_axes)
                     for i, _ in enumerate(world_axes)})
    # Don't overwrite CTYPEi keywords we've already created
    wcs_dict.update({f'CTYPE{i}': axis.upper()[:8]
                     for i, axis in enumerate(world_axes, start=1)
                     if f'CTYPE{i}' not in wcs_dict})

    crval = [wcs_dict[f'CRVAL{i+1}'] for i, _ in enumerate(world_axes)]
    crpix = np.array(wcs.backward_transform(*crval)) + 1
    if nworld_axes == 1:
        wcs_dict['CRPIX1'] = crpix
    else:
        # Comply with FITS standard, must define CRPIXj for "extra" axes
        wcs_dict.update({f'CRPIX{j}': cpix for j, cpix in enumerate(np.concatenate([crpix, [0] * (nworld_axes-len(ndd.shape))]), start=1)})
    for i, unit in enumerate(wcs.output_frame.unit, start=1):
        try:
            wcs_dict[f'CUNIT{i}'] = unit.name
        except AttributeError:
            pass

    return wcs_dict

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def model_is_affine(model):
    """"
    Test a Model for affinity. This is currently done by checking the
    name of its class (or the class names of all its submodels)

    TODO: Is this the right thing to do? We could compute the affine
    matrices *assuming* affinity, and then check that a number of random
    points behave as expected. Is that better?
    """
    if isinstance(model, dict):  # handle fix_inputs()
        return True
    try:
        return np.logical_and.reduce([model_is_affine(m)
                                      for m in model])
    except TypeError:
        # TODO: Delete "Const" one fix_inputs() broadcastingis fixed
        return model.__class__.__name__[:5] in ('Affin', 'Rotat', 'Scale',
                                                'Shift', 'Ident', 'Mappi',
                                                'Const')

def calculate_affine_matrices(func, shape):
    """
    Compute the matrix and offset necessary of an affine transform that
    represents the supplied function. This is done by computing the
    linear matrix along all axes extending from the centre of the region,
    and then calculating the offset such that the transformation is
    accurate at the centre of the region. The matrix and offset are returned
    in the standard python order (i.e., y-first for 2D).

    Parameters
    ----------
    func : callable
        function that maps input->output coordinates
    shape : sequence
        shape to use for fiducial points

    Returns
    -------
    AffineMatrices(array, array)
        affine matrix and offset

    """
    indim = len(shape)
    try:
        ndim = len(func(*shape))  # handle increase in number of axes
    except TypeError:
        ndim = 1
    halfsize = [0.5 * length for length in shape] + [1.] * (ndim - indim)
    points = np.array([halfsize] * (2 * ndim + 1)).T
    points[:, 1:ndim + 1] += np.eye(ndim) * points[:, 0]
    points[:, ndim + 1:] -= np.eye(ndim) * points[:, 0]
    if ndim > 1:
        transformed = np.array(list(zip(*list(func(*point[:indim])
                                              for point in points.T)))).T
    else:
        transformed = np.array([func(*points)]).T
    matrix = np.array([[0.5 * (transformed[j + 1, i] - transformed[ndim + j + 1, i]) / halfsize[j]
                        for j in range(ndim)] for i in range(ndim)])
    offset = transformed[0] - np.dot(matrix, halfsize)
    return AffineMatrices(matrix[::-1, ::-1], offset[::-1])


# -------------------------------------------------------------------------
# This stuff will hopefully all go into gwcs.utils
# -------------------------------------------------------------------------
def read_wcs_from_header(header):
    """
    Extract basic FITS WCS keywords from a FITS Header.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        FITS Header with WCS information.

    Returns
    -------
    wcs_info : dict
        A dictionary with WCS keywords.
    """
    wcs_info = {}

    try:
        wcs_info['WCSAXES'] = header['WCSAXES']
    except KeyError:
        p = re.compile(r'ctype[\d]*', re.IGNORECASE)
        ctypes = header['CTYPE*']
        keys = list(ctypes.keys())
        for key in keys[::-1]:
            if p.split(key)[-1] != "":
                keys.remove(key)
        wcs_info['WCSAXES'] = len(keys)
    wcsaxes = wcs_info['WCSAXES']
    # if not present call get_csystem
    wcs_info['RADESYS'] = header.get('RADESYS', header.get('RADECSYS', 'FK5'))
    wcs_info['VAFACTOR'] = header.get('VAFACTOR', 1)
    # NAXIS=0 if we're reading from a PHU
    wcs_info['NAXIS'] = header.get('NAXIS') or max(int(k[5:]) for k in header['CRPIX*'].keys())
    # date keyword?
    # wcs_info['DATEOBS'] = header.get('DATE-OBS', 'DATEOBS')
    wcs_info['EQUINOX'] = header.get("EQUINOX", None)
    wcs_info['EPOCH'] = header.get("EPOCH", None)
    wcs_info['DATEOBS'] = header.get("MJD-OBS", header.get("DATE-OBS", None))

    ctype = []
    cunit = []
    crpix = []
    crval = []
    cdelt = []
    for i in range(1, wcsaxes + 1):
        ctype.append(header['CTYPE{0}'.format(i)])
        cunit.append(header.get('CUNIT{0}'.format(i), None))
        crpix.append(header.get('CRPIX{0}'.format(i), 0.0))
        crval.append(header.get('CRVAL{0}'.format(i), 0.0))
        cdelt.append(header.get('CDELT{0}'.format(i), 1.0))

    has_cd = len(header['CD?_?']) > 0
    cd = np.zeros((wcsaxes, wcsaxes))
    for i in range(1, wcsaxes + 1):
        for j in range(1, wcsaxes + 1):
            if has_cd:
                cd[i - 1, j - 1] = header.get('CD{0}_{1}'.format(i, j), 0)
            else:
                cd[i - 1, j - 1] = cdelt[i - 1] * header.get('PC{0}_{1}'.format(i, j),
                                                             1 if i == j else 0)
    wcs_info['CTYPE'] = ctype
    wcs_info['CUNIT'] = cunit
    wcs_info['CRPIX'] = crpix
    wcs_info['CRVAL'] = crval
    wcs_info['CD'] = cd
    return wcs_info

def get_axes(header):
    """
    Matches input with spectral and sky coordinate axes.

    Parameters
    ----------
    header : `astropy.io.fits.Header` or dict
        FITS Header (or dict) with basic WCS information.

    Returns
    -------
    sky_inmap, spectral_inmap, unknown : list
        indices in the output representing sky and spectral coordinates.

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    # Split each CTYPE value at "-" and take the first part.
    # This should represent the coordinate system.
    ctype = [ax.split('-')[0].upper() for ax in wcs_info['CTYPE']]
    sky_inmap = []
    spec_inmap = []
    unknown = []
    skysystems = np.array(list(sky_pairs.values())).flatten()
    for ax in ctype:
        ind = ctype.index(ax)
        if ax in specsystems:
            spec_inmap.append(ind)
        elif ax in skysystems:
            sky_inmap.append(ind)
        else:
            unknown.append(ind)

    if sky_inmap:
        _is_skysys_consistent(ctype, sky_inmap)

    return sky_inmap, spec_inmap, unknown


def _is_skysys_consistent(ctype, sky_inmap):
    """ Determine if the sky axes in CTYPE match to form a standard celestial system."""
    if len(sky_inmap) != 2:
        raise ValueError("{} sky coordinate axes found. "
                         "There must be exactly 2".format(len(sky_inmap)))

    for item in sky_pairs.values():
        if ctype[sky_inmap[0]] == item[0]:
            if ctype[sky_inmap[1]] != item[1]:
                raise ValueError(
                    "Inconsistent ctype for sky coordinates {0} and {1}".format(*ctype))
            break
        elif ctype[sky_inmap[1]] == item[0]:
            if ctype[sky_inmap[0]] != item[1]:
                raise ValueError(
                    "Inconsistent ctype for sky coordinates {0} and {1}".format(*ctype))
            sky_inmap.reverse()
            break

def _get_contributing_axes(wcs_info, world_axes):
    """
    Returns a tuple indicating which axes in the pixel frame make a
    contribution to an axis or axes in the output frame.

    Parameters
    ----------
    wcs_info : dict
        dict of WCS information
    world_axes : int or iterable of int
        axes in the world coordinate system

    Returns
    -------
    axes : list
        axes whose pixel coordinates affect the output axis/axes
    """
    cd = wcs_info['CD']
    try:
        return sorted(set(np.nonzero(cd[tuple(world_axes), :wcs_info['NAXIS']])[1]))
    except TypeError:  # world_axes is an int
        return sorted(np.nonzero(cd[world_axes, :wcs_info['NAXIS']])[0])
    #return sorted(set(j for j in range(wcs_info['NAXIS'])
    #                    for i in world_axes if cd[i, j] != 0))

def make_fitswcs_transform(header):
    """
    Create a basic FITS WCS transform.
    It does not include distortions.

    Parameters
    ----------
    header : `astropy.io.fits.Header` or dict
        FITS Header (or dict) with basic WCS information

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    # If a pixel axis maps directly to an output axis, we want to have that
    # model completely self-contained, so don't put all the CRPIXj shifts
    # in a single CompoundModel at the beginning
    transforms = []

    # The tricky stuff!
    sky_model = fitswcs_image(wcs_info)
    linear_models = fitswcs_linear(wcs_info)
    all_models = linear_models
    if sky_model:
        all_models.append(sky_model)

    # Now arrange the models so the inputs and outputs are in the right places
    all_models.sort(key=lambda m: m.meta['output_axes'][0])
    input_axes = [ax for m in all_models for ax in m.meta['input_axes']]
    output_axes = [ax for m in all_models for ax in m.meta['output_axes']]
    if input_axes != list(range(len(input_axes))):
        input_mapping = models.Mapping(input_axes)
        transforms.append(input_mapping)

    transforms.append(functools.reduce(core._model_oper('&'), all_models))

    if output_axes != list(range(len(output_axes))):
        output_mapping = models.Mapping(output_axes)
        transforms.append(output_mapping)

    return functools.reduce(core._model_oper('|'), transforms)

def fitswcs_image(header):
    """
    Make a complete transform from CRPIX-shifted pixels to
    sky coordinates from FITS WCS keywords. A Mapping is inserted
    at the beginning, which may be removed later

    Parameters
    ----------
    header : `astropy.io.fits.Header` or dict
        FITS Header or dict with basic FITS WCS keywords.

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    crpix = wcs_info['CRPIX']
    cd = wcs_info['CD']
    # get the part of the PC matrix corresponding to the imaging axes
    sky_axes, spec_axes, unknown = get_axes(wcs_info)
    if not sky_axes:
        if len(unknown) == 2:
            sky_axes = unknown
        else:  # No sky here
            return
    pixel_axes = _get_contributing_axes(wcs_info, sky_axes)
    if len(pixel_axes) > 2:
        raise ValueError("More than 2 pixel axes contribute to the sky coordinates")

    translation_models = [models.Shift(-(crpix[i] - 1), name='crpix' + str(i + 1))
                          for i in pixel_axes]
    translation = functools.reduce(lambda x, y: x & y, translation_models)
    transforms = [translation]

    # If only one axis is contributing to the sky (e.g., slit spectrum)
    # then it must be that there's an extra axis in the CD matrix, so we
    # create a "ghost" orthogonal axis here so an inverse can be defined
    # Modify the CD matrix in case we have to use a backup Matrix Model later
    if len(pixel_axes) == 1:
        cd[sky_axes[0], -1] = -cd[sky_axes[1], pixel_axes[0]]
        cd[sky_axes[1], -1] = cd[sky_axes[0], pixel_axes[0]]
        sky_cd = cd[np.ix_(sky_axes, pixel_axes + [-1])]
        affine = models.AffineTransformation2D(matrix=sky_cd, name='cd_matrix')
        # TODO: replace when PR#10362 is in astropy
        #rotation = models.fix_inputs(affine, {'y': 0})
        rotation = models.Mapping((0, 0)) | models.Identity(1) & models.Const1D(0) | affine
        rotation.inverse = affine.inverse | models.Mapping((0,), n_inputs=2)
    else:
        sky_cd = cd[np.ix_(sky_axes, pixel_axes)]
        rotation = models.AffineTransformation2D(matrix=sky_cd, name='cd_matrix')
    transforms.append(rotation)

    projection = gwutils.fitswcs_nonlinear(wcs_info)
    if projection:
        transforms.append(projection)

    sky_model = functools.reduce(lambda x, y: x | y, transforms)
    sky_model.name = 'SKY'
    sky_model.meta.update({'input_axes': pixel_axes,
                           'output_axes': sky_axes})
    return sky_model

def fitswcs_linear(header):
    """
    Create WCS linear transforms for any axes not associated with
    celestial coordinates. We require that each world axis aligns
    precisely with only a single pixel axis.

    Parameters
    ----------
    header : `astropy.io.fits.Header` or dict
        FITS Header or dict with basic FITS WCS keywords.

    """
    # We *always* want the wavelength solution model to be called "WAVE"
    # even if the CTYPE keyword is "AWAV"
    model_name_mapping = {"AWAV": "WAVE"}

    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    cd = wcs_info['CD']
    crpix = wcs_info['CRPIX']
    crval = wcs_info['CRVAL']
    # get the part of the CD matrix corresponding to the imaging axes
    sky_axes, spec_axes, unknown = get_axes(wcs_info)
    if not sky_axes and len(unknown) == 2:
        unknown = []

    linear_models = []
    for ax in spec_axes + unknown:
        pixel_axes = _get_contributing_axes(wcs_info, ax)
        if len(pixel_axes) == 1:
            pixel_axis = pixel_axes[0]
            linear_model = (models.Shift(1 - crpix[pixel_axis],
                                            name='crpix' + str(pixel_axis + 1)) |
                            models.Scale(cd[ax, pixel_axis]) |
                            models.Shift(crval[ax]))
            ctype = wcs_info['CTYPE'][ax][:4].upper()
            linear_model.name = model_name_mapping.get(ctype, ctype)
            linear_model.outputs = (wcs_info['CTYPE'][ax],)
            linear_model.meta.update({'input_axes': pixel_axes,
                                      'output_axes': [ax]})
            linear_models.append(linear_model)
        else:
            raise ValueError(f"Axis {ax} depends on more than one input axis")

    return linear_models
