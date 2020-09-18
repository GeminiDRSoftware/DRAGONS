from os import path

import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt
from gempy.utils import logutils

from scipy.ndimage import shift
from geminidr.gemini.lookups import DQ_definitions as DQ

from . import maskdb

# ------------------------------------------------------------------------------
# This code is looked up by gempy as part of the configuration for the
# appropriate instrument and evaled by the infrastructure. It has initially
# been written to support gemini_tools.ExposureGroup.
#
# All calculations are now done in PIXELS

def pointing_in_field(ad, refpos, frac_FOV=1.0, frac_slit=1.0):

    """
    See gemini_tools.pointing_in_field() for the API. This is an
    instrument-specific back end that you shouldn't be calling directly.

    No inputs are validated at this level; that's the responsibility of the
    calling function, for reasons of efficiency.
    
    The GNIRS FOV is determined by whether the calculated center point 
    (according to the center of mass of the illumination mask) of the
    image falls within the illumination mask of the reference image.
    
    :param pos: AstroData instance to be checked for whether it belongs
                in the same sky grouping as refpos
    :type pos: AstroData instance
    
    :param refpos: This is the POFFSET and QOFFSET of the reference image
    :type refpos: tuple of floats
    
    :param frac_FOV: For use with spectroscopy data
    :type frac_FOV: float
    
    :param frac_slit: For use with spectroscopy data
    :type frac_slit: float
    """
    # Object location in AD = refpos + shift
    xshift = ad.detector_x_offset() - refpos[0]
    yshift = ad.detector_y_offset() - refpos[1]

    # We inverse-scale by frac_FOV
    fov_xshift = -int(xshift / frac_FOV)
    fov_yshift = -int(yshift / frac_FOV)
    # Imaging:
    if 'IMAGE' in ad.tags:
        if (abs(fov_yshift) >= ad[0].shape[1] or
                abs(fov_xshift) >= ad[0].shape[1]):
            return False

        illum = get_illum_mask_filename(ad)
        if illum:
            illum_ad = gt.clip_auxiliary_data(adinput=ad,
                            aux=astrodata.open(illum), aux_type="bpm")
            illum_data = illum_ad[0].data
        else:
            raise IOError("Cannot find illumination mask for {}".
                          format(ad.filename))

        # Shift the illumination mask and see if the shifted keyhole
        # overlaps with the original keyhole
        shifted_data = shift(illum_data, (fov_yshift, fov_xshift),
                             order=0, cval=DQ.unilluminated)
        return (illum_data | shifted_data == 0).any()

    # Spectroscopy:
    elif 'SPECT' in ad.tags:
        raise NotImplementedError("FOV lookup not yet supported for GNIRS "
                                  "Spectroscopy")

    # Some engineering observation or bad mask value etc.:
    else:
        raise ValueError("Can't determine FOV for unrecognized GNIRS config "
          "({}, {})".format(ad.focal_plane_mask(), ad.disperser()))

def get_illum_mask_filename(ad):
    """
    Gets the illumMask filename for an input science frame, using
    illumMask_dict in geminidr.gnirs.lookups.maskdb.py

    Returns
    -------
    str/None: Filename of the appropriate illumination mask
    """
    log = logutils.get_logger(__name__)
    key1 = ad.camera()
    filter = ad.filter_name(pretty=True)
    if filter in ['Y', 'J', 'H', 'K', 'H2', 'PAH']:
        key2 = 'Wings'
    elif filter in ['JPHOT', 'HPHOT', 'KPHOT']:
        key2 = 'NoWings'
    else:
        log.warning("Unrecognised filter, no illumination mask can "
                         "be found for {}".format(ad.filename))
        return None

    try:
        illum = path.join(maskdb.illumMask_dict[key1, key2])
    except KeyError:
        log.warning("No illumination mask found for {}".format(ad.filename))
        return None

    return illum if illum.startswith(path.sep) else \
        path.join(path.dirname(maskdb.__file__), 'BPM', illum)