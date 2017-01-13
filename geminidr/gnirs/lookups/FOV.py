from os import path

import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt

from geminidr.gemini.lookups.keyword_comments import keyword_comments
from . import mask_dict

# ------------------------------------------------------------------------------
# This code is looked up by gempy as part of the configuration for the
# appropriate instrument and evaled by the infrastructure. It has initially
# been written to support gemini_tools.ExposureGroup. 
def pointing_in_field(pos, refpos, frac_FOV=1.0, frac_slit=1.0):

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
    # Since this function gets looked up and evaluated, we have to do any
    # essential imports in-line (but Python caches them)
    import math
    
    # Extract pointing info in terms of the x and y offsets
    # Since we are only looking at the center position of the image relative
    # to the reference image, the PA of the image to be classified is 
    # sufficient (luckily!)
    theta = math.radians(pos.phu.PA)
    scale = pos.pixel_scale()
    position = (pos.phu.POFFSET, pos.phu.QOFFSET)
    deltap = (refpos[0] - position[0]) / scale
    deltaq = (refpos[1] - position[1]) / scale
    xshift = (deltap * math.cos(theta)) - (deltaq * math.sin(theta))
    yshift =  (deltap * math.sin(theta)) + (deltaq * math.cos(theta))
    ad = pos
    
    # Imaging:
    if 'IMAGE' in pos.tags:
        illum_ad = gt.clip_auxiliary_data(adinput=pos,
                        aux=fetch_illum_mask(pos), aux_type="bpm",
                        keyword_comments=keyword_comments)
        illum_data = illum_ad[0].data

        # Finding the center of the illumination mask
        center_illum = (illum_ad.phu.CENMASSX, illum_ad.phu.CENMASSY)
        checkpos = (int(center_illum[0] + xshift),
                    int(center_illum[1] + yshift))
        
        # If the position to check is going to fall outside the illumination
        # mask, return straight away to avoid an error
        if ((abs(xshift) >= abs(center_illum[0])) or 
            (abs(yshift) >= abs(center_illum[1]))):
            return False

        # Note that numpy data arrays are reversed in x and y    
        return illum_data[checkpos[1], checkpos[0]] == 0 

    # Spectroscopy:
    elif 'SPECT' in ad.tags:
        raise NotImplementedError("FOV lookup not yet supported for GNIRS "
                                  "Spectroscopy")

    # Some engineering observation or bad mask value etc.:
    else:
        raise ValueError("Can't determine FOV for unrecognized GNIRS config "
          "({}, {})".format(ad.focal_plane_mask(), ad.disperser()))

def fetch_illum_mask(ad):
    # Fetches the appropriate illumination mask for an astrodata instance
            
    key1 = ad.camera()
    filter = ad.filter_name(pretty=True)
    if filter in ['Y', 'J', 'H', 'K', 'H2', 'PAH']:
        key2 = 'Wings'
    elif filter in ['JPHOT', 'HPHOT', 'KPHOT']:
        key2 = 'NoWings'
    else:
        raise ValueError("Unrecognised filter, no illumination mask can "
                         "be found for {}".format(ad.filename))

    try:
        illum = path.join(path.dirname(mask_dict.__file__), 'BPM',
                          mask_dict.illum_masks[key1, key2])
    except KeyError:
        raise IOError("No illumination mask found for {}".format(ad.filename))

    try:
        illum_ad = astrodata.open(illum)
    except:
        raise IOError("Cannot convert {} into an AstroData object".format(illum))
                
    return illum_ad