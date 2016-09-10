from astrodata import AstroData
from astrodata.utils import Lookups
from astrodata.utils.ConfigSpace  import lookup_path
from gempy.gemini import gemini_tools as gt

from .. import keyword_comments
from . import IllumMaskDict

# ------------------------------------------------------------------------------
keyword_comments = keyword_comments.keyword_comments

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
    theta = math.radians(pos.phu_get_key_value("PA"))
    scale = pos.pixel_scale()
    position = (pos.phu_get_key_value('POFFSET'),
                pos.phu_get_key_value('QOFFSET'))
    deltap = (refpos[0] - position[0]) / scale
    deltaq = (refpos[1] - position[1]) / scale
    xshift = (deltap * math.cos(theta)) - (deltaq * math.sin(theta))
    yshift =  (deltap * math.sin(theta)) + (deltaq * math.cos(theta))                 
    
    # Imaging:
    if 'GNIRS_IMAGE' in pos.types:
        illum_ad = fetch_illum_mask(pos)
        
        # Checking the size of the illumination mask                
        final_illum = None
        if illum_ad is not None:
            # Clip the illumination mask to match the size of the input 
            # AstroData object science 
            final_illum = gt.clip_auxiliary_data(adinput=pos, aux=illum_ad, 
                                                 aux_type="bpm", 
                                                 keyword_comments=keyword_comments)[0]
        illum_data = final_illum['DQ'].data

        # Finding the center of the illumination mask
        center_illum = (final_illum.phu_get_key_value('CENMASSX'), 
                        final_illum.phu_get_key_value('CENMASSY'))
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
    elif 'GNIRS_SPECT' in ad.types:
        raise NotImplementedError("FOV lookup not yet supported for GNIRS "
                                  "Spectroscopy")

    # Some engineering observation or bad mask value etc.:
    else:
        raise ValueError("Can't determine FOV for unrecognized GNIRS config " \
          "(%s, %s)" % (str(ad.focal_plane_mask()), str(ad.disperser())))

def fetch_illum_mask(ad):
    # Fetches the appropriate illumination mask for an astrodata instance
            
    illum_mask_dict = IllumMaskDict.illum_masks
    key1 = ad.camera().as_pytype()
    filter = ad.filter_name(pretty=True).as_pytype()
    if filter in ['Y', 'J', 'H', 'K', 'H2', 'PAH']:
        key2 = 'Wings'
    elif filter in ['JPHOT', 'HPHOT', 'KPHOT']:
        key2 = 'NoWings'
    else:
        raise ValueError("Unrecognised filter, no illumination mask can "
                         "be found for %s" % ad.filename)
    key = (key1,key2)
    if key in illum_mask_dict:
        illum = lookup_path(illum_mask_dict[key])
    else:
        raise IOError("No illumination mask found for %s" % pos.filename)
    
    illum_ad = None
    if isinstance(illum, AstroData):
        illum_ad = illum
    else:
        illum_ad = AstroData(illum)
        if illum_ad is None:
            raise TypeError("Cannot convert %s into an AstroData object" 
                            % illum)                
                
    return illum_ad