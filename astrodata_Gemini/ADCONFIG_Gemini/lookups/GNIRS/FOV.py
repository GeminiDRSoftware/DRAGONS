# This code is looked up by gempy as part of the configuration for the
# appropriate instrument and evaled by the infrastructure. It has initially
# been written to support gemini_tools.ExposureGroup. 

def pointing_in_field(pos, refpos, frac_FOV=1.0, frac_slit=1.0):

    """
    See gemini_tools.pointing_in_field() for the API. This is an
    instrument-specific back end that you shouldn't be calling directly.

    No inputs are validated at this level; that's the responsibility of the
    calling function, for reasons of efficiency.

    :type pos: AstroData instance
    :type refpos: tuple of floats
    :type frac_FOV: float
    :type frac_slit: float
    """

    # Since this function gets looked up & evaled, we have to do any
    # essential imports in-line (but Python caches them):
    import math, re

    # Use the first argument for looking up instrument & configuration
    # properties, since the second position doesn't always correspond to a
    # single exposure, ie. AstroData instance.
    ad = pos

    # Extract pointing info. (currently p/q but this will be replaced with
    # RA/Dec/PA) from the AstroData instance.
    position = (ad.phu_get_key_value('POFFSET'),
                ad.phu_get_key_value('QOFFSET'))

    # TO DO: References to the field size will need changing to decimal
    # degrees once we pass absolute co-ordinates?

    # TO DO: The following is used because as of r4619, the pixel_scale()
    # descriptor slows us down by 50x for some reason (and prepare has
    # already updated the header from the descriptor anyway so it doesn't
    # need recalculating here). The first branch of this condition can be
    # removed once pixel_scale() is improved or has the same check has
    # been added to it:
    if 'PREPARED' in ad.types:
        scale = ad.phu_get_key_value('PIXSCALE')
    else:
        scale = ad.pixel_scale().get_value()

    # Imaging:
    if 'GNIRS_IMAGE' in ad.types:
        # Need to fetch the illumination mask
        from astrodata import AstroData
        from astrodata.utils import Lookups
        from astrodata.utils.ConfigSpace  import lookup_path
        from gempy.gemini import gemini_tools as gt
        illum_mask_dict = Lookups.get_lookup_table("Gemini/GNIRS/IllumMaskDict",
                                                   "illum_masks")
        key1 = ad.camera().as_pytype()
        filter = ad.filter_name(pretty=True).as_pytype()
        if filter in ['Y', 'J', 'H', 'K']:
            key2 = 'Broadband'
        elif filter in ['JPHOT', 'HPHOT', 'KPHOT', 'H2', 'PAH']:
            key2 = 'Narrowband'
        else:
            raise ValueError("Unrecognised filter, no illumination mask can "
                             "be found for %s, so the pointing in field "
                             "cannot be determined" % ad.filename)
        key = (key1,key2)
        if key in illum_mask_dict:
            illum = lookup_path(illum_mask_dict[key])
        else:
            raise IOError("No illumination mask found for %s, the pointing in " 
                          "field cannot be determined " % ad.filename)
        illum_ad = None
        if isinstance(illum, AstroData):
            illum_ad = illum
        else:
            illum_ad = AstroData(illum)
            if illum_ad is None:
                raise TypeError("Cannot convert %s into an AstroData object, "
                                "the point in field cannot be determined" 
                                % illum)                
                
        # Checking the size of the illumination mask                
        final_illum = None
        if illum_ad is not None:
            # Clip the illumination mask to match the size of the input 
            # AstroData object science 
            final_illum = gt.clip_auxiliary_data(adinput=ad, aux=illum_ad,
                                                aux_type="bpm")[0]
        illum_data = final_illum['DQ'].data

#        # Defining the cass rotator center of GNIRS
#        center_dict = Lookups.get_lookup_table("Gemini/GNIRS/gnirsCenterDict", 
#                                               "gnirsCenterDict")
#        key = ad.observation_type().as_pytype()
#        if key in center_dict:
#            center = lookup_path(center_dict[key])
#        else:
#            center = None
#            log.warning("The cass rotator center of the image %s cannot be"
#                        "determined" % ad.filename)
#
#        # Next, finding the dx, dy between the cass center and the reference
#        (dx, dy) = (center[0] - refpos[0], center[1] - refpos[1])

        return illum_data[refpos[0],refpos[1]] == 0 

    # Spectroscopy:
    elif 'GNIRS_SPECT' in ad.types:
        raise NotImplementedError("FOV lookup not yet supported for GNIRS "
                                  "Spectroscopy")

    # Some engineering observation or bad mask value etc.:
    else:
        raise ValueError("Can't determine FOV for unrecognized GNIRS config " \
          "(%s, %s)" % (str(ad.focal_plane_mask()), str(ad.disperser())))

