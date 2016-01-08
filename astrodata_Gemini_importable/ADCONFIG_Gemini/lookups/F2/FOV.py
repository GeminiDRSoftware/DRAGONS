# This code is looked up by gempy as part of the configuration for the
# appropriate instrument and evaled by the infrastructure. It has initially
# been written to support gemini_tools.ExposureGroup and is semi-optimized. 

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

    # GeMS truncates the FOV to 2' and since there are currently no AO
    # type classifications, use the pixel scale that we just looked up to
    # figure out whether the AO field applies or not:
    with_gems = (scale < 0.1)

    # Imaging:
    if 'F2_IMAGE' in ad.types:
        dist = 60. if with_gems else 183.
        return math.sqrt(sum([(x-r)**2 for x, r in zip(position,refpos)])) \
             < frac_FOV * dist
    # Long slit:
    elif 'F2_LS' in ad.types:
        # Parse slit width in pixels from mask name:
        mask = str(ad.focal_plane_mask())
        try:
            width = float(re.match("([0-9]{1,2})pix-slit$", mask).group(1))
        except (AttributeError, TypeError, ValueError):
            raise ValueError("Failed to parse width for F2 slit (%s)" % mask)

        # TO DO: This first number should be correct but the Web page doesn't
        # confirm it, so check with the instrument team:
        dist = 60. if with_gems else 131.5

        # Tuple of (adjusted) field extent in each direction (assumes p,q):
        # The slit length is from the Web page, under long-slit spectroscopy.
        dist = (frac_slit*0.5*width*scale, frac_FOV*dist)

        # Is the position within the field boundaries along both/all axes?
        return all([abs(x-r) < d for x, r, d in zip(position,refpos,dist)])

    # MOS:
    elif 'F2_MOS' in ad.types:
        # If we need to check the MOS mask name at some point, the regexp is
        # "G(N|S)[0-9]{4}(A|B)(Q|L)[0-9]{3}-[0-9]+$" (harmlessly relaxing the
        # final running mask number to avoid a new release if Gemini were to
        # expand the range in future). It seems better to define that naming
        # convention here and for GMOS in duplicate than redirect to yet
        # another lookup file when people might want to call this code from
        # inside a loop.

        # Still need to get some example data & MDF(s) and work on this:
        raise NotImplementedError("FOV lookup not yet supported for F2 MOS")

    # Some engineering observation or bad mask value etc.:
    else:
        raise ValueError("Can't determine FOV for unrecognized F2 config " \
          "(%s, %s)" % (str(ad.focal_plane_mask()), str(ad.disperser())))

