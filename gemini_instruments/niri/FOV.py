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
    # removed once pixel_scale() is improved or has the same check has been
    # added to it:
    if 'PREPARED' in ad.types:
        scale = ad.phu_get_key_value('PIXSCALE')
    else:
        scale = ad.pixel_scale().get_value()

    # Imaging:
    if 'NIRI_IMAGE' in ad.types:
        dist = 512.* scale
        return all([abs(x-r) < dist for x, r in zip(position,refpos)])

    # Long slit:
    elif 'NIRI_SPECT' in ad.types:
        # I'm leaving NIRI spectroscopy as an exercise for later, after
        # identifying the following complications: The NIRI slits have
        # different lengths, depending where they are installed in the MOS
        # wheel. Some of the slits are also off-centre in p. Thus we need to
        # look up slit characteristics by name. Since that information may be
        # of more general use, it should probably go in a separate look-up
        # table, but 1) that needs designing and 2) adding an extra
        # get_lookup_table() in here will generate a lot of overhead. I'm
        # thinking that gemini_python Lookups need managing more intelligently
        # via a caching object, which would have to be implemented first. Since
        # NIRI spectroscopy currently isn't being used, it doesn't seem urgent
        # to put a stopgap solution here right away.
        #
        # The f/32 4-pixel slit has a width that varies along its length, but
        # maybe we can just use an approximation for that one.
        #
        # The centring information for "blue" slits is here (I would check the
        # sense of the p offset using the OT):
        # http://internal.gemini.edu/science/instruments/NIRI/iaa/iaa.html
        #
        # The following commented code should be correct but is incomplete:

        # # Parse slit width in pixels from mask name:
        # mask = str(ad.focal_plane_mask())
        # matches = re.match("f(6|32)-([0-9]{1,2})pix(Bl)?_G[0-9]+$", mask)
        # try:
        #     width = float(matches.group(2))
        # except (AttributeError, TypeError, ValueError):
        #     raise ValueError("Failed to parse width for NIRI slit (%s)" % mask)

        # # Look up the slit length & offset here.

        # # Tuple of (adjusted) field extent in each direction (assumes p,q):
        # dist = (frac_slit*0.5*width*scale, frac_FOV*length)

        # # Is the position within the field boundaries along both/all axes?
        # return all([abs(x-r-o) < d for x, r, d, o in \
        #            zip(position,refpos,dist,offset)])

        raise NotImplementedError("FOV lookup not yet supported for NIRI " \
          "spectroscopy")

    # Some engineering observation or bad mask value etc.:
    else:
        raise ValueError("Can't determine FOV for unrecognized NIRI config. " \
          "(%s, %s)" % (str(ad.focal_plane_mask()), str(ad.disperser())))

