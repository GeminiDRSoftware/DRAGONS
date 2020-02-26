# This code is looked up by gempy as part of the configuration for the
# appropriate instrument and evaled by the infrastructure. It has initially
# been written to support gemini_tools.ExposureGroup and is semi-optimized.
#
# All calculations are now done in PIXELS

def pointing_in_field(ad, refpos, frac_FOV=1.0, frac_slit=1.0):

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

    # Extract pointing info from the AstroData instance.
    position = (ad.detector_x_offset(), ad.detector_y_offset())

    # TO DO: References to the field size will need changing to decimal
    # degrees once we pass absolute co-ordinates?

    # Imaging:
    if 'IMAGE' in ad.tags:
        return all([abs(x-r) < frac_FOV * ad[0].shape[0]
                    for x, r in zip(position, refpos)])

    # Long slit:
    elif 'SPECT' in ad.tags:
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

        raise NotImplementedError("FOV lookup not yet supported for NIRI "
          "spectroscopy")

    # Some engineering observation or bad mask value etc.:
    else:
        raise ValueError("Can't determine FOV for unrecognized NIRI config. "
          "(%s, %s)" % (str(ad.focal_plane_mask()), str(ad.disperser())))

