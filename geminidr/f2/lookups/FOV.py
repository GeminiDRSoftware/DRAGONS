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

    # Since this function gets looked up & evaled, we have to do any
    # essential imports in-line (but Python caches them):
    import re

    # Extract pointing info. (currently p/q but this will be replaced with
    # RA/Dec/PA) from the AstroData instance.
    position = (ad.detector_x_offset(), ad.detector_y_offset())

    # GeMS truncates the FOV to 2' with 0.09" pixels
    is_ao = ad.is_ao()

    # Imaging:
    if 'IMAGE' in ad.tags:
        diameter = 1300 if is_ao else 2048
        return sum([(x-r)**2 for x, r in zip(position,refpos)]) < (frac_FOV * diameter)**2
    # Long slit:
    elif 'LS' in ad.tags:
        # Parse slit width in pixels from mask name:
        mask = ad.focal_plane_mask()
        try:
            width = float(re.match("([0-9]{1,2})pix-slit$", mask).group(1))
        except (AttributeError, TypeError, ValueError):
            raise ValueError("Failed to parse width for F2 slit (%s)" % mask)

        # TO DO: This first number should be correct but the Web page doesn't
        # confirm it, so check with the instrument team:
        dist = 1300 if is_ao else 1460

        # Tuple of (adjusted) field extent in each direction (X, Y)
        # The slit length is from the Web page, under long-slit spectroscopy.
        dist = (frac_FOV * dist, frac_slit * 0.5 * width)

        # Is the position within the field boundaries along both/all axes?
        return all([abs(x-r) < d for x, r, d in zip(position, refpos, dist)])

    # MOS:
    elif 'MOS' in ad.tags:
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

