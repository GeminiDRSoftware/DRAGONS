# This code is looked up by gempy as part of the configuration for the
# appropriate instrument and evaled by the infrastructure. It has initially
# been written to support gemini_tools.ExposureGroup and is semi-optimized. 

# Some of this should probably be factored out into an imported function in
# common with other instruments but that will need to wait for now - JT.
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

    # Extract pointing info.
    position = (ad.detector_x_offset(), ad.detector_y_offset())

    # TO DO: References to the field size will need changing to decimal
    # degrees once we pass absolute co-ordinates?

    # There's only an imaging mode (and probably only AO at f/32), so just
    # use that field size here. With gaps between the arrays, it's ~4240 pixels
    dist = frac_FOV * 4240

    # Return whether the separation between the pointings is within the field
    # size along both axes:
    return all(abs(x-r) < dist for x, r in zip(position, refpos))

