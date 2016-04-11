# This code is looked up by gempy as part of the configuration for the
# appropriate instrument and evaled by the infrastructure. It has initially
# been written to support gemini_tools.ExposureGroup and is semi-optimized. 

# Some of this should probably be factored out into an imported function in
# common with other instruments but that will need to wait for now - JT.

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

    # Extract pointing info. (currently p/q but this will be replaced with
    # RA/Dec/PA) from the AstroData instance.
    position = (pos.phu_get_key_value('POFFSET'),
                pos.phu_get_key_value('QOFFSET'))

    # TO DO: References to the field size will need changing to decimal
    # degrees once we pass absolute co-ordinates?

    # There's only an imaging mode (and probably only AO at f/32), so just
    # use that field size here.
    # The field is nominally 85", but using the average increment from the
    # WCS, along with the 2.4" gap size documented on the Web page and four
    # unilluminated rows/columns at either side, I get 82.8", so call it 83:
    dist = frac_FOV * 83.

    # Return whether the separation between the pointings is within the field
    # size along both axes:
    return all(abs(x-r) < dist for x, r in zip(position, refpos))

