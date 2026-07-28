# This module should be removed when we have a working gempy for new astrodata

import math
import re
from functools import wraps

import numpy as np

from astropy import coordinates, units as u

# The unitDict dictionary defines the factors for the function
# convert_units
unitDict = {
    'meters': 0,
    'micrometers': -6,
    'nanometers': -9,
    'angstroms': -10,
}

def isBlank(bstring):
    return not (bstring and bstring.strip())

def removeComponentID(instr):
    """
    Remove a component ID from a filter name
    :param instr: the filter name
    :type instr: string
    :rtype: string
    :return: the filter name with the component ID removed, or `None` if the input is not a valid string
    """
    try:
        m = re.match(r"(?P<filt>.*?)_G(.*?)", instr)
    except TypeError:
        return None

    if not m:
        # There was no "_G" in the input string. Return the input string
        ret_str = str(instr)
    else:
        ret_str = str(m.group("filt"))
    return ret_str

def getComponentID(instr):
    """
    Return the ID in a component name
    :param instr: the filter name
    :type instr: string
    :rtype: string
    :return: the filter ID with the rest removed, or `None` if the input is not a valid filter name with an ID
    """
    try:
        m = re.match(r".*_(?P<id>G\d{4})", instr)
    except TypeError:
        return None

    if not m:
        ret_str = None
    else:
        ret_str = str(m.group("id"))

    return ret_str

def parse_percentile(string):
    # Given the type of string that ought to be present in the site condition
    # headers, this function returns the integer percentile number
    #
    # Is it 'Any' - ie 100th percentile?
    if (string == "Any"):
        return 100

    # Is it a xx-percentile string?
    try:
        m = re.match(r"^(\d\d)-percentile$", string)
    except TypeError:
        return None

    if m:
        return int(m.group(1))

    # We didn't recognise it
    return None


def convert_units(input_units, input_value, output_units):
    """
    :param input_units: the units of the value specified by input_value.
                        Possible values are 'meters', 'micrometers',
                        'nanometers' and 'angstroms'.
    :type input_units: string
    :param input_value: the input value to be converted from the
                        input_units to the output_units
    :type input_value: float
    :param output_units: the units of the returned value. Possible values
                         are 'meters', 'micrometers', 'nanometers' and
                         'angstroms'.
    :type output_units: string
    :rtype: float
    :return: the converted value of input_value from input_units to
             output_units
    """
    # Determine the factor required to convert the input_value from the
    # input_units to the output_units
    power = unitDict[input_units] - unitDict[output_units]
    factor = math.pow(10, power)

    # Return the converted output value
    if input_value is not None:
        return input_value * factor


def return_requested_units(input_units='nm'):
    """
    Decorator that replaces the repeated code for asMicrometers,
    asNanometers, asAngstroms. Should be replaced by a "units='nm'"
    parameter, but time is limited.

    Returns Python `float` (or a list thereof) with the same number of digits
    as np.float32, to avoid excessive precision.
    """
    def inner_decorator(fn):
        @wraps(fn)
        def gn(instance, asMicrometers=False, asNanometers=False, asAngstroms=False,
               **kwargs):
            unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
            output_units = u.m # By default
            if unit_arg_list.count(True) == 1:
                # Just one of the unit arguments was set to True. Return the
                # central wavelength in these units
                if asMicrometers:
                    output_units = u.um
                elif asNanometers:
                    output_units = u.nm
                else:
                    output_units = u.AA

            # Ensure we return a list, not an array
            # nm are the "standard" DRAGONS wavelength unit
            retval = fn(instance, **kwargs)
            if retval is None:
                return retval
            if isinstance(retval, list):
                return [
                    None if v is None else float(str(np.float32(
                        (v * u.Unit(input_units)).to(output_units).value
                    )))
                    for v in retval
                ]
            return float(str(np.float32(
                (retval * u.Unit(input_units)).to(output_units).value
            )))
        return gn
    return inner_decorator


def toicrs(frame, ra, dec, equinox=2000.0, ut_datetime=None):
    # Utility function. Converts and RA and Dec in the specified reference frame
    # and equinox at ut_datetime into ICRS. This is used by the ra and dec descriptors.

    # Assume equinox is julian calendar
    equinox = 'J{}'.format(equinox)

    # astropy doesn't understand APPT coordinates. However, it does understand
    # CIRS coordinates, and we can convert from APPT to CIRS by adding the
    # equation of origins to the RA. We can get that using ERFA.
    # To proceed with this, we first let astopy construct the CIRS frame, so
    # that we can extract the obstime object from that to pass to erfa.
    appt_frame = (frame == 'APPT')
    if frame == 'APPT':
        frame = 'cirs'
    if frame == 'FK5':
        frame = 'fk5'

    # Try this with the passed frame but, if it doesn't work, convert to "cirs"
    # If that doesn't work, then raise an error
    try:
        coords = coordinates.SkyCoord(ra=ra*u.degree, dec=dec*u.degree,
                                      frame=frame, equinox=equinox,
                                      obstime=ut_datetime)
    except ValueError:
        frame = 'cirs'
        coords = coordinates.SkyCoord(ra=ra*u.degree, dec=dec*u.degree,
                                      frame=frame, equinox=equinox,
                                      obstime=ut_datetime)

    if appt_frame:
        # Call ERFA.apci13 to get the Equation of Origin (EO).
        # We just discard the astrom context return
        try:
            # With astropy 4.2 erfa becomes a dependency and lives in an
            # independent Python package: https://github.com/liberfa/pyerfa
            import erfa
        except ImportError:
            from astropy import _erfa as erfa

        astrom, eo = erfa.apci13(coords.obstime.jd1, coords.obstime.jd2)
        # eo comes back as a single element array in radians
        eo = float(eo)
        eo = eo * u.radian
        # re-create the coords frame object with the corrected ra
        coords = coordinates.SkyCoord(ra=coords.ra+eo, dec=coords.dec,
                                      frame=coords.frame.name,
                                      equinox=coords.equinox,
                                      obstime=coords.obstime)

    # Now we can just convert to ICRS...
    icrs = coords.icrs

    # And return values in degrees
    return (icrs.ra.degree, icrs.dec.degree)

def detsec_to_pixels(ad, detx, dety):
    # Utility function to convert a location in "detector section pixels" to
    # an image extension and real pixels on that extension.
    xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
    for i, detsec in enumerate(ad.detector_section()):
        if (detx < detsec.x1 or detx >= detsec.x2 or dety < detsec.y1 or
            dety >= detsec.y2):
            continue
        datasec = ad.data_section()[i]
        return (i, datasec.x1 + (detx - detsec.x1) // xbin,
                   datasec.y1 + (dety - detsec.y1) // ybin)
    return None

### END temporary functions
