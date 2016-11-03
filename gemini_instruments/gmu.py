# This module should be removed when we have a working gempy for new astrodata

import math
import re
from astropy import coordinates, units, _erfa

# The unitDict dictionary defines the factors for the function
# convert_units
unitDict = {
    'meters': 0,
    'micrometers': -6,
    'nanometers': -9,
    'angstroms': -10,
}


def removeComponentID(instr):
    """
    Remove a component ID from a filter name
    :param instr: the filter name
    :type instr: string
    :rtype: string
    :return: the filter name with the component ID removed
    """
    m = re.match(r"(?P<filt>.*?)_G(.*?)", instr)
    if not m:
        # There was no "_G" in the input string. Return the input string
        ret_str = str(instr)
    else:
        ret_str = str(m.group("filt"))
    return ret_str


def sectionStrToIntList(section):
    """
    Convert the input section in the form '[x1:x2,y1:y2]' to a tuple in the
    form (x1 - 1, x2, y1 - 1, y2), where x1, x2, y1 and y2 are
    integers. The values in the output tuple are converted to use 0-based and
    non-inclusive indexing, making it compatible with numpy.

    :param section: the section (in the form [x1:x2,y1:y2]) to be
                    converted to a tuple
    :type section: string

    :rtype: tuple
    :return: the converted section as a tuple that uses 0-based and
             non-inclusive in the form (x1 - 1, x2, y1 - 1, y2)
    """
    # Strip the square brackets from the input section and then create a
    # list in the form ['x1:x2','y1:y2']
    xylist = section.strip('[]').split(',')

    # Create variables containing the single x1, x2, y1 and y2 values
    x1 = int(xylist[0].split(':')[0]) - 1
    x2 = int(xylist[0].split(':')[1])
    y1 = int(xylist[1].split(':')[0]) - 1
    y2 = int(xylist[1].split(':')[1])

    # Return the tuple in the form (x1 - 1, x2, y1 - 1, y2)
    return (x1, x2, y1, y2)


def parse_percentile(string):
    # Given the type of string that ought to be present in the site condition
    # headers, this function returns the integer percentile number
    #
    # Is it 'Any' - ie 100th percentile?
    if (string == "Any"):
        return 100

    # Is it a xx-percentile string?
    m = re.match("^(\d\d)-percentile$", string)
    if (m):
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

### END temporary functions
