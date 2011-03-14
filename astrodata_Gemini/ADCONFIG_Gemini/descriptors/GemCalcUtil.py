# GEMINI Descriptor Calculator Utility Functions

import math
import re

## UTILITY FUNCTIONS
## These functions are not necessarily directly related to descriptors, but are
## just things many descriptors happen to have to do, e.g., like certain
## string processing.

def removeComponentID(instr):
    """
    Remove a component id from a filter name
    """
    m = re.match (r"(?P<filt>.*?)_G(.*?)", instr)
    if not m:
        # There was no "_G" in the input string. Return the input string
        ret_str = instr
    else:
        ret_str = m.group("filt")

    return ret_str

def rasextodec(string):
    """
    Convert hh:mm:ss.sss to decimal degrees
    """
    m = re.match("(\d+):(\d+):(\d+\.\d+)", string)
    if m:
        hours = float(m.group(1))
        minutes = float(m.group(2))
        secs = float(m.group(3))
        
        minutes += (secs/60.0)
        hours += (minutes/60.0)
        
        degrees = hours * 15.0
    
    return degrees

def degsextodec(string):
    """
    Convert [-]dd:mm:ss.sss to decimal degrees
    """
    m = re.match("(-*)(\d+):(\d+):(\d+\.\d+)", string)
    if m:
        sign = m.group(1)
        if sign == '-':
            sign = -1.0
        else:
            sign = +1.0
        
        degs = float(m.group(2))
        minutes = float(m.group(3))
        secs = float(m.group(4))
        
        minutes += (secs/60.0)
        degs += (minutes/60.0)
        
        degs *= sign
    
    return degs

def convert_units(self, input_units, input_value, output_units):
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
    power = self.unitDict[input_units] - self.unitDict[output_units]
    factor = math.pow(10, power)
    
    # Return the converted output value
    return input_value * factor

# The unitDict dictionary defines the factors for the function
# convert_units
unitDict = {
    'meters':0,
    'micrometers':-6,
    'nanometers':-9,
    'angstroms':-10,
           }

def section_to_tuple(self, section):
    """
    Convert the input section in the form [x1:x2,y1:y2] to a tuple in the
    form (x1 - 1, x2 - 1, y1 - 1, y2 - 1), where x1, x2, y1 and y2 are
    integers. The values in the output tuple are converted to use 0-based
    indexing, making it compatible with numpy.
    :param section: the section (in the form [x1:x2,y1:y2]) to be
                    converted to a tuple
    :type section: string
    :rtype: tuple
    :return: the converted section as a tuple that uses 0-based indexing
             in the form (x1 - 1, x2 - 1, y1 - 1, y2 - 1)
    """
    # Strip the square brackets from the input section and then create a
    # list in the form ['x1:x2', 'y1:y2']
    xylist = section.strip('[]').split(',')
    
    # Create variables containing the single x1, x2, y1 and y2 values
    x1 = int(xylist[0].split(':')[0]) - 1
    x2 = int(xylist[0].split(':')[1]) - 1
    y1 = int(xylist[1].split(':')[0]) - 1
    y2 = int(xylist[1].split(':')[1]) - 1

    # Return the tuple in the form (x1, x2, y1, y2)
    return (x1, x2, y1, y2)
