# GEMINI Descriptor Calculator Utility Functions

import math

## UTILITY FUNCTIONS
## These functions are not necessarily directly related to descriptors, but are
## just things many descriptors happen to have to do, e.g., like certain
## string processing.

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
    return input_value * factor

# The unitDict dictionary defines the factors for the function
# convert_units
unitDict = {
    'meters':0,
    'micrometers':-6,
    'nanometers':-9,
    'angstroms':-10,
           }
