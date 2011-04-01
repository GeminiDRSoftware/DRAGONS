# The string module contains utilities functions that manipulate strings

import re

def removeComponentID(instr):
    """
    Remove a component ID from a filter name
    :param instr: the filter name
    :type instr: string
    :rtype: string
    :return: the filter name with the component ID removed
    """
    m = re.match (r"(?P<filt>.*?)_G(.*?)", instr)
    if not m:
        # There was no "_G" in the input string. Return the input string
        ret_str = str(instr)
    else:
        ret_str = str(m.group("filt"))

    return ret_str

def section_to_tuple(section):
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

