# GEMINI Descriptor Calculator Utility Functions

import re

### UTILITY FUNCTIONS
## these functions are not nec. directly related to descriptors, but are
## just things many descriptors happen to have to do, e.g. like certain
## string processing, the first function here is to remove a component
## id from a filter

def removeComponentID(instr):
    m = re.match (r"(?P<filt>.*?)_(.*?)", instr)
    if (m == None):
        #then there was no "_" return FILTER val as filter
        return instr
    else:
        retstr = m.group("filt")
        return retstr        
