# GEMINI Descriptor Calculator Utility Functions

import re

### UTILITY FUNCTIONS
## these functions are not nec. directly related to descriptors, but are
## just things many descriptors happen to have to do, e.g. like certain
## string processing, the first function here is to remove a component
## id from a filter

def removeComponentID(instr):
    m = re.match (r"(?P<filt>.*?)_G(.*?)", instr)
    if (m == None):
        #then there was no "_G" return FILTER val as filter
        return instr
    else:
        retstr = m.group("filt")
        return retstr


# This is to convert hh:mm:ss.sss to decimal degrees
def rasextodec(string):
  m = re.match("(\d+):(\d+):(\d+\.\d+)", string)
  if(m):
    hours = float(m.group(1))
    minutes = float(m.group(2))
    secs = float(m.group(3))

    minutes += (secs/60.0)
    hours += (minutes/60.0)

    degrees = hours * 15.0

    return degrees


## This is to convert [-]dd:mm:ss.sss to decimal degrees
def degsextodec(string):
  m = re.match("(-*)(\d+):(\d+):(\d+\.\d+)", string)
  if(m):
    sign = m.group(1)
    if(sign=='-'):
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

