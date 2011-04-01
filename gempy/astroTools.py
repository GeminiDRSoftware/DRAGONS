# The astroTools module contains astronomy specific utilities functions

import re

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
