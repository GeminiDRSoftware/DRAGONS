#Author: Kyle Mede, May 2010
#This module provides functions to be used by all GMOS primitives.
import os

import pyfits as pf
import numpy as np
from copy import deepcopy
import time
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
from astrodata.Errors import ToolboxError
    
def statsecConverter(statsecStr):
    try:
        (left,right)=statsecStr.split('][')
        (extname,extver)=left.split('[')[1].split(',')
        (X,Y)=right.split(']')[0].split(',')
        (x1,x2)=X.split(':')
        (y1,y2)=Y.split(':')
        
        return extname.upper(), int(extver), int(x1), int(x2), int(y1), int(y2)
    except: 
        raise ToolboxError('An problem occured trying to convert the statsecStr '+
                     statsecStr+' into its useful components')         
#----------------------------------------------------------------------------       