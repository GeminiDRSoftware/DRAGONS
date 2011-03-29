import os, sys, re
from sets import Set

import time
from astrodata.ReductionObjects import PrimitiveSet
from astrodata.adutils import gemLog
from astrodata.Errors import PrimitiveError
from astrodata import IDFactory
from gempy import geminiTools as gemt
import numpy as np
import pyfits as pf
from datetime import datetime
import shutil

class GENERALPrimitives(PrimitiveSet):
    """ 
    This is the class of all primitives for the GEMINI astrotype of 
    the hierarchy tree.  It inherits all the primitives to the level above
    , 'PrimitiveSet'.
    
    """
    astrotype = 'GENERAL'
    
    def init(self, rc):
        return 
    init.pt_hide = True
    
    def addInputs(self, rc):
        log = gemLog.getGeminiLog(logType=rc['logType'],logLevel=rc['logLevel'])
        import glob as gl
        if rc["files"] == None:
            glob = "./*.fits"
        else:
            glob = rc["files"]

        log.status("Listing for: "+ glob)
        files = gl.glob(glob)
        files.sort()
        if len(files) == 0:
            log.status("No files")    
        else:
            log.status("\t"+"\n\t".join(files))
        yield rc
        add = True # rc["inputs"]
        if add:
            rc.addInput(files)
        yield rc
        
    def copy(self, rc):
        for ad in rc.getInputsAsAstroData():
            from copy import deepcopy
            nd = deepcopy(ad)
            nd.filename = "copy_"+os.path.basename(ad.filename)
            rc.reportOutput(nd)
        yield rc
    def setInputs(self, rc):
        files = rc["files"]
        if files != None:
            a = files.split(" ")
            if len(a)>0:
                rc.addInput(a)
        yield rc
    def clearInputs(self, rc):
        rc.clearInput()
        yield rc
        
    def listDir(self,rc):
        log = gemLog.getGeminiLog(logType=rc['logType'],logLevel=rc['logLevel'])
        if rc["dir"] == None:
            thedir = "."
        else:   
            thedir = rc["dir"]
        log.status("Listing for: "+ thedir)
        files = os.listdir(thedir)
        sfiles = []
        for f in files:
            if f[-5:].lower() == ".fits":
                sfiles.append(f)
        sfiles.sort()
        if len(sfiles) == 0:
            log.status("No FITS files")    
        else:
            log.status("\n\t".join(sfiles))
        yield rc
