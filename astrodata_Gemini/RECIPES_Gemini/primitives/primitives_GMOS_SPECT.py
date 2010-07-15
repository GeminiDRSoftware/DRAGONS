#Author: Kyle Mede, June 2010
#from Reductionobjects import Reductionobject
from primitives_GEMINI import GEMINIPrimitives
# All GEMINI IRAF task wrappers.
import time
from astrodata.adutils import filesystem
from astrodata.adutils.future import gemLog
from astrodata import IDFactory
from astrodata import Descriptors
from astrodata.data import AstroData

from pyraf.iraf import tables, stsdas, images
from pyraf.iraf import gemini
import pyraf
import iqtool
from iqtool.iq import getiq
from gempy.instruments.gmos import *

log=gemLog.getGeminiLog()

import pyfits
import numdisplay
import string

yes = pyraf.iraf.yes
no = pyraf.iraf.no

# NOTE, the sys.stdout stuff is to shut up gemini and gmos startup... some primitives
# don't execute pyraf code and so do not need to print this interactive 
# package init display (it shows something akin to the dir(gmos)
import sys, StringIO, os
SAVEOUT = sys.stdout
capture = StringIO.StringIO()
sys.stdout = capture
gemini()
gemini.gmos()
sys.stdout = SAVEOUT

class GMOS_SPECTException:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised in Recipe System"):
        """This constructor takes a message to print to the user."""
        self.message = msg
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message


class GMOS_SPECTPrimitives(GEMINIPrimitives):
    astrotype = "GMOS_SPECT"
    
    def init(self, rc):
        
        if "iraf" in rc and "adata" in rc["iraf"]:
            pyraf.iraf.set (adata=rc["iraf"]['adata'])  
        else:
            # @@REFERENCEIMAGE: used to set adata path for primitives
            if len(rc.inputs) > 0:
                (root, name) = os.path.split(rc.inputs[0].filename)
                pyraf.iraf.set (adata=root)
                if "iraf" not in rc:
                    rc.update({"iraf":{}})
                if "adata" not in rc["iraf"]:
                    rc["iraf"].update({"adata":root}) 
        
        GEMINIPrimitives.init(self, rc)
        return rc

    
    def display(self, rc):
        from adutils.future import gemDisplay
        import pyraf
        from pyraf import iraf
        iraf.set(stdimage='imtgmos')
        ds = gemDisplay.getDisplayService()
        for i in range(0, len(rc.inputs)):   
            inputRecord = rc.inputs[i]
            gemini.gmos.gdisplay( inputRecord.filename, i+1, fl_imexam=iraf.no,
                Stdout = rc.getIrafStdout(), Stderr = rc.getIrafStderr() )
            # this version had the display id conversion code which we'll need to redo
            # code above just uses the loop index as frame number
            #gemini.gmos.gdisplay( inputRecord.filename, ds.displayID2frame(rq.disID), fl_imexam=iraf.no,
            #    Stdout = coi.getIrafStdout(), Stderr = coi.getIrafStderr() )
        yield rc
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
    #$$$$$$$$$$$$$$$$$$$$ NEW STUFF BY KYLE FOR: PREPARE $$$$$$$$$$$$$$$$$$$$$
    '''
    These are the primitives for 'prepare' that are just to check how the general primitives in 
    primitives_GEMINI will work for a GMOS_SPEC image and its MDF file.  Since these are just to check 
    how it will work, they are operational but not thuroughly tested or robust and will be needing more
    work when we start writing the rest of the prims for GMOS_SPEC.  Because of this, sorry for the lack
    of commenting if you end up reading this code.
    '''

    def standardizeInstrumentHeaders(self,rc):
        '''
        makes the changes and additions to the headers of the input files that are instrument specific
        '''
        
        try: 
            for ad in rc.getInputs(style="AD"): 
                log.status('calling stdInstHdrs','status')
                stdInstHdrs(ad) 
                
                log.status('instrument headers fixed','status') 
                
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc
    #-----------------------------------------------------------------------
    def attachMDF(self,rc):
        '''
        this works to add an MDF if there is a MASKNAME in the images PHU only.  
        will be upgraded later, early testing complete
        '''
        
        try:           
            for ad in rc.getInputs(style ='AD'):
                infilename = ad.filename
                log.status('file having MDF attached= '+infilename,'status')
                
                pathname = 'kyles_test_images/' #$$$$ HARDCODED FOR NOW, TILL FIX COMES FROM CRAIG
                maskname = ad.phuGetKeyValue("MASKNAME")
                log.stdinfo("maskname = "+maskname,'stdinfo')
                inMDFname = 'kyles_test_images/'+maskname +'.fits'
                log.status('input MDF file = '+inMDFname,'status')
                admdf = AstroData(inMDFname)
                admdf.renameExt('MDF',1)  #$$$ HARDCODED EXTVER=1 FOR NOW, CHANGE LATER??
                admdf.extSetKeyValue(len(admdf)-1,'EXTNAME', 'MDF',"Extension name" )
                admdf.extSetKeyValue(len(admdf)-1,'EXTVER', 1,"Extension version" ) #$$$ HARDCODED EXTVER=1 FOR NOW, CHANGE LATER?? this one is to add the comment
                
                log.debug(admdf[0].getHeader())
                log.debug(admdf.info())
                
                ad.append(moredata=admdf)  
                log.status(ad.info(),'status')
                
                ad.filename=fileNameUpdater(ad.filename,postpend=rc["outsuffix"], strip=False)
                rc.reportOutput(ad)
                
                #addMDF(ad,mdf)     #$$ the call to the tool box function, currently not in use
                log.status('finished adding the MDF','status')
        except:
            log.critical("Problem preparing the image.", 'critical')
            raise 
        
        yield rc
    #------------------------------------------------------------------------
    def validateInstrumentData(self,rc):
        '''
        instrument specific validations for the input file
        '''
        
        try:        
            for ad in rc.getInputs(style="AD"):
                log.status('validating data for file = '+ad.filename,'status')
                valInstData(ad)
                log.status('data validated for file = '+ad.filename,'status')
                
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc
   #----------------------------------------------------------------------------------
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Prepare primitives end here $$$$$$$$$$$$$$$$$$$$$$$$$$$$
         
def pyrafBoolean(pythonBool):
    '''
    a very basic function to reduce code repetition that simply 'casts' any given 
    Python boolean into a pyraf/iraf one for use in the CL scripts
    '''
    
    if pythonBool:
        return 'pyraf.iraf.yes'
    elif  not pythonBool:
        return 'pyraf.iraf.'
    else:
        print "DANGER DANGER Will Robinson, pythonBool not True or False"        
    #$$$$$$$$$$$$$$$$$$$$$$$ END OF KYLES NEW STUFF $$$$$$$$$$$$$$$$$$$$$$$$$$
        