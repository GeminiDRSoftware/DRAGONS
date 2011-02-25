#Author: Kyle Mede, June 2010
#from Reductionobjects import Reductionobject
from primitives_GMOS import GMOSPrimitives, pyrafLoader
# All GEMINI IRAF task wrappers.
import time
from astrodata.adutils import filesystem
from astrodata.adutils import gemLog
from astrodata import IDFactory
from astrodata import Descriptors
from astrodata.data import AstroData

from gempy.instruments import geminiTools  as gemt
from gempy.instruments import gmosTools  as gmost

log=gemLog.getGeminiLog()

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

class GMOS_SPECTPrimitives(GMOSPrimitives):
    """
    This is the class of all primitives for the GMOS_SPECT level of the type 
    hierarchy tree.  It inherits all the primitives to the level above
    , 'GMOSPrimitives'.
    """
    astrotype = "GMOS_SPECT"
    
    def init(self, rc):
        GMOSPrimitives.init(self, rc)
        return rc

    def standardizeInstrumentHeaders(self,rc):
        '''
        This primitive is called by standardizeHeaders to makes the changes and 
        additions to the headers of the input files that are GMOS_SPEC 
        specific.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        '''
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try: 
            for ad in rc.getInputs(style="AD"): 
                log.status('calling stdInstHdrs','status')
                gmost.stdInstHdrs(ad) 
                
                log.status('instrument headers fixed','status') 
                
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        
        yield rc

    def attachMDF(self,rc):
        '''
        This primitive is used to add an MDF if there is a MASKNAME in the   
        images PHU only. It is called by the primitive standardizeStructure
        during the prepare recipe if the parameter addMDF=True.
        ***********************************************
        will be upgraded later, early testing complete
        ***********************************************
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        '''
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
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
                
                ad.filename=gemt.fileNameUpdater(ad.filename,suffix=rc["suffix"], strip=False)
                rc.reportOutput(ad)
                
                log.status('finished adding the MDF','status')
        except:
            log.critical("Problem preparing the image.", 'critical')
            raise 
        yield rc

    def validateInstrumentData(self,rc):
        '''
        This primitive is called by validateData to validate the GMOS_SPEC 
        specific data checks for all input files.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: int. 
                        This value can be set for each primitive individually 
                        in a recipe only (ie. not in the parameter file). 
                        If no value is specified during the recipe, the value 
                        set during the call to reduce or its default (2) will 
                        be used.
        '''
        log = gemLog.getGeminiLog(logLevel=int(rc['logLevel']))
        try:        
            for ad in rc.getInputs(style="AD"):
                log.status('validating data for file = '+ad.filename,'status')
                gmost.valInstData(ad)
                log.status('data validated for file = '+ad.filename,'status')
                
        except:
            log.critical("Problem preparing the image.",'critical')
            raise 
        yield rc

