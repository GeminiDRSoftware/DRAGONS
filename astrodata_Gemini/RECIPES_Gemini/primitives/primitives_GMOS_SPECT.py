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
from astrodata.Errors import PrimitiveError
from gempy.instruments import geminiTools  as gemt
from gempy.instruments import gmosTools  as gmost

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
        """
        This primitive is called by standardizeHeaders to makes the changes and 
        additions to the headers of the input files that are GMOS_SPEC 
        specific.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logLevel=rc['logLevel'])
        try: 
            for ad in rc.getInputs(style="AD"): 
                log.debug('calling stdInstHdrs for '+ad.filename)
                gmost.stdInstHdrs(ad) 
                
                log.status('instrument headers fixed') 
                
        except:
            log.critical("Problem preparing the image.")
            raise PrimitiveError("Problem preparing the image.")
        
        yield rc

    def attachMDF(self,rc):
        """
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
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logLevel=rc['logLevel'])
        try:           
            for ad in rc.getInputs(style ='AD'):
                infilename = ad.filename
                log.status('file having MDF attached= '+infilename)
                
                pathname = 'kyles_test_images/' #$$$$ HARDCODED FOR NOW, TILL FIX COMES FROM CRAIG
                maskname = ad.phuGetKeyValue("MASKNAME")
                log.stdinfo("maskname = "+maskname,'stdinfo')
                inMDFname = 'kyles_test_images/'+maskname +'.fits'
                log.status('input MDF file = '+inMDFname)
                admdf = AstroData(inMDFname)
                admdf.renameExt('MDF',1)  #$$$ HARDCODED EXTVER=1 FOR NOW, CHANGE LATER??
                admdf.extSetKeyValue(len(admdf)-1,'EXTNAME', 'MDF',"Extension name" )
                admdf.extSetKeyValue(len(admdf)-1,'EXTVER', 1,"Extension version" ) #$$$ HARDCODED EXTVER=1 FOR NOW, CHANGE LATER?? this one is to add the comment
                
                log.debug(admdf[0].getHeader())
                log.debug(admdf.info())
                
                ad.append(moredata=admdf)  
                log.status(ad.info())
                
                ad.filename=gemt.fileNameUpdater(ad.filename,
                                                 suffix=rc["suffix"], 
                                                 strip=False)
                rc.reportOutput(ad)
                
                log.status('finished adding the MDF')
        except:
            log.critical("Problem preparing the image.")
            raise PrimitiveError("Problem preparing the image.")
        yield rc

    def validateInstrumentData(self,rc):
        """
        This primitive is called by validateData to validate the GMOS_SPEC 
        specific data checks for all input files.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (ie. 'critical'  
                        , 'status', 'fullinfo'...)
        """
        log = gemLog.getGeminiLog(logLevel=rc['logLevel'])
        try:        
            for ad in rc.getInputs(style="AD"):
                log.debug('calling gmost.valInstData for '+ad.filename)
                gmost.valInstData(ad)
                log.status('data validated for file = '+ad.filename)
                
        except:
            log.critical('Problem validating one of  '+rc.inputsAsStr())
            raise PrimitiveError('Problem validating one of  '+rc.inputsAsStr())
        yield rc

