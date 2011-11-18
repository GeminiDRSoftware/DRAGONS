from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import preprocessing as pp
from gempy.science import resample as rs
from gempy.science import display as ds
from primitives_GENERAL import GENERALPrimitives

class DisplayPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the display primitives
    for the GEMINI level of the type hierarchy tree. It inherits
    all the primitives from the level above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def display(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "display", "starting"))
        
        # Loop over each input AstroData object in the input list
        frame = rc["frame"]
        for ad in rc.get_inputs_as_astrodata():
            
            if frame>16:
                log.warning("Too many images; only the first 16 are displayed.")
                break

            try:
                ad = ds.display(adinput=ad, frame=frame, extname=rc["extname"],
                                zscale=rc["zscale"], threshold=rc["threshold"])
            except:
                log.warning("Could not display %s" % ad.filename)
            
            frame+=1
        
        yield rc
