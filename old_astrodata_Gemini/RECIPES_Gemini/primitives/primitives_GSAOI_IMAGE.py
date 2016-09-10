import sys
import pywcs
import numpy as np

from astrodata import AstroData
from astrodata.utils import logutils
from astrodata.utils import Lookups

from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at

from primitives_GSAOI import GSAOIPrimitives

class GSAOI_IMAGEPrimitives(GSAOIPrimitives):
    """
    This is the class containing all of the primitives for the GSAOI_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GSAOIPrimitives'.
    """
    astrotype = "GSAOI_IMAGE"
    
    def init(self, rc):
        GSAOIPrimitives.init(self, rc)
        return rc

    def darkCorrectAndStackFlats(self, rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "darkCorrectAndStackFlats", 
                                 "starting"))
 
        adinput = rc.get_inputs_as_astrodata()        
        recipe_list = []
        
        # For Z- and J-band GSAOI, there are only illuminated dome flats. GSAOI
        # does not have dark frames, but the dark current is not significant.
        if ((adinput[0].wavelength_band().as_pytype() == 'Z') or 
            (adinput[0].wavelength_band().as_pytype() == 'J')):
            recipe = "stackFrames"
            recipe_list.append(recipe)
            log.stdinfo("{} band uses only illuminated dome flats. No dark "
                        "frames are taken, but the dark current is not "
                        "significant"
                        .format(adinput[0].wavelength_band().as_pytype()))
        # For GSAOI H-band and above, the equivalent to lamp on and lamp off 
        # flats are taken.
        else:
            recipe = "lampOnLampOff"
            recipe_list.append(recipe)
            log.stdinfo("For {} band, the dome-flat equivalents to lamp on "
                        "and lamp off flats exist, so these are subtracted "
                        "for dark correction"
                        .format(adinput[0].wavelength_band().as_pytype()))
            
        rc.run("\n".join(recipe_list))
        
        yield rc
