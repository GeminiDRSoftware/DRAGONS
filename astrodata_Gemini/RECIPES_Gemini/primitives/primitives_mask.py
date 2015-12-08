import os
from copy import deepcopy
from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils import Lookups
from astrodata.utils.ConfigSpace  import lookup_path
from gempy.gemini import gemini_tools as gt
from primitives_GENERAL import GENERALPrimitives

class MaskPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the mask-related primitives
    for the GEMINI level of the type hierarchy tree. It inherits all
    the primitives from the level above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def addObjectMaskToDQ(self, rc):
        """
        This primitive combines the object mask in a OBJMASK extension
        into the DQ plane
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)

        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addObjectMaskToDQ", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["addObjectMaskToDQ"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            for sciext in ad["SCI"]:
                extver = sciext.extver()
                dqext = ad["DQ",extver]
                mask = ad["OBJMASK",extver]
                if mask is None:
                    log.warning("No object mask present for "\
                                    "%s[SCI,%d]; "\
                                    "cannot apply object mask" %
                                (ad.filename,extver))
                else:
                    if dqext is not None:
                        ad["DQ",extver].data = dqext.data | mask.data
                    else:
                        dqext = deepcopy(mask)
                        dqext.rename_ext("DQ",extver)
                        ad.append(dqext)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

