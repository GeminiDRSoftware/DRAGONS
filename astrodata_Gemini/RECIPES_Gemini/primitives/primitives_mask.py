import os
from copy import deepcopy
from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils import Lookups
from astrodata.utils.ConfigSpace  import lookup_path
from gempy.gemini import gemini_tools as gt
from primitives_GENERAL import GENERALPrimitives

import math
import numpy as np

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

    def applyDQPlane(self, rc):
        """
        This primitive sets the value of pixels in the science plane according
        to flags from the DQ plane. 

        :param replace_flags: An integer indicating which DQ plane flags are 
                              to be applied, e.g. a flag of 70 indicates 
                              2 + 4 + 64. The default of 255 flags all values 
                              up to 128.
        :type replace_flags: str

        :param replace_value: Either "median" or "average" to replace the 
                              bad pixels specified by replace_flags with 
                              the median or average of the other pixels, or
                              a numerical value with which to replace the
                              bad pixels. 
        :type replace_value: str        
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "applyDQPlane", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["applyDQPlane"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get the inputs from the rc
        replace_flags = rc["replace_flags"]
        replace_value = rc["replace_value"]
        
        # Check which flags should be replaced
        count = 0
        flag_list = []
        for digit in str(bin(replace_flags)[2:])[::-1]:
            if digit == "1":
                flag_list.append(int(math.pow(2,count)))
            count +=1
        log.stdinfo("The flags {} will be applied".format(flag_list))  
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            for sciext in ad["SCI"]:
                extver = sciext.extver()
                dqext = ad["DQ",extver]
            
                # Check that there is a DQ extension
                if dqext is None:
                    log.warning("No DQ plane exists for %s, so the correction "
                                "cannot be applied" % ad.filename)
                    continue
        
                # If we need the median or average, we need to find where the 
                # pixels are good
                if replace_value in ["median", "average"]:
                    good_pixels = np.where(dqext.data & replace_flags == 0)
                    if replace_value == "median":
                        rep_val = np.median(sciext.data[good_pixels])
                        log.fullinfo("Replacing bad pixels in {0} with the "
                                     "median of the good data: {1:.2f}".
                                     format('["SCI",'+str(extver)+']', rep_val))
                    else:
                        rep_val = np.average(sciext.data[good_pixels])
                        log.fullinfo("Replacing bad pixels in {0} with the "
                                     "average of the good data: {1:.2f}".
                                     format('["SCI",'+str(extver)+']', rep_val))
                else:
                    try:
                        rep_val = float(replace_value)
                        log.stdinfo("Replacing bad pixels in {0} with the user "
                                    "input value: {1:.2f}".
                                    format('["SCI",'+str(extver)+']', rep_val))
                    except:
                        log.warning("Value for replacement should be "
                                    "specified as 'median', 'average', or as "
                                    "a number")
                        continue

                # Replacing the bad pixel values
                bad_pixels = np.where(dqext.data & replace_flags != 0)    
                sciext.data[bad_pixels] = rep_val
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
    

