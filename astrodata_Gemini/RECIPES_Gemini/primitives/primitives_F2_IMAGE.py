import numpy as np
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from primitives_F2 import F2Primitives

class F2_IMAGEPrimitives(F2Primitives):
    """
    This is the class containing all of the primitives for the F2_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'F2Primitives'.
    """
    astrotype = "F2_IMAGE"
    
    def init(self, rc):
        F2Primitives.init(self, rc)
        return rc
    
    def normalize(self, rc):
        """
        This primitive normalises the input AstroData object
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "normalize", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["normalize"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the normalize primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by normalize" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Loop over each science extension in each input AstroData object
            for ext in ad["SCI"]:
                
                # Normalise the input AstroData object. Calculate the mean
                # value of the science extension
                mean = np.mean(ext.data)
                # Divide the science extension by the mean value of the science
                # extension
                log.fullinfo("Normalizing %s[%s,%d] by dividing by the mean " \
                             "= %f" % (ad.filename, ext.extname(),
                                       ext.extver(), mean))
                ext = ext.div(mean)

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
