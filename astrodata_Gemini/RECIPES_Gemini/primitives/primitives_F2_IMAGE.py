from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import preprocessing as pp
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
        This primitive normalises the input flat
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "normalize", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the normalize primitive has been run previously
            if ad.phu_get_key_value("NORMALIZ"):
                log.warning("%s has already been processed by normalize" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the normalize_flat_image user level function
            ad = pp.normalize_flat_image(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
