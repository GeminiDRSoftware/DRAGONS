from astrodata.adutils import logutils
from gempy.gemini import gemini_tools as gt
from primitives_bookkeeping import BookkeepingPrimitives
from primitives_calibration import CalibrationPrimitives
from primitives_display import DisplayPrimitives
from primitives_mask import MaskPrimitives
from primitives_photometry import PhotometryPrimitives
from primitives_preprocess import PreprocessPrimitives
from primitives_qa import QAPrimitives
from primitives_register import RegisterPrimitives
from primitives_resample import ResamplePrimitives
from primitives_stack import StackPrimitives
from primitives_standardize import StandardizePrimitives

class GEMINIPrimitives(BookkeepingPrimitives,CalibrationPrimitives,
                       DisplayPrimitives, MaskPrimitives,
                       PhotometryPrimitives,PreprocessPrimitives,
                       QAPrimitives,RegisterPrimitives,
                       ResamplePrimitives,StackPrimitives,
                       StandardizePrimitives):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        BookkeepingPrimitives.init(self, rc)
        CalibrationPrimitives.init(self, rc)
        DisplayPrimitives.init(self, rc)
        MaskPrimitives.init(self, rc)
        PhotometryPrimitives.init(self, rc)
        PreprocessPrimitives.init(self, rc)
        QAPrimitives.init(self, rc)
        RegisterPrimitives.init(self, rc)
        ResamplePrimitives.init(self, rc)
        StackPrimitives.init(self, rc)
        StandardizePrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def standardizeGeminiHeaders(self, rc):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of Gemini data.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeGeminiHeaders",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeGeminiHeaders"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the standardizeGeminiHeaders primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by "
                            "standardizeGeminiHeaders" % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are common to all Gemini data.
            log.status("Updating keywords that are common to all Gemini data")
            
            # Original name
            ad.store_original_name()
            
            # Number of science extensions
            gt.update_key(adinput=ad, keyword="NSCIEXT",
                          value=ad.count_exts("SCI"), comment=None,
                          extname="PHU") 
            
            # Number of extensions
            gt.update_key(adinput=ad, keyword="NEXTEND", value=len(ad),
                          comment=None, extname="PHU")
            
            # Physical units (assuming raw data has units of ADU)
            gt.update_key(adinput=ad, keyword="BUNIT", value="adu",
                          comment=None, extname="SCI")
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects 
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
