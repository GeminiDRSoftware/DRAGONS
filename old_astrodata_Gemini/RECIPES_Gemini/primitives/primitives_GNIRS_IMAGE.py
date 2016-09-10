import sys
import pywcs
import numpy as np
import math
import numpy as np
import scipy.ndimage

from copy import deepcopy

from astrodata import AstroData
from astrodata.utils import logutils
from astrodata.utils import Lookups
from astrodata.utils.ConfigSpace  import lookup_path

from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at

from astrodata_Gemini.ADCONFIG_Gemini.lookups.GNIRS import FOV as fov

from primitives_GNIRS import GNIRSPrimitives

# ------------------------------------------------------------------------------
class GNIRS_IMAGEPrimitives(GNIRSPrimitives):
    """
    This is the class containing all of the primitives for the GNIRS_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GNIRSPrimitives'.

    """
    astrotype = "GNIRS_IMAGE"
    
    def init(self, rc):
        GNIRSPrimitives.init(self, rc)
        return rc

    def addIllumMaskToDQ(self, rc):
        """
        This primitive combines the illumination mask from the lookup directory
        into the DQ plane
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addIllumMaskToDQ", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["addIllumMaskToDQ"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get list of input and identify a suitable reference frame.
        # In most case it will be the first image, but for lamp-on, lamp-off
        # flats, one wants the reference frame to be a lamp-on since there's
        # next to no signal in the lamp-off.
        #
        # BEWARE: See note on the only GCAL_IR_OFF case below this block.   
        inputs = rc.get_inputs_as_astrodata()
        if len(inputs) > 0:
            lampons = [ad for ad in inputs if 'GCAL_IR_ON' in ad.types]
            if len(lampons) > 0:
                reference = lampons[0]
            else:
                reference = inputs[0]
        else:
            log.warning("No inputs provided to addIllumMaskToDQ")
            reference = None
                   
        # When only a GCAL_IR_OFF is available:
        # To cover the one-at-a-time mode check for a compatible list
        # if list found, try to find a lamp-on in there to use as
        # reference for the mask and the shifts.
        # The mask's name and the shifts should stored in the headers 
        # of the reference to simplify this and speed things up.
        #
        # NOT NEEDED NOW because we calling addIllumMask after the
        # lamp-offs have been subtracted.  But kept the idea here
        # in case we needed.
        
        # Fetching a corrected illumination mask with a keyhole that aligns 
        # with the science data
        if reference:
            corr_illum_ad = _position_illum_mask(reference['SCI'][0])
            if corr_illum_ad is None:
                log.warning("No illumination mask found for %s, no mask can "
                            "be added to the DQ planes of the inputs" % 
                            reference.filename)
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing (empty 'inputs')
                adoutput_list.extend(inputs)
                inputs = []

        # Loop over each input AstroData object
        for ad in inputs:
            
            # Clip the illumination mask to match the size of the 
            # input AstroData object science.
            final_illum = gt.clip_auxiliary_data(adinput=ad, aux=corr_illum_ad,
                                                 aux_type="bpm",
                                            keyword_comments=self.keyword_comments)[0]
 
            # binary_OR the illumination mask or create a DQ plane from it.
            if ad['DQ',1]:
                ad['DQ',1].data = ad['DQ',1].data | final_illum['DQ',1].data
            else:
                dqext = deepcopy(final_illum['DQ',1])
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

    def normalizeFlat(self, rc):
        """
        This primitive normalizes each science extension of the input
        AstroData object by its mean
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "normalizeFlat", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["normalizeFlat"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the normalizeFlat primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by normalizeFlat" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # Loop over each science extension in each input AstroData object
            for ext in ad['SCI']:

                # Find the mean of the good GNIRS data 
                mean_data = ext.data[np.where(ad['DQ'].data == 0)]
                mean = np.mean(mean_data, dtype=np.float64)
                
                # Divide the science extension by the mean value of the science
                # extension
                log.fullinfo("Normalizing %s[%s,%d] by dividing by the mean " \
                             "= %f" % (ad.filename, ext.extname(),
                                       ext.extver(), mean))
                ext = ext.div(mean)

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

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

##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################
    
def _position_illum_mask(adinput=None):
    """
    This function is used to reposition a GNIRS illumination mask so that 
    the keyhole matches with the science data.
        
    :param adinput: Input reference image for which the keyhole should be 
    matched
    :type adinput: Single astrodata object
    """
    from skimage.morphology import binary_dilation
    
    # Instantiate the log
    log = logutils.get_logger(__name__)

    # Fetch the illumination mask
    illum_ad = fov.fetch_illum_mask(adinput)

    # Normalizing and thresholding the science data to get a rough 
    # illumination mask. A 5x5 box around non-illuminated pixels is also 
    # flagged as non-illuminated to better handle the edge effects. The
    # limit for thresholding is set to an empirically determined value of
    # 2 for the moment - ideally, this should be replaced with a 
    # statistically determined value or function.
    addata = adinput['SCI',1].data
    adpixdata = np.copy(addata) / addata.mean()
    
    threshpixdata = np.zeros(adpixdata.shape, np.int16)
    threshpixdata[np.where(adpixdata < 2.)] = 1
    structure = np.ones((5,5))
    threshpixdata = binary_dilation(threshpixdata, structure)
    
    # This mask identifies the non-illuminated pixels.  We want
    # to feed the keyhole to the center_of_mass. We invert the mask.
    keyhole = 1 - threshpixdata

    # Finding the centre of mass of the rough pixel mask and using
    # this in comparison with the centre of mass of the illumination
    # mass to adjust the keyholes to align. Note that the  
    # center_of_mass function has switched x and y axes compared to normal.        
    comx_illummask = illum_ad.phu_get_key_value('CENMASSX')
    comy_illummask = illum_ad.phu_get_key_value('CENMASSY')
    y, x = scipy.ndimage.measurements.center_of_mass(keyhole)
    if not np.isnan(x) and not np.isnan(y):        
        dx = int(x - comx_illummask)
        dy = int(y - comy_illummask)
    else:
        log.warning("The centre of mass of %s cannot be measured, so "
                    "the illumination mask cannot be positioned and "
                    "will be used without adjustment" % adinput)
        return illum_ad
    
    # Recording the shifts in the header of the illumination mask
    log.stdinfo("Applying shifts to the illumination mask: dx = {}px, dy = "
                "{}px.".format(dx, dy))
    illum_ad.phu_set_key_value('OBSHIFTX', dx, "Relative x shift to object "
                               "frame")
    illum_ad.phu_set_key_value('OBSHIFTY', dy, "Relative y shift to object "
                               "frame")

    # Applying the offsets to the illumination mask
    illumpixdata1 = illum_ad['DQ',1].data
    illumpixdata2 = np.roll(illumpixdata1, dx, 1)
    illumpixdata3 = np.roll(illumpixdata2, dy, 0)
    illum_ad['DQ',1].data = illumpixdata3

    return illum_ad
