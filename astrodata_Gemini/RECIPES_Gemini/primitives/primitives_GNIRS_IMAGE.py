import sys
import pywcs
import numpy as np

from astrodata import AstroData
from astrodata.utils import logutils
from astrodata.utils import Lookups
from astrodata.utils.ConfigSpace  import lookup_path
import numpy as np
import scipy.ndimage

from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at

from primitives_GNIRS import GNIRSPrimitives

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

        # Fetching a corrected illumination mask with a keyhole that aligns 
        # with the science data
        reference = None
        for ad in rc.get_inputs_as_astrodata():
            if reference is None:
                reference = ad
            if('GCAL_IR_ON' in ad.types):
                reference = ad
                break
        corr_illum_ad = _position_illum_mask(reference['SCI'][0])
            
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Check that the illumination mask is present
            if corr_illum_ad is None:
                log.warning("No illumination mask found for %s, no mask can "
                            "be added to the DQ plane" % ad.filename)
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
                
            final_illum = None
            # Clip the illumination mask to match the size of the 
            # input AstroData object science and pad with overscan 
            # region, if necessary
            final_illum = gt.clip_auxiliary_data(adinput=ad, 
                                                 aux=corr_illum_ad,
                                                 aux_type="cal")[0]
            illum_data = final_illum['SCI'].data

            # Loop over each science extension in each input AstroData object
            for sciext in ad['SCI']:

                extver = sciext.extver()
                dqext = ad["DQ",extver]
                
                if illum_data is None:
                    log.warning("No illumination mask present for %s[SCI,%d]; "
                                "cannot apply illumination mask" % 
                                (ad.filename,extver))
                else:
                    if dqext is not None:
                        ad["DQ",extver].data = dqext.data | illum_data
                    else:
                        dqext = deepcopy(final_illum['SCI'])
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

    def applyIllumMask(self, rc):
        """
        This primitive applies the illumination mask for GNIRS, setting all 
        data not in the illuminated region to zero.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "applyIllumMask", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["applyIllumMask"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Check whether the myScienceStep primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by applyIllumMask"
                            % ad.filename)

                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # Check that there is a DQ extension
            if ad['DQ'] is None:
                log.warning("No DQ plane exists for %s, so the illumination"
                            "mask cannot be applied" % ad.filename)
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
                    
            # Loop over each science extension in each input AstroData object
            for ext in ad['SCI']:

                # Find the median value of the region inside the FOV
                median = np.median(ext.data[np.where(ad['DQ'].data != 64)])
                
                # Set values outside of the GNIRS FOV to match the median
                ext.data[np.where(ad['DQ'].data == 64)] = median
                
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
            gt.mark_history(adinput=ad, keyword=timestamp_key)

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
    # Instantiate the log
    log = logutils.get_logger(__name__)

    # Fetch the illumination mask
    illum_mask_dict = Lookups.get_lookup_table("Gemini/GNIRS/IllumMaskDict",
                                               "illum_masks")
    key1 = adinput.camera().as_pytype()
    filter = adinput.filter_name(pretty=True).as_pytype()
    if filter in ['Y', 'J', 'H', 'K']:
        key2 = 'Broadband'
    elif filter in ['JPHOT', 'HPHOT', 'KPHOT', 'H2', 'PAH']:
        key2 = 'Narrowband'
    else:
        log.warning("Unrecognised filter, no illumination mask can "
                    "be found for" % adinput.filename)
        return None
                
    key = (key1,key2)
    if key in illum_mask_dict:
        illum = lookup_path(illum_mask_dict[key])
    else:
        illum = None
        log.warning("No illumination mask found for %s, no mask can "
                    "be added to the DQ plane" % adinput.filename)
        return None
        
    # Ensure that the illumination mask is an AstroData object
    illum_ad = None
    if illum is not None:
        log.fullinfo("Using %s as illumination mask" % str(illum))
    if isinstance(illum, AstroData):
        illum_ad = illum
    else:
        illum_ad = AstroData(illum)
        if illum_ad is None:
            log.warning("Cannot convert %s into an AstroData object, "
                        "no illumination mask will be added to the "
                        "DQ plane" % illum)                
            return None

    # Averaging and thresholding the science data to get a rough 
    # illumination mask
    addata = adinput['SCI',1].data
    adpixdata = np.copy(addata) 
    adpixdata /= np.average(adpixdata)
    (irange, jrange) = adpixdata.shape
    threshpixdata = np.empty(adpixdata.shape, np.int16)
    for i in [i+2 for i in range(irange-4)]:
        for j in [j+2 for j in range(jrange-4)]:
            if ((adpixdata[i,j] < 2) or (adpixdata[i+1,j] < 2) or 
                (adpixdata[i+2,j] < 2) or (adpixdata[i-1,j] < 2) or 
                (adpixdata[i-2,j] < 2) or (adpixdata[i,j+1] < 2) or 
                (adpixdata[i+1,j+1] < 2) or (adpixdata[i+2,j+1] < 2) or 
                (adpixdata[i-1,j+1] < 2) or (adpixdata[i-2,j+1] < 2) or 
                (adpixdata[i,j+2] < 2) or (adpixdata[i+1,j+2] < 2) or 
                (adpixdata[i+2,j+2] < 2) or (adpixdata[i-1,j+2] < 2) or 
                (adpixdata[i-2,j+2] < 2) or (adpixdata[i,j-1] < 2) or 
                (adpixdata[i+1,j-1] < 2) or (adpixdata[i+2,j-1] < 2) or 
                (adpixdata[i-1,j-1] < 2) or (adpixdata[i-2,j-1] < 2) or 
                (adpixdata[i,j-2] < 2) or (adpixdata[i+1,j-2] < 2) or 
                (adpixdata[i+2,j-2] < 2) or (adpixdata[i-1,j-2] < 2) or 
                (adpixdata[i-2,j-2] < 2)):
                threshpixdata[i,j] = 64
            else:
                threshpixdata[i,j] = 0

    # Finding the centre of mass of the rough pixel mask and using
    # this in comparison with the centre of mass of the illumination
    # mass to adjust the keyholes to align        
    x, y = scipy.ndimage.measurements.center_of_mass(threshpixdata)
    comx_illummask = illum_ad.phu_get_key_value('CENMASSX')
    comy_illummask = illum_ad.phu_get_key_value('CENMASSY')
    # 20. is a fudge factor found empirically
    dy = int((x - comx_illummask) * -20.)
    dx = int((y - comy_illummask) * -20.)

    # Recording the shifts in the header of the illumination mask
    log.stdinfo("Applying shifts to the illumination mask: dx = {}px, dy = "
                "{}px.".format(dx, dy))
    illum_ad.phu_set_key_value('OBSHIFTX', dx, "Relative x shift to object "
                               "frame")
    illum_ad.phu_set_key_value('OBSHIFTY', dy, "Relative y shift to object "
                               "frame")

    # Applying the offsets to the illumination mask
    illumpixdata1 = illum_ad['SCI',1].data
    illumpixdata2 = np.roll(illumpixdata1, dx, 1)
    illumpixdata3 = np.roll(illumpixdata2, dy, 0)
    illum_ad['SCI',1].data = illumpixdata3

    return illum_ad
