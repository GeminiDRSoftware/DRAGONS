from astrodata import Lookups
from astrodata import Descriptors

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardNICIKeyDict import stdkeyDictNICI
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class NICI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dict with the local dict of this descriptor class
    globalStdkeyDict.update(stdkeyDictNICI)
    
    def __init__(self):
        pass
    
    def camera(self, dataset, **args):
        """
        Return the camera value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        hdu = dataset.hdulist
        camera = hdu[0].header[stdkeyDictNICI['key_camera']]

        ret_camera = str(camera)
        
        return ret_camera
        
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        hdu = dataset.hdulist
        coadds = hdu[1].header[stdkeyDictNICI['key_coadds_r']]
        exptime = hdu[1].header[stdkeyDictNICI['key_exptime_r']]

        ret_exposure_time = float(exptime * coadds)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, **args):
        """
        Return the filter_name value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        hdu = dataset.hdulist
        filter1 = hdu[1].header[stdkeyDictNICI['key_filter_r']]
        filter2 = hdu[2].header[stdkeyDictNICI['key_filter_b']]
        filter1 = GemCalcUtil.removeComponentID(filter1)
        filter2 = GemCalcUtil.removeComponentID(filter2)

        ret_filter_name = str(filter1 + '|' + filter2)
        
        return ret_filter_name
    
    def focal_plane_mask(self, dataset, **args):
        """
        Return the focal_plane_mask value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        hdu = dataset.hdulist
        focal_plane_mask = \
            hdu[0].header[stdkeyDictNICI['key_focal_plane_mask']]

        ret_focal_plane_mask = str(focal_plane_mask)
                        
        return ret_focal_plane_mask
    
    def pixel_scale(self, dataset, **args):
        """
        Return the pixel_scale value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        ret_pixel_scale = float(0.018)
        
        return ret_pixel_scale
    
    def pupil_mask(self, dataset, **args):
        """
        Return the pupil_mask value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        hdu = dataset.hdulist
        pupil_mask = hdu[0].header[stdkeyDictNICI['key_pupil_mask']]

        ret_pupil_mask = str(pupil_mask)
        
        return ret_pupil_mask
