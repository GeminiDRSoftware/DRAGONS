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
    
