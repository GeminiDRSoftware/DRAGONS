from astrodata import Lookups
from astrodata import Descriptors
import re

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardTRECSKeyDict import stdkeyDictTRECS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class TRECS_DescriptorCalc(GEMINI_DescriptorCalc):
    
    def __init__(self):
        pass
    
    def camera(self, dataset, **args):
        """
        Return the camera value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        hdu = dataset.hdulist
        camera = hdu[0].header[stdkeyDictTRECS['key_camera']]
        
        ret_camera = str(camera)
        
        return ret_camera
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        hdu = dataset.hdulist
        disperser = hdu[0].header[stdkeyDictTRECS['key_disperser']]
        
        if disperser == 'LowRes-10':
            ret_central_wavelength = 10.5
        elif disperser == 'LowRes-20':
            ret_central_wavelength = 20.0
        else:
            ret_central_wavelength = None
        
        return ret_central_wavelength
    
    def disperser(self, dataset, **args):
        """
        Return the disperser value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        hdu = dataset.hdulist
        disperser = hdu[0].header[stdkeyDictTRECS['key_disperser']]
        
        ret_disperser = str(disperser)
        
        return ret_disperser
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        hdu = dataset.hdulist
        exposure_time = hdu[0].header[stdkeyDictTRECS['key_exposure_time']]
        
        ret_exposure_time = float(exposure_time)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, **args):
        """
        Return the filter_name value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        hdu = dataset.hdulist
        filter1 = hdu[0].header[stdkeyDictTRECS['key_filter1']]
        filter2 = hdu[0].header[stdkeyDictTRECS['key_filter2']]
        
        # create list of filter values
        filters = [filter1,filter2]
        
        # reject 'Open'
        filters2 = []
        for filt in filters:
            if ('Open' in filt):
                pass
            else:
                filters2.append(filt)
        
        filters = filters2
        
        # Block means an opaque mask was in place, which of course
        # blocks any other in place filters
        if 'Block' in filters:
            ret_filter_name = 'blank'
        
        if len(filters) == 0:
            ret_filter_name = 'open'
        else:
            filters.sort()
            ret_filter_name = str('&'.join(filters))
        
        return ret_filter_name
    
    def focal_plane_mask(self, dataset, **args):
        """
        Return the focal_plane_mask value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        hdu = dataset.hdulist
        focal_plane_mask = \
            hdu[0].header[stdkeyDictTRECS['key_focal_plane_mask']]
        
        ret_focal_plane_mask = str(focal_plane_mask)
        
        return ret_focal_plane_mask
    
    def gain(self, dataset, **args):
        """
        Return the gain value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        hdu = dataset.hdulist
        biaslevel = hdu[0].header[stdkeyDictTRECS['key_biaslevel']]
        
        if biaslevel == '2':
            ret_gain = 214.0
        elif biaslevel == '1':
            ret_gain = 718.0
        else:
            ret_gain = None
        
        return ret_gain
    
    def pixel_scale(self, dataset, **args):
        """
        Return the pixel_scale value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        ret_pixel_scale = 0.089
        
        return ret_pixel_scale
    
    def pupil_mask(self, dataset, **args):
        """
        Return the pupil_mask value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        hdu = dataset.hdulist
        pupil_mask = hdu[0].header[stdkeyDictTRECS['key_pupil_mask']]
        
        ret_pupil_mask = str(pupil_mask)
        
        return ret_pupil_mask
    
    def dispersion(self, dataset, **args):
        """
        Return the dispersion value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the dispersion (angstroms/pixel)
        """
        hdu = dataset.hdulist
        disperser = hdu[0].header[stdkeyDictTRECS['key_disperser']]
        
        if disperser == 'LowRes-10':
            ret_dispersion = 0.022
        elif disperser == 'LowRes-20':
            ret_dispersion = 0.033
        else:
            ret_dispersion = None
        
        return ret_dispersion
