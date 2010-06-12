from astrodata import Lookups
from astrodata import Descriptors
import math

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardGNIRSKeyDict import stdkeyDictGNIRS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class GNIRS_DescriptorCalc(GEMINI_DescriptorCalc):
    
    gnirsArrayDict = None
    gnirsConfigDict = None
    
    def __init__(self):
        self.gnirsArrayDict = \
            Lookups.getLookupTable('Gemini/GNIRS/GNIRSArrayDict',
                                   'gnirsArrayDict')
        self.gnirsConfigDict = \
            Lookups.getLookupTable('Gemini/GNIRS/GNIRSConfigDict',
                                   'gnirsConfigDict')
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        hdu = dataset.hdulist
        central_wavelength = \
            hdu[0].header[stdkeyDictGNIRS['key_central_wavelength']]
        
        ret_central_wavelength = float(central_wavelength)
        
        return ret_central_wavelength
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the disperser value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to strip the component ID from the
        returned disperser name
        @param pretty: set to True to return a meaningful disperser name
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        # No specific pretty names, just use stripID
        if pretty:
            stripID=True
        
        hdu = dataset.hdulist
        disperser = hdu[0].header[stdkeyDictGNIRS['key_disperser']]
        
        if stripID:
            ret_disperser = str(GemCalcUtil.removeComponentID(disperser))
        else:
            ret_disperser = str(disperser)
        
        return ret_disperser
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        hdu = dataset.hdulist
        exposure_time = hdu[0].header[globalStdkeyDict['key_exposure_time']]
        coadds = dataset.coadds()
        
        if dataset.isType('GNIRS_RAW') == True and coadds != 1:
            ret_exposure_time = float(exposure_time * coadds)
        else:
            ret_exposure_time = float(exposure_time)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the filter_name value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID from the
        returned filter name
        @param pretty: set to True to return a meaningful filter name
        @rtype: string
        @return: the unique filter identifier string
        """
        # No specific pretty names, just use stripID
        if pretty:
            stripID=True
        
        hdu = dataset.hdulist
        filter1 = hdu[0].header[stdkeyDictGNIRS['key_filter1']]
        filter2 = hdu[0].header[stdkeyDictGNIRS['key_filter2']]
        
        if stripID:
            filter1 = GemCalcUtil.removeComponentID(filter1)
            filter2 = GemCalcUtil.removeComponentID(filter2)
        
        # Create list of filter values
        filters = [filter1,filter2]
        
        # reject 'Open'
        filters2 = []
        for filt in filters:
            if 'Open' in filt:
                pass
            else:
                filters2.append(filt)
        
        filters = filters2
        
        if 'DARK' in filters:
            ret_filter_name = 'blank'
        
        if len(filters) == 0:
            ret_filter_name = 'open'
        else:
            ret_filter_name = str('&'.join(filters))
        
        return ret_filter_name
    
    def focal_plane_mask(self, dataset, **args):
        """
        Return the focal_plane_mask value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        hdu = dataset.hdulist
        focal_plane_mask = \
            hdu[0].header[stdkeyDictGNIRS['key_focal_plane_mask']]
        
        ret_focal_plane_mask = str(focal_plane_mask)
        
        return ret_focal_plane_mask
    
    def gain(self, dataset, **args):
        """
        Return the gain value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the gain (electrons/ADU)
        """
        hdu = dataset.hdulist
        headerbias = hdu[0].header[stdkeyDictGNIRS['key_bias']]
        
        biasvalues = self.gnirsArrayDict.keys()
        
        for bias in biasvalues:
            if abs(float(bias) - abs(headerbias)) < 0.1:
                array = self.gnirsArrayDict[bias]
            else:
                array = None
        
        if array != None:
            ret_gain = float(array[2])
        else:
            ret_gain = None
        
        return ret_gain
    
    gnirsArrayDict = None
    
    def non_linear_level(self, dataset, **args):
        """
        Return the non_linear_level value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the non-linear level in the raw images (ADU)
        """
        # non_linear_level depends on whether data has been corrected for
        # non-linearity ... need to check this ...
        hdu = dataset.hdulist
        headerbias = hdu[0].header[stdkeyDictGNIRS['key_bias']]
        coadds = dataset.coadds()
        
        biasvalues = self.gnirsArrayDict.keys()
        for bias in biasvalues:
            if abs(float(bias) - abs(headerbias)) < 0.1:
                array = self.gnirsArrayDict[bias]
            else:
                array = None
        
        if array != None:
            well = float(array[3])
            linearlimit = float(array[4])
            nonlinearlimit = float(array[8])
            saturation = int(well * coadds)
            ret_non_linear_level = int(saturation * linearlimit)
        else:
            ret_non_linear_level = None
        
        return ret_non_linear_level
    
    gnirsArrayDict = None
    
    def observation_mode(self, dataset, **args):
        """
        Return the observation_mode value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the observing mode
        """
        hdu = dataset.hdulist
        prism = hdu[0].header[stdkeyDictGNIRS['key_prism']]
        decker = hdu[0].header[stdkeyDictGNIRS['key_decker']]
        disperser = hdu[0].header[stdkeyDictGNIRS['key_disperser']]
        camera = hdu[0].header[globalStdkeyDict['key_camera']]
        
        observation_mode_key = (prism, decker, disperser, camera)
        
        array = self.gnirsConfigDict[observation_mode_key]
        
        ret_observation_mode = str(array[3])
        
        return ret_observation_mode
    
    gnirsConfigDict = None
    
    def pixel_scale(self, dataset, **args):
        """
        Return the pixel_scale value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the pixel scale (arcsec/pixel)
        """
        hdu = dataset.hdulist
        prism = hdu[0].header[stdkeyDictGNIRS['key_prism']]
        decker = hdu[0].header[stdkeyDictGNIRS['key_decker']]
        disperser = hdu[0].header[stdkeyDictGNIRS['key_disperser']]
        camera = dataset.camera()
        
        pixel_scale_key = (prism, decker, disperser, camera)
        
        array = self.gnirsConfigDict[pixel_scale_key]
        
        ret_pixel_scale = float(array[2])
        
        return ret_pixel_scale
    
    gnirsConfigDict = None
    
    def read_noise(self, dataset, **args):
        """
        Return the read_noise value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the estimated readout noise (electrons)
        """
        hdu = dataset.hdulist
        headerbias = hdu[0].header[stdkeyDictGNIRS['key_bias']]
        lnrs = hdu[0].header[stdkeyDictGNIRS['key_lnrs']]
        ndavgs = hdu[0].header[stdkeyDictGNIRS['key_ndavgs']]
        coadds = dataset.coadds()
        
        biasvalues = self.gnirsArrayDict.keys()
        for bias in biasvalues:
            if abs(float(bias) - abs(headerbias)) < 0.1:
                array = self.gnirsArrayDict[bias]
            else:
                array = None
        
        if array != None:
            read_noise = float(array[1])
            ret_read_noise = float((read_noise * math.sqrt(coadds)) \
                / (math.sqrt(lnrs) * math.sqrt(ndavgs)))
        else:
            ret_read_noise = None
        
        return ret_read_noise
    
    gnirsArrayDict = None
    
    def saturation_level(self, dataset, **args):
        """
        Return the saturation_level value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the saturation level in the raw images (ADU)
        """
        hdu = dataset.hdulist
        headerbias = hdu[0].header[stdkeyDictGNIRS['key_bias']]
        coadds = dataset.coadds()
        
        biasvalues = self.gnirsArrayDict.keys()
        for bias in biasvalues:
            if abs(float(bias) - abs(headerbias)) < 0.1:
                array = self.gnirsArrayDict[bias]
            else:
                array = None
        
        if array != None:
            well = array[3]
            ret_saturation_level = int(well * coadds)
        else:
            ret_saturation_level = None
        
        return ret_saturation_level
    
    gnirsArrayDict = None
