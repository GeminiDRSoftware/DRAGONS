from astrodata import Lookups
from astrodata import Descriptors
import math

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardNIFSKeyDict import stdkeyDictNIFS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class NIFS_DescriptorCalc(GEMINI_DescriptorCalc):

    nifsArrayDict = None
    nifsConfigDict = None    
    
    def __init__(self):
        self.nifsArrayDict = \
            Lookups.getLookupTable('Gemini/NIFS/NIFSArrayDict',
                                   'nifsArrayDict')
        self.nifsConfigDict = \
            Lookups.getLookupTable('Gemini/NIFS/NIFSConfigDict',
                                   'nifsConfigDict')
    
    def camera(self, dataset, **args):
        """
        Return the camera value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            ret_camera = hdu[0].header[stdkeyDictNIFS['key_camera']]
        
        except KeyError:
            return None
        
        return str(ret_camera)
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        try:
            hdu = dataset.hdulist
            ret_central_wavelength = \
                hdu[0].header[stdkeyDictNIFS['key_central_wavelength']]
        
        except KeyError:
            return None
        
        return float(ret_central_wavelength)
    
    def disperser(self, dataset, stripID = False, pretty=False, **args):
        """
        Return the disperser value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID from the
        returned disperser name
        @param pretty: set to True to return a meaningful disperser name
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        try:
            # No specific pretty names, just stripID
            if pretty:
                stripID=True

            hdu = dataset.hdulist
            disperser = hdu[0].header[stdkeyDictNIFS['key_disperser']]
            
            if stripID:
                ret_disperser = GemCalcUtil.removeComponentID(disperser)
            else:
                ret_disperser = disperser
        
        except KeyError:
            return None
        
        return str(ret_disperser)
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exposure_time = \
                hdu[0].header[globalStdkeyDict['key_exposure_time']]
            coadds = dataset.coadds()
            
            if dataset.isType('NIFS_RAW') == True and coadds != 1:
                ret_exposure_time = exposure_time * coadds
            else:
                ret_exposure_time = exposure_time
        
        except KeyError:
            return None
        
        return float(ret_exposure_time)
    
    def filter_name(self, dataset, pretty=False, stripID=False, **args):
        """
        Return the filter_name value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID from the
        returned filter name
        @param pretty: set to True to return a meaningful filter name
        @rtype: string
        @return: the unique filter identifier string
        """
        try:
            # No specific pretty names, just use stripID
            if pretty:
                stripID=True

            hdu = dataset.hdulist
            filter = hdu[0].header[stdkeyDictNIFS['key_filter']]
            if stripID:
                filter = GemCalcUtil.removeComponentID(filter)

            if filter == 'Blocked':
                ret_filter_name = 'blank'
            else:
                ret_filter_name = filter
        
        except KeyError:
            return None
        
        return str(ret_filter_name)
    
    def focal_plane_mask(self, dataset, **args):
        """
        Return the focal_plane_mask value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            ret_focal_plane_mask = \
                hdu[0].header[stdkeyDictNIFS['key_focal_plane_mask']]
        
        except KeyError:
            return None
                        
        return str(ret_focal_plane_mask)
    
    def gain(self, dataset, **args):
        """
        Return the gain value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictNIFS['key_bias']]
            
            biasvalues = self.nifsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.nifsArrayDict[bias]
                else:
                    array = None
            
            if array != None:
                ret_gain = array[1]
            else:
                return None
        
        except KeyError:
            return None
        
        return float(ret_gain)
    
    nifsArrayDict = None

    def non_linear_level(self, dataset, **args):
        """
        Return the non_linear_level value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        try:
            # non_linear_level depends on whether data has been corrected for
            # non-linearity ... need to check this ...
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictNIFS['key_bias']]
            coadds = dataset.coadds()

            biasvalues = self.nifsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.nifsArrayDict[bias]
                else:
                    array = None

            if array != None:
                well = float(array[2])
                linearlimit = float(array[3])
                nonlinearlimit = float(array[7])
            else:
                return None

            saturation = int(well * coadds)
            ret_non_linear_level = int(saturation * linearlimit)
            #ret_non_linear_level = int(saturation * nonlinearlimit)
        
        except KeyError:
            return None
                
        return int(ret_non_linear_level)
    
    nifsArrayDict = None
    
    def observation_mode(self, dataset, **args):
        """
        Return the observation_mode value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        try:
            hdu = dataset.hdulist
            focal_plane_mask = \
                hdu[0].header[stdkeyDictNIFS['key_focal_plane_mask']]
            disperser = hdu[0].header[stdkeyDictNIFS['key_disperser']]
            filter = hdu[0].header[stdkeyDictNIFS['key_filter']]

            observation_mode_key = (focal_plane_mask, disperser, filter)
            
            array = self.nifsConfigDict[observation_mode_key]

            ret_observation_mode = array[3]
        
        except KeyError:
            return None
        
        return str(ret_observation_mode)
    
    nifsConfigDict = None
    
    def pixel_scale(self, dataset, **args):
        """
        Return the pixel_scale value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        try:
            hdu = dataset.hdulist
            focal_plane_mask = \
                hdu[0].header[stdkeyDictNIFS['key_focal_plane_mask']]
            disperser = hdu[0].header[stdkeyDictNIFS['key_disperser']]
            filter = hdu[0].header[stdkeyDictNIFS['key_filter']]

            pixel_scale_key = (focal_plane_mask, disperser, filter)

            array = self.nifsConfigDict[pixel_scale_key]

            ret_pixel_scale = array[2]
        
        except KeyError:
            return None
        
        return float(ret_pixel_scale)
    
    nifsConfigDict = None
    
    def read_noise(self, dataset, **args):
        """
        Return the read_noise value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise (electrons)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictNIFS['key_bias']]
            lnrs = hdu[0].header[stdkeyDictNIFS['key_lnrs']]
            coadds = dataset.coadds()

            biasvalues = self.nifsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.nifsArrayDict[bias]
                else:
                    array = None

            if array != None:
                read_noise = float(array[0])
            else:
                return None
        
            ret_read_noise = (read_noise * math.sqrt(coadds)) / math.sqrt(lnrs)
        
        except KeyError:
            return None

        return float(ret_read_noise)
    
    nifsArrayDict = None
    
    def saturation_level(self, dataset, **args):
        """
        Return the saturation_level value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictNIFS['key_bias']]
            coadds = dataset.coadds()

            biasvalues = self.nifsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.nifsArrayDict[bias]
                else:
                    array = None
            
            if array != None:
                well = float(array[2])
                ret_saturation_level = int(well * coadds)
            else:
                return None
        
        except KeyError:
            return None
        
        return int(ret_saturation_level)
    
    nifsArrayDict = None
