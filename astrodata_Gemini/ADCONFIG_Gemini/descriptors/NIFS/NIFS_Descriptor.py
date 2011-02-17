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
        hdu = dataset.hdulist
        camera = hdu[0].header[stdkeyDictNIFS['key_camera']]
        
        ret_camera = str(camera)
        
        return ret_camera
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        hdu = dataset.hdulist
        central_wavelength = \
            hdu[0].header[stdkeyDictNIFS['key_central_wavelength']]
        
        ret_central_wavelength = float(central_wavelength)
        
        return ret_central_wavelength
    
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
        # No specific pretty names, just stripID
        if pretty:
            stripID=True
        
        hdu = dataset.hdulist
        disperser = hdu[0].header[stdkeyDictNIFS['key_disperser']]
        
        if stripID:
            ret_disperser = str(GemCalcUtil.removeComponentID(disperser))
        else:
            ret_disperser = str(disperser)
        
        return ret_disperser
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        hdu = dataset.hdulist
        exposure_time = hdu[0].header[globalStdkeyDict['key_exposure_time']]
        coadds = dataset.coadds()
        
        if dataset.isType('NIFS_RAW') == True and coadds != 1:
            ret_exposure_time = float(exposure_time * coadds)
        else:
            ret_exposure_time = float(exposure_time)
        
        return ret_exposure_time
    
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
            ret_filter_name = str(filter)
        
        return ret_filter_name
    
    def focal_plane_mask(self, dataset, **args):
        """
        Return the focal_plane_mask value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        hdu = dataset.hdulist
        focal_plane_mask = \
            hdu[0].header[stdkeyDictNIFS['key_focal_plane_mask']]
        
        ret_focal_plane_mask = str(focal_plane_mask)
        
        return ret_focal_plane_mask
    
    def gain(self, dataset, **args):
        """
        Return the gain value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        hdu = dataset.hdulist
        headerbias = hdu[0].header[stdkeyDictNIFS['key_bias']]
        
        biasvalues = self.nifsArrayDict.keys()
        for bias in biasvalues:
            if abs(float(bias) - abs(headerbias)) < 0.1:
                array = self.nifsArrayDict[bias]
            else:
                array = None
        
        if array != None:
            ret_gain = float(array[1])
        else:
            ret_gain = None
        
        return ret_gain
    
    nifsArrayDict = None
    
    def non_linear_level(self, dataset, **args):
        """
        Return the non_linear_level value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
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
            saturation = int(well * coadds)
            ret_non_linear_level = int(saturation * linearlimit)
        else:
            ret_non_linear_level = None
        
        return ret_non_linear_level
    
    nifsArrayDict = None
    
    def observation_epoch(self, dataset, **args):
        """
        Return the observation_epoch value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing epoch
        """
        hdu = dataset.hdulist
        observation_epoch = \
            hdu[0].header[stdkeyDictNIFS['key_observation_epoch']]
        
        ret_observation_epoch = str(observation_epoch)
        
        return ret_observation_epoch
    
    def pixel_scale(self, dataset, **args):
        """
        Return the pixel_scale value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        hdu = dataset.hdulist
        focal_plane_mask = \
            hdu[0].header[stdkeyDictNIFS['key_focal_plane_mask']]
        disperser = hdu[0].header[stdkeyDictNIFS['key_disperser']]
        filter = hdu[0].header[stdkeyDictNIFS['key_filter']]
        
        pixel_scale_key = (focal_plane_mask, disperser, filter)
        
        array = self.nifsConfigDict[pixel_scale_key]
        
        ret_pixel_scale = float(array[2])
        
        return ret_pixel_scale
    
    nifsConfigDict = None
    
    def read_mode(self, dataset, **args):
        """
        Return the read_mode value for NIFS
        Returns the following values as defined in the OT: 'Faint Object',
        'Medium Object', 'Bright Object'. Returns 'Invalid' if the headers
        don't make sense wrt these definitions
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the read mode
        """
        hdu = dataset.hdulist
        lnrs = hdu[0].header[stdkeyDictNIFS['key_lnrs']]
        headerbias = hdu[0].header[stdkeyDictNIFS['key_bias']]
        
        read_mode = 'Invalid'
        
        if lnrs == 1:
            read_mode = 'Bright Object'
        
        if lnrs == 4:
            read_mode = 'Medium Object'
        
        if lnrs == 16:
            read_mode = 'Faint Object'
        
        return read_mode
    
    def read_noise(self, dataset, **args):
        """
        Return the read_noise value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise (electrons)
        """
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
            ret_read_noise = float((read_noise * math.sqrt(coadds)) \
                / math.sqrt(lnrs))
        else:
            ret_read_noise = None
        
        return ret_read_noise
    
    nifsArrayDict = None
    
    def saturation_level(self, dataset, **args):
        """
        Return the saturation_level value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
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
            ret_saturation_level = None
        
        return ret_saturation_level
    
    nifsArrayDict = None
