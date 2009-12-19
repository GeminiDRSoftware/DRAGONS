from astrodata import Lookups
from astrodata import Descriptors

import math

from astrodata import Calculator
from Calculator import Calculator

import GemCalcUtil 
from StandardNICIKeyDict import stdkeyDictNICI

class NICI_RAWDescriptorCalc(Calculator):

    def airmass(self, dataset):
        """
        Return the airmass value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the mean airmass for the observation
        """
        try:
            hdu = dataset.hdulist
            retairmassfloat = hdu[0].header[stdkeyDictNICI["key_nici_airmass"]]
        
        except KeyError:
            return None
        
        return float(retairmassfloat)
    
    def camera(self, dataset):
        """
        Return the camera value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retcamerastring = hdu[0].header[stdkeyDictNICI["key_nici_camera"]]
        
        except KeyError:
            return None
        
        return str(retcamerastring)
    
    def cwave(self, dataset):
        """
        Return the cwave value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        retcwavefloat = None
        
        return retcwavefloat
    
    def datasec(self, dataset):
        """
        Return the datasec value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the data section
        """
        retdatasecstring = None
        
        return str(retdatasecstring)
    
    def detsec(self, dataset):
        """
        Return the detsec value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the detector section
        """
        retdetsecstring = None
        
        return str(retdetsecstring)
    
    def disperser(self, dataset):
        """
        Return the disperser value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        retdisperserstring = None
        
        return str(retdisperserstring)
        
    def exptime(self, dataset):
        """
        Return the exptime value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        retexptimefloat = None
        
        return retexptimefloat
    
    def filterid(self, dataset):
        """
        Return the filterid value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter ID number string
        """
        retfilteridstring = None
        
        return str(retfilteridstring)
    
    def filtername(self, dataset):
        """
        Return the filtername value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        retfilternamestring = None
        
        return str(retfilternamestring)
    
    def fpmask(self, dataset):
        """
        Return the fpmask value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retfpmaskstring = hdu[0].header[stdkeyDictNICI["key_nici_fpmask"]]
       
        except KeyError:
            return None
                        
        return str(retfpmaskstring)
    
    def gain(self, dataset):
        """
        Return the gain value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        retgainfloat = None
        
        return retgainfloat
    
    def instrument(self, dataset):
        """
        Return the instrument value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the instrument used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retinstrumentstring = hdu[0].header[stdkeyDictNICI["key_nici_instrument"]]
        
        except KeyError:
            return None
                        
        return str(retinstrumentstring)
    
    def mdfrow(self, dataset):
        """
        Return the mdfrow value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the corresponding reference row in the MDF
        """
        retmdfrowint = None
        
        return retmdfrowint
    
    def nonlinear(self, dataset):
        """
        Return the nonlinear value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        retnonlinearint = None
        
        return retnonlinearint

    def nsciext(self, dataset):
        """
        Return the nsciext value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        retnsciextint = dataset.countExts("SCI")
        
        return int(retnsciextint)

    def object(self, dataset):
        """
        Return the object value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the name of the object acquired
        """
        try:
            hdu = dataset.hdulist
            retobjectstring = hdu[0].header[stdkeyDictNICI["key_nici_object"]]
        
        except KeyError:
            return None
                        
        return str(retobjectstring)
    
    def obsmode(self, dataset):
        """
        Return the obsmode value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        retobsmodestring = None
        
        return str(retobsmodestring)
    
    def pixscale(self, dataset):
        """
        Return the pixscale value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        retpixscalefloat = 0.018
        
        return float(retpixscalefloat)
    
    def pupilmask(self, dataset):
        """
        Return the pupilmask value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        try:
            hdu = dataset.hdulist
            retpupilmaskstring = hdu[0].header[stdkeyDictNICI["key_nici_pupilmask"]]
        
        except KeyError:
            return None
        
        return str(retpupilmaskstring)
    
    def rdnoise(self, dataset):
        """
        Return the rdnoise value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise (electrons)
        """
        retrdnoisefloat = None
        
        return retrdnoisefloat
    
    def satlevel(self, dataset):
        """
        Return the satlevel value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        retsaturationint = None
        
        return retsaturationint
    
    def utdate(self, dataset):
        """
        Return the utdate value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT date of the observation (YYYY-MM-DD)
        """
        try:
            hdu = dataset.hdulist
            retutdatestring = hdu[0].header[stdkeyDictNICI["key_nici_utdate"]]
        
        except KeyError:
            return None
        
        return str(retutdatestring)
    
    def uttime(self, dataset):
        """
        Return the uttime value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT at the start of the observation (HH:MM:SS.S)
        """
        try:
            hdu = dataset.hdulist
            retuttimestring = hdu[0].header[stdkeyDictNICI["key_nici_uttime"]]
        
        except KeyError:
            return None
        
        return str(retuttimestring)
    
    def wdelta(self, dataset):
        """
        Return the wdelta value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the dispersion (angstroms/pixel)
        """
        retwdeltafloat = None
        
        return retwdeltafloat
    
    def wrefpix(self, dataset):
        """
        Return the wrefpix value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the reference pixel of the central wavelength
        """
        retwrefpixfloat = None
        
        return retwrefpixfloat
    
    def xccdbin(self, dataset):
        """
        Return the xccdbin value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector x-axis
        """
        retxccdbinint = None
        
        return retxccdbinint
    
    def yccdbin(self, dataset):
        """
        Return the yccdbin value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector y-axis
        """
        retyccdbinint = None
        
        return retyccdbinint
