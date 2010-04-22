from astrodata import Lookups
from astrodata import Descriptors
import re

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardTRECSKeyDict import stdkeyDictTRECS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class TRECS_DescriptorCalc(GEMINI_DescriptorCalc):

    def camera(self, dataset, **args):
        """
        Return the camera value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retcamerastring = hdu[0].header[stdkeyDictTRECS["key_trecs_camera"]]
        
        except KeyError:
            return None
        
        return str(retcamerastring)
    
    def cwave(self, dataset, **args):
        """
        Return the cwave value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        try:
            hdu = dataset.hdulist
            disperser = hdu[0].header[stdkeyDictTRECS["key_trecs_disperser"]]

            if disperser == "LowRes-10":
                retcwavefloat = 10.5
            elif disperser == "LowRes-20":
                retcwavefloat = 20.0
            else:
                return None
        
        except KeyError:
            return None
        
        return float(retcwavefloat)
    
    def datasec(self, dataset, **args):
        """
        Return the datasec value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the data section
        """
        retdatasecstring = None
        
        return str(retdatasecstring)
    
    def detsec(self, dataset, **args):
        """
        Return the detsec value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the detector section
        """
        retdetsecstring = None
        
        return str(retdetsecstring)
    
    def disperser(self, dataset, **args):
        """
        Return the disperser value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retdisperserstring = hdu[0].header[stdkeyDictTRECS["key_trecs_disperser"]]
        
        except KeyError:
            return None
        
        return str(retdisperserstring)
    
    def exptime(self, dataset, **args):
        """
        Return the exptime value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            retexptimefloat = float(hdu[0].header[stdkeyDictTRECS["key_trecs_exptime"]])
                    
        except KeyError:
            return None
        
        return float(retexptimefloat)
    
    def filterid(self, dataset, **args):
        """
        Return the filterid value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter ID number string
        """
        retfilteridstring = None
        
        return str(retfilteridstring)
    
    def filtername(self, dataset, **args):
        """
        Return the filtername value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        try:
            hdu = dataset.hdulist
            filter1 = hdu[0].header[stdkeyDictTRECS["key_trecs_filter1"]]
            filter2 = hdu[0].header[stdkeyDictTRECS["key_trecs_filter2"]]

            # create list of filter values
            filters = [filter1,filter2]

            # reject "Open"
            filters2 = []
            for filt in filters:
                if ("Open" in filt):
                    pass
                else:
                    filters2.append(filt)
            
            filters = filters2
            
            # Block means an opaque mask was in place, which of course
            # blocks any other in place filters
            if "Block" in filters:
                retfilternamestring = "blank"
            
            if len(filters) == 0:
                retfilternamestring = "open"
            else:
                filters.sort()
                retfilternamestring = "&".join(filters)
            
        except KeyError:
            return None
        
        return str(retfilternamestring)
    
    def fpmask(self, dataset, **args):
        """
        Return the fpmask value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retfpmaskstring = hdu[0].header[stdkeyDictTRECS["key_trecs_fpmask"]]
        
        except KeyError:
            return None
                        
        return str(retfpmaskstring)
    
    def gain(self, dataset, **args):
        """
        Return the gain value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        try:
            hdu = dataset.hdulist
            biaslevel = hdu[0].header[stdkeyDictTRECS["key_trecs_biaslevel"]]

            if biaslevel == "2":
                retgainfloat = 214.0
            elif biaslevel == "1":
                retgainfloat = 718.0
            else:
                return None
        
        except KeyError:
            return None

        return float(retgainfloat)
    
    def mdfrow(self, dataset, **args):
        """
        Return the mdfrow value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the corresponding reference row in the MDF
        """
        retmdfrowint = None
        
        return retmdfrowint
    
    def nonlinear(self, dataset, **args):
        """
        Return the nonlinear value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        retnonlinearint = None
        
        return retnonlinearint
    
    def nsciext(self, dataset, **args):
        """
        Return the nsciext value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        retnsciextint = dataset.countExts(None)
        
        return int(retnsciextint)
    
    def obsmode(self, dataset, **args):
        """
        Return the obsmode value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        try:
            hdu = dataset.hdulist
            retobsmodestring = hdu[0].header[stdkeyDictTRECS["key_trecs_obsmode"]]
        
        except KeyError:
            return None
        
        return str(retobsmodestring)
    
    def pixscale(self, dataset, **args):
        """
        Return the pixscale value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        retpixscalefloat = 0.089

        return float(retpixscalefloat)
    
    def pupilmask(self, dataset, **args):
        """
        Return the pupilmask value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        try:
            hdu = dataset.hdulist
            retpupilmaskstring = hdu[0].header[stdkeyDictTRECS["key_trecs_pupilmask"]]
        
        except KeyError:
            return None
        
        return str(retpupilmaskstring)
    
    def rdnoise(self, dataset, **args):
        """
        Return the rdnoise value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise
        """
        retrdnoisefloat = None
        
        return retrdnoisefloat
    
    def satlevel(self, dataset, **args):
        """
        Return the satlevel value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        retsaturationint = None
        
        return retsaturationint
    
    def wdelta(self, dataset, **args):
        """
        Return the wdelta value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the dispersion (angstroms/pixel)
        """
        try:
            hdu = dataset.hdulist
            disperser = hdu[0].header[stdkeyDictTRECS["key_trecs_disperser"]]

            if disperser == "LowRes-10":
                retwdeltafloat = 0.022
            elif disperser == "LowRes-20":
                retwdeltafloat = 0.033
            else:
                return None
        
        except KeyError:
            return None
        
        return float(retwdeltafloat)
    
    def wrefpix(self, dataset, **args):
        """
        Return the wrefpix value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the reference pixel of the central wavelength
        """
        retwrefpixfloat = None
        
        return retwrefpixfloat
    
    def xccdbin(self, dataset, **args):
        """
        Return the xccdbin value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector x-axis
        """
        retxccdbinint = None
        
        return retxccdbinint
    
    def yccdbin(self, dataset, **args):
        """
        Return the yccdbin value for TRECS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector y-axis
        """
        retyccdbinint = None
        
        return retyccdbinint
