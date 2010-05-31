from astrodata import Lookups
from astrodata import Descriptors

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardNICIKeyDict import stdkeyDictNICI
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class NICI_DescriptorCalc(GEMINI_DescriptorCalc):
    
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
        try:
            hdu = dataset.hdulist
            retcamerastring = hdu[0].header[stdkeyDictNICI['key_nici_camera']]
        
        except KeyError:
            return None
        
        return str(retcamerastring)
    
    def cwave(self, dataset, **args):
        """
        Return the cwave value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        retcwavefloat = None
        
        return retcwavefloat
    
    def datasec(self, dataset, **args):
        """
        Return the datasec value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the data section
        """
        retdatasecstring = None
        
        return str(retdatasecstring)
    
    def detsec(self, dataset, **args):
        """
        Return the detsec value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the detector section
        """
        retdetsecstring = None
        
        return str(retdetsecstring)
    
    def disperser(self, dataset, **args):
        """
        Return the disperser value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        retdisperserstring = None
        
        return str(retdisperserstring)
        
    def exptime(self, dataset, **args):
        """
        Return the exptime value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """

        try:
          hdu = dataset.hdulist
          coadds = hdu[0].header[stdkeyDictNICI['key_nici_coadds_r']]
          exptime = hdu[0].header[stdkeyDictNICI['key_nici_exptime_r']]

          retexptimefloat = float(exptime) * float(coadds)
        
          return retexptimefloat

        except KeyError:
          return None

    
    def filterid(self, dataset, **args):
        """
        Return the filterid value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter ID number string
        """
        retfilteridstring = None
        
        return str(retfilteridstring)
    
    def filtername(self, dataset, **args):
        """
        Return the filtername value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        try:
            hdu = dataset.hdulist
            filter1 = hdu[1].header[stdkeyDictNICI['key_nici_filter_r']]
            filter2 = hdu[2].header[stdkeyDictNICI['key_nici_filter_b']]
            filter1 = GemCalcUtil.removeComponentID(filter1)
            filter2 = GemCalcUtil.removeComponentID(filter2)

            retfilternamestring = filter1+'|'+filter2
        except KeyError: 
            return None
        except:
            return None
        
        return str(retfilternamestring)
    
    def fpmask(self, dataset, **args):
        """
        Return the fpmask value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retfpmaskstring = hdu[0].header[stdkeyDictNICI['key_nici_fpmask']]
       
        except KeyError:
            return None
                        
        return str(retfpmaskstring)
    
    def gain(self, dataset, **args):
        """
        Return the gain value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        #retgainfloat = None
        retgainfloat = -99.99
        
        return retgainfloat
    
    def mdfrow(self, dataset, **args):
        """
        Return the mdfrow value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the corresponding reference row in the MDF
        """
        retmdfrowint = None
        
        return retmdfrowint
    
    def nonlinear(self, dataset, **args):
        """
        Return the nonlinear value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        retnonlinearint = None
        
        return retnonlinearint

    def nsciext(self, dataset, **args):
        """
        Return the nsciext value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        retnsciextint = dataset.countExts('SCI')
        
        return int(retnsciextint)

    def obsmode(self, dataset, **args):
        """
        Return the obsmode value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        retobsmodestring = None
        
        return str(retobsmodestring)
    
    def pixscale(self, dataset, **args):
        """
        Return the pixscale value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        retpixscalefloat = 0.018
        
        return float(retpixscalefloat)
    
    def pupilmask(self, dataset, **args):
        """
        Return the pupilmask value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        try:
            hdu = dataset.hdulist
            retpupilmaskstring = hdu[0].header[stdkeyDictNICI['key_nici_pupilmask']]
        
        except KeyError:
            return None
        
        return str(retpupilmaskstring)
    
    def rdnoise(self, dataset, **args):
        """
        Return the rdnoise value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise (electrons)
        """
        retrdnoisefloat = None
        
        return retrdnoisefloat
    
    def satlevel(self, dataset, **args):
        """
        Return the satlevel value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        retsaturationint = None
        
        return retsaturationint
    
    def wdelta(self, dataset, **args):
        """
        Return the wdelta value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the dispersion (angstroms/pixel)
        """
        retwdeltafloat = None
        
        return retwdeltafloat
    
    def wrefpix(self, dataset, **args):
        """
        Return the wrefpix value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the reference pixel of the central wavelength
        """
        retwrefpixfloat = None
        
        return retwrefpixfloat
    
    def xccdbin(self, dataset, **args):
        """
        Return the xccdbin value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector x-axis
        """
        retxccdbinint = None
        
        return retxccdbinint
    
    def yccdbin(self, dataset, **args):
        """
        Return the yccdbin value for NICI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector y-axis
        """
        retyccdbinint = None
        
        return retyccdbinint

