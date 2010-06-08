from astrodata import Lookups
from astrodata import Descriptors
import re

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardMICHELLEKeyDict import stdkeyDictMICHELLE
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class MICHELLE_DescriptorCalc(GEMINI_DescriptorCalc):
    
    def __init__(self):
        pass
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        try:
            hdu = dataset.hdulist
            ret_central_wavelength = \
                hdu[0].header[stdkeyDictMICHELLE['key_central_wavelength']]
        
        except KeyError:
            return None
        
        return float(ret_central_wavelength)
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the disperser value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        # The Michelle components don't have component IDs so we just ignore the stripID and pretty options
        try:
            hdu = dataset.hdulist
            retdisperserstring = hdu[0].header[stdkeyDictMICHELLE['key_disperser']]
        
        except KeyError:
            return None
        
        return str(retdisperserstring)
    
    def exptime(self, dataset, **args):
        """
        Return the exptime value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exposure = float(hdu[0].header[stdkeyDictMICHELLE['key_exposure']])
            numexpos = float (hdu[0].header[stdkeyDictMICHELLE['key_numexpos']])
            numext = float(hdu[0].header[stdkeyDictMICHELLE['key_numext']])

            retexptimefloat = exposure * numexpos * numext
        
        except KeyError:
            return None
        
        return float(retexptimefloat)
    
    def filtername(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the filtername value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        # The Michelle filters don't have ID strings, so we just ignore the stripID and pretty options
        try:
            hdu = dataset.hdulist
            filter = hdu[0].header[stdkeyDictMICHELLE['key_filter']]
            
            if filter == 'NBlock':
                retfilternamestring = 'blank'
            else:
                retfilternamestring = filter
            
        except KeyError:
            return None
        
        return str(retfilternamestring)
    
    def fpmask(self, dataset, **args):
        """
        Return the fpmask value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retfpmaskstring = hdu[0].header[stdkeyDictMICHELLE['key_fpmask']]
        
        except KeyError:
            return None
                        
        return str(retfpmaskstring)
    
    def gain(self, dataset, **args):
        """
        Return the gain value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        try:
            hdu = dataset.hdulist
            retgainfloat = hdu[0].header[stdkeyDictMICHELLE['key_gain']]
        
        except KeyError:
            return None
        
        return float(retgainfloat)
    
    def nonlinear(self, dataset, **args):
        """
        Return the nonlinear value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        retnonlinearint = None
        
        return retnonlinearint
    
    def nsciext(self, dataset, **args):
        """
        Return the nsciext value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        try:
            hdu = dataset.hdulist
            retnsciextint = hdu[0].header[stdkeyDictMICHELLE['key_nsciext']]

        except KeyError:
            return None
        
        return int(retnsciextint)
    
    def obsmode(self, dataset, **args):
        """
        Return the obsmode value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        try:
            hdu = dataset.hdulist
            retobsmodestring = hdu[0].header[stdkeyDictMICHELLE['key_obsmode']]
        
        except KeyError:
            return None
        
        return str(retobsmodestring)
    
    def pixscale(self, dataset, **args):
        """
        Return the pixscale value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        try:
            hdu = dataset.hdulist
            retpixscalefloat = hdu[0].header[stdkeyDictMICHELLE['key_pixscale']]
        
        except KeyError:
            return None
        
        return float(retpixscalefloat)
    
    def pupilmask(self, dataset, **args):
        """
        Return the pupilmask value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        retpupilmaskstring = None
        
        return str(retpupilmaskstring)
    
    def rdnoise(self, dataset, **args):
        """
        Return the rdnoise value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise
        """
        retrdnoisefloat = None
        
        return retrdnoisefloat
    
    def satlevel(self, dataset, **args):
        """
        Return the satlevel value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        retsaturationint = None
        
        return retsaturationint
    
    def utdate(self, dataset, **args):
        """
        Return the utdate value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT date of the observation (YYYY-MM-DD)
        """
        try:
            hdu = dataset.hdulist
            retutdatestring = hdu[0].header[stdkeyDictMICHELLE['key_utdate']]
        
        except KeyError:
            return None
        
        return str(retutdatestring)
    
    def wdelta(self, dataset, **args):
        """
        Return the wdelta value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the dispersion (angstroms/pixel)
        """
        try:
            hdu = dataset.hdulist
            retwdeltafloat = hdu[0].header[stdkeyDictMICHELLE['key_wdelta']]
        
        except KeyError:
            return None
        
        return float(retwdeltafloat)
    
    def wrefpix(self, dataset, **args):
        """
        Return the wrefpix value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the reference pixel of the central wavelength
        """
        retwrefpixfloat = None
        
        return retwrefpixfloat
