from astrodata import Lookups
from astrodata import Descriptors
import math

import astrodata
from astrodata.Calculator import Calculator

import GemCalcUtil 
from StandardNIFSKeyDict import stdkeyDictNIFS

class NIFS_RAWDescriptorCalc(Calculator):

    nifsArrayDict = None
    nifsConfigDict = None    
    
    def __init__(self):
        self.nifsArrayDict = Lookups.getLookupTable("Gemini/NIFS/NIFSArrayDict", "nifsArrayDict")
        self.nifsConfigDict = Lookups.getLookupTable("Gemini/NIFS/NIFSConfigDict", "nifsConfigDict")
    
    def airmass(self, dataset, **args):
        """
        Return the airmass value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the mean airmass for the observation
        """
        try:
            hdu = dataset.hdulist
            retairmassfloat = hdu[0].header[stdkeyDictNIFS["key_nifs_airmass"]]
        
        except KeyError:
            return None
        
        return float(retairmassfloat)
    
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
            retcamerastring = hdu[0].header[stdkeyDictNIFS["key_nifs_camera"]]
        
        except KeyError:
            return None
        
        return str(retcamerastring)
    
    def cwave(self, dataset, **args):
        """
        Return the cwave value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        try:
            hdu = dataset.hdulist
            retcwavefloat = hdu[0].header[stdkeyDictNIFS["key_nifs_cwave"]]
        
        except KeyError:
            return None
        
        return float(retcwavefloat)
    
    def datasec(self, dataset, **args):
        """
        Return the datasec value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the data section
        """
        try:
            for ext in dataset:
                # get the value - NIFS raw data will only have one data extension
                retdatasecstring = ext.header[stdkeyDictNIFS["key_nifs_datasec"]]
        
        except KeyError:
            return None
        
        return str(retdatasecstring)
    
    def detsec(self, dataset, **args):
        """
        Return the detsec value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the detector section
        """
        try:
            for ext in dataset:
                # get the value - NIFS raw data will only have one data extension
                retdetsecstring = ext.header[stdkeyDictNIFS["key_nifs_detsec"]]
        
        except KeyError:
            return None
        
        return str(retdetsecstring)
    
    def disperser(self, dataset, **args):
        """
        Return the disperser value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retdisperserstring = hdu[0].header[stdkeyDictNIFS["key_nifs_disperser"]]
        
        except KeyError:
            return None
        
        return str(retdisperserstring)
    
    def exptime(self, dataset, **args):
        """
        Return the exptime value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exptime = hdu[0].header[stdkeyDictNIFS["key_nifs_exptime"]]
            coadds = hdu[0].header[stdkeyDictNIFS["key_nifs_coadds"]]
            
            if dataset.isType("NIFS_RAW") == True:
                if coadds != 1:
                    coaddexp = exptime
                    retexptimefloat = exptime * coadds
                else:
                    retexptimefloat = exptime
            else:
                return exptime
        
        except KeyError:
            return None
        
        return float(retexptimefloat)
    
    def filterid(self, dataset, **args):
        """
        Return the filterid value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter ID number string
        """
        retfilteridstring = None
        
        return str(retfilteridstring)
    
    def filtername(self, dataset, **args):
        """
        Return the filtername value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        try:
            hdu = dataset.hdulist
            filter = hdu[0].header[stdkeyDictNIFS["key_nifs_filter"]]
            filter = GemCalcUtil.removeComponentID(filter)

            if filter == "Blocked":
                retfilternamestring = "blank"
            else:
                retfilternamestring = filter
        
        except KeyError:
            return None
        
        return str(retfilternamestring)
    
    def fpmask(self, dataset, **args):
        """
        Return the fpmask value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retfpmaskstring = hdu[0].header[stdkeyDictNIFS["key_nifs_fpmask"]]
        
        except KeyError:
            return None
                        
        return str(retfpmaskstring)
    
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
            headerbias = hdu[0].header[stdkeyDictNIFS["key_nifs_bias"]]
            
            biasvalues = self.nifsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.nifsArrayDict[bias]
                else:
                    array = None
            
            if array != None:
                retgainfloat = array[1]
            else:
                return None
        
        except KeyError:
            return None
        
        return float(retgainfloat)
    
    nifsArrayDict = None

    def instrument(self, dataset, **args):
        """
        Return the instrument value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the instrument used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retinstrumentstring = hdu[0].header[stdkeyDictNIFS["key_nifs_instrument"]]
        
        except KeyError:
            return None
                        
        return str(retinstrumentstring)
    
    def mdfrow(self, dataset, **args):
        """
        Return the mdfrow value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the corresponding reference row in the MDF
        """
        retmdfrowint = None
        
        return retmdfrowint
    
    def nonlinear(self, dataset, **args):
        """
        Return the nonlinear value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictNIFS["key_nifs_bias"]]
            coadds = hdu[0].header[stdkeyDictNIFS["key_nifs_coadds"]]

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
            retnonlinearint = int(saturation * linearlimit)
            #retnonlinearint = int(saturation * nonlinearlimit)
        
        except KeyError:
            return None
                
        return int(retnonlinearint)
    
    nifsArrayDict = None
    
    def nsciext(self, dataset, **args):
        """
        Return the nsciext value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        retnsciextint = dataset.countExts(None)
        
        return int(retnsciextint)
    
    def object(self, dataset, **args):
        """
        Return the object value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the name of the object acquired
        """
        try:
            hdu = dataset.hdulist
            retobjectstring = hdu[0].header[stdkeyDictNIFS["key_nifs_object"]]
        
        except KeyError:
            return None
                        
        return str(retobjectstring)
    
    def obsmode(self, dataset, **args):
        """
        Return the obsmode value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        try:
            hdu = dataset.hdulist
            fpmask = hdu[0].header[stdkeyDictNIFS["key_nifs_fpmask"]]
            grating = hdu[0].header[stdkeyDictNIFS["key_nifs_grating"]]
            filter = hdu[0].header[stdkeyDictNIFS["key_nifs_filter"]]

            obsmodekey = (fpmask, grating, filter)
            
            array = self.nifsConfigDict[obsmodekey]

            retobsmodestring = array[3]
        
        except KeyError:
            return None
        
        return str(retobsmodestring)
    
    nifsConfigDict = None
    
    def pixscale(self, dataset, **args):
        """
        Return the pixscale value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        try:
            hdu = dataset.hdulist
            fpmask = hdu[0].header[stdkeyDictNIFS["key_nifs_fpmask"]]
            grating = hdu[0].header[stdkeyDictNIFS["key_nifs_grating"]]
            filter = hdu[0].header[stdkeyDictNIFS["key_nifs_filter"]]

            pixscalekey = (fpmask, grating, filter)

            array = self.nifsConfigDict[pixscalekey]

            retpixscalefloat = array[2]
        
        except KeyError:
            return None
        
        return float(retpixscalefloat)
    
    nifsConfigDict = None
    
    def pupilmask(self, dataset, **args):
        """
        Return the pupilmask value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        retpupilmaskstring = None
        
        return str(retpupilmaskstring)
    
    def rdnoise(self, dataset, **args):
        """
        Return the rdnoise value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise (electrons)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictNIFS["key_nifs_bias"]]
            coadds = hdu[0].header[stdkeyDictNIFS["key_nifs_coadds"]]
            lnrs = hdu[0].header[stdkeyDictNIFS["key_nifs_lnrs"]]

            biasvalues = self.nifsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.nifsArrayDict[bias]
                else:
                    array = None

            if array != None:
                readnoise = float(array[0])
            else:
                return None
        
            retrdnoisefloat = (readnoise * math.sqrt(coadds)) / math.sqrt(lnrs)
        
        except KeyError:
            return None

        return float(retrdnoisefloat)
    
    nifsArrayDict = None
    
    def satlevel(self, dataset, **args):
        """
        Return the satlevel value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictNIFS["key_nifs_bias"]]
            coadds = hdu[0].header[stdkeyDictNIFS["key_nifs_coadds"]]

            biasvalues = self.nifsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.nifsArrayDict[bias]
                else:
                    array = None
            
            if array != None:
                well = float(array[2])
                retsaturationint = int(well * coadds)
            else:
                return None
        
        except KeyError:
            return None
        
        return int(retsaturationint)
    
    nifsArrayDict = None
    
    def utdate(self, dataset, **args):
        """
        Return the utdate value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT date of the observation (YYYY-MM-DD)
        """
        try:
            hdu = dataset.hdulist
            retutdatestring = hdu[0].header[stdkeyDictNIFS["key_nifs_utdate"]]
        
        except KeyError:
            return None
        
        return str(retutdatestring)
    
    def uttime(self, dataset, **args):
        """
        Return the uttime value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT at the start of the observation (HH:MM:SS.S)
        """
        try:
            hdu = dataset.hdulist
            retuttimestring = hdu[0].header[stdkeyDictNIFS["key_nifs_uttime"]]
        
        except KeyError:
            return None
        
        return str(retuttimestring)
    
    def wdelta(self, dataset, **args):
        """
        Return the wdelta value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the dispersion (angstroms/pixel)
        """
        retwdeltafloat = None
        
        return retwdeltafloat
    
    def wrefpix(self, dataset, **args):
        """
        Return the wrefpix value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the reference pixel of the central wavelength
        """
        retwrefpixfloat = None
        
        return retwrefpixfloat
    
    def xccdbin(self, dataset, **args):
        """
        Return the xccdbin value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector x-axis
        """
        retxccdbinint = None
        
        return retxccdbinint
    
    def yccdbin(self, dataset, **args):
        """
        Return the yccdbin value for NIFS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector y-axis
        """
        retyccdbinint = None
        
        return retyccdbinint
