from astrodata import Lookups
from astrodata import Descriptors
import math

import astrodata
from astrodata.Calculator import Calculator

import GemCalcUtil 
from StandardGNIRSKeyDict import stdkeyDictGNIRS

class GNIRS_RAWDescriptorCalc(Calculator):

    gnirsArrayDict = None
    gnirsConfigDict = None
    
    def __init__(self):
        self.gnirsArrayDict = Lookups.getLookupTable("Gemini/GNIRS/GNIRSArrayDict", "gnirsArrayDict")
        self.gnirsConfigDict = Lookups.getLookupTable("Gemini/GNIRS/GNIRSConfigDict", "gnirsConfigDict")
    
    def airmass(self, dataset, **args):
        """
        Return the airmass value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the mean airmass for the observation
        """
        try:
            hdu = dataset.hdulist
            retairmassfloat = hdu[0].header[stdkeyDictGNIRS["key_gnirs_airmass"]]
        
        except KeyError:
            return None
        
        return float(retairmassfloat)
    
    def camera(self, dataset, **args):
        """
        Return the camera value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retcamerastring = hdu[0].header[stdkeyDictGNIRS["key_gnirs_camera"]]
        
        except KeyError:
            return None
        
        return str(retcamerastring)
    
    def cwave(self, dataset, **args):
        """
        Return the cwave value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        try:
            hdu = dataset.hdulist
            retcwavefloat = hdu[0].header[stdkeyDictGNIRS["key_gnirs_cwave"]]
        
        except KeyError:
            return None
        
        return float(retcwavefloat)
    
    def datasec(self, dataset, **args):
        """
        Return the datasec value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the data section
        """
        retdatasecstring = None
        
        return str(retdatasecstring)
    
    def detsec(self, dataset, **args):
        """
        Return the detsec value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the detector section
        """
        retdetsecstring = None
        
        return str(retdetsecstring)
    
    def disperser(self, dataset, stripID=False, **args):
        """
        Return the disperser value for GNIRS
        @param dataset: the data set
        @param stripID: set to True to strip the component ID from the returned string
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retdisperserstring = hdu[0].header[stdkeyDictGNIRS["key_gnirs_disperser"]]

            if(stripID):
              retdisperserstring=GemCalcUtil.removeComponentID(retdisperserstring)
        
        except KeyError:
            return None
        
        return str(retdisperserstring)
    
    def exptime(self, dataset, **args):
        """
        Return the exptime value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exptime = hdu[0].header[stdkeyDictGNIRS["key_gnirs_exptime"]]
            coadds = hdu[0].header[stdkeyDictGNIRS["key_gnirs_coadds"]]
            
            if dataset.isType("GNIRS_RAW") == True:
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
        Return the filterid value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter ID number string
        """
        retfilteridstring = None
        
        return str(retfilteridstring)
    
    def filtername(self, dataset, **args):
        """
        Return the filtername value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        try:
            hdu = dataset.hdulist
            filter1 = hdu[0].header[stdkeyDictGNIRS["key_gnirs_filter1"]]
            filter2 = hdu[0].header[stdkeyDictGNIRS["key_gnirs_filter2"]]
            filter1 = GemCalcUtil.removeComponentID(filter1)
            filter2 = GemCalcUtil.removeComponentID(filter2)

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
            
            if "DARK" in filters:
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
        Return the fpmask value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retfpmaskstring = hdu[0].header[stdkeyDictGNIRS["key_gnirs_fpmask"]]
        
        except KeyError:
            return None
                        
        return str(retfpmaskstring)
    
    def gain(self, dataset, **args):
        """
        Return the gain value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictGNIRS["key_gnirs_bias"]]
            
            biasvalues = self.gnirsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.gnirsArrayDict[bias]
                else:
                    array = None

            if array != None:
                retgainfloat = array[2]
            else:
                return None
        
        except KeyError:
            return None
        
        return float(retgainfloat)

    gnirsArrayDict = None
    
    def instrument(self, dataset, **args):
        """
        Return the instrument value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the instrument used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retinstrumentstring = hdu[0].header[stdkeyDictGNIRS["key_gnirs_instrument"]]
        
        except KeyError:
            return None
                        
        return str(retinstrumentstring)
    
    def mdfrow(self, dataset, **args):
        """
        Return the mdfrow value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the corresponding reference row in the MDF
        """
        retmdfrowint = None
        
        return retmdfrowint
    
    def nonlinear(self, dataset, **args):
        """
        Return the nonlinear value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictGNIRS["key_gnirs_bias"]]
            coadds = hdu[0].header[stdkeyDictGNIRS["key_gnirs_coadds"]]

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
            else:
                return None
            
            saturation = int(well * coadds)
            retnonlinearint = int(saturation * linearlimit)
            #retnonlinearint = int(saturation * nonlinearlimit)
        
        except KeyError:
            return None
        
        return int(retnonlinearint)
    
    gnirsArrayDict = None
    
    def nsciext(self, dataset, **args):
        """
        Return the nsciext value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        retnsciextint = dataset.countExts(None)
        
        return int(retnsciextint)
    
    def object(self, dataset, **args):
        """
        Return the object value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the name of the object acquired
        """
        try:
            hdu = dataset.hdulist
            retobjectstring = hdu[0].header[stdkeyDictGNIRS["key_gnirs_object"]]
        
        except KeyError:
            return None
                        
        return str(retobjectstring)
    
    def obsmode(self, dataset, **args):
        """
        Return the obsmode value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        try:
            hdu = dataset.hdulist
            prism = hdu[0].header[stdkeyDictGNIRS["key_gnirs_prism"]]
            decker = hdu[0].header[stdkeyDictGNIRS["key_gnirs_decker"]]
            grating = hdu[0].header[stdkeyDictGNIRS["key_gnirs_grating"]]
            camera = hdu[0].header[stdkeyDictGNIRS["key_gnirs_camera"]]
            
            obsmodekey = (prism, decker, grating, camera)
            
            array = self.gnirsConfigDict[obsmodekey]
            
            retobsmodestring = array[3]
        
        except KeyError:
            return None
        
        return str(retobsmodestring)
    
    gnirsConfigDict = None

    def pixscale(self, dataset, **args):
        """
        Return the pixscale value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        try:
            hdu = dataset.hdulist
            prism = hdu[0].header[stdkeyDictGNIRS["key_gnirs_prism"]]
            decker = hdu[0].header[stdkeyDictGNIRS["key_gnirs_decker"]]
            grating = hdu[0].header[stdkeyDictGNIRS["key_gnirs_grating"]]
            camera = hdu[0].header[stdkeyDictGNIRS["key_gnirs_camera"]]

            pixscalekey = (prism, decker, grating, camera)

            array = self.gnirsConfigDict[pixscalekey]

            retpixscalefloat = array[2]
                            
        except KeyError:
            return None
        
        return float(retpixscalefloat)
    
    gnirsConfigDict = None
    
    def pupilmask(self, dataset, **args):
        """
        Return the pupilmask value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        retpupilmaskstring = None
        
        return str(retpupilmaskstring)
    
    def rdnoise(self, dataset, **args):
        """
        Return the rdnoise value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise (electrons)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictGNIRS["key_gnirs_bias"]]
            coadds = hdu[0].header[stdkeyDictGNIRS["key_gnirs_coadds"]]
            lnrs = hdu[0].header[stdkeyDictGNIRS["key_gnirs_lnrs"]]
            ndavgs = hdu[0].header[stdkeyDictGNIRS["key_gnirs_ndavgs"]]

            biasvalues = self.gnirsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.gnirsArrayDict[bias]
                else:
                    array = None

            if array != None:
                readnoise = float(array[1])
            else:
                return None

            retrdnoisefloat = (readnoise * math.sqrt(coadds)) / (math.sqrt(lnrs) * math.sqrt(ndavgs))

        except KeyError:
            return None
        
        return float(retrdnoisefloat)
    
    gnirsArrayDict = None
    
    def satlevel(self, dataset, **args):
        """
        Return the satlevel value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        try:
            hdu = dataset.hdulist
            headerbias = hdu[0].header[stdkeyDictGNIRS["key_gnirs_bias"]]
            coadds = hdu[0].header[stdkeyDictGNIRS["key_gnirs_coadds"]]

            biasvalues = self.gnirsArrayDict.keys()
            for bias in biasvalues:
                if abs(float(bias) - abs(headerbias)) < 0.1:
                    array = self.gnirsArrayDict[bias]
                else:
                    array = None

            if array != None:
                well = array[3]
            else:
                return None

            retsaturationint = int(well * coadds)
        
        except KeyError:
            return None
        
        return int(retsaturationint)
    
    gnirsArrayDict = None
    
    def utdate(self, dataset, **args):
        """
        Return the utdate value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT date of the observation (YYYY-MM-DD)
        """
        try:
            hdu = dataset.hdulist
            retutdatestring = hdu[0].header[stdkeyDictGNIRS["key_gnirs_utdate"]]
        
        except KeyError:
            return None
        
        return str(retutdatestring)
    
    def uttime(self, dataset, **args):
        """
        Return the uttime value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT at the start of the observation (HH:MM:SS.S)
        """
        try:
            hdu = dataset.hdulist
            retuttimestring = hdu[0].header[stdkeyDictGNIRS["key_gnirs_uttime"]]
        
        except KeyError:
            return None
        
        return str(retuttimestring)
    
    def wdelta(self, dataset, **args):
        """
        Return the wdelta value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the dispersion (angstroms/pixel)
        """
        retwdeltafloat = None
        
        return retwdeltafloat
    
    def wrefpix(self, dataset, **args):
        """
        Return the wrefpix value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the reference pixel of the central wavelength
        """
        retwrefpixfloat = None
        
        return retwrefpixfloat
    
    def xccdbin(self, dataset, **args):
        """
        Return the xccdbin value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector x-axis
        """
        retxccdbinint = None
        
        return retxccdbinint
    
    def yccdbin(self, dataset, **args):
        """
        Return the yccdbin value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector y-axis
        """
        retyccdbinint = None
        
        return retyccdbinint
