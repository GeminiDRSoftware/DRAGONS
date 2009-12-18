import Lookups
import Descriptors
import re

from Calculator import Calculator
from datetime import datetime
from time import strptime

import GemCalcUtil
from StandardGMOSKeyDict import stdkeyDictGMOS

class GMOS_RAWDescriptorCalc(Calculator):

    gmosampsGain = None
    gmosampsGainBefore20060831 = None
    gmosampsRdnoise = None
    gmosampsRdnoiseBefore20060831 = None
    
    def __init__(self):
        # self.gmosampsGain = Lookups.getLookupTable("Gemini/GMOS/GMOSAmpTables", "gmosampsGain")
        # self.gmosampsGainBefore20060831 = Lookups.getLookupTable("Gemini/GMOS/GMOSAmpTables", "gmosampsGainBefore20060831")
        
        # slightly more efficiently, we can get both at once since they are in
        # the same lookup space
        self.gmosampsGain,self.gmosampsGainBefore20060831 = Lookups.getLookupTable("Gemini/GMOS/GMOSAmpTables", "gmosampsGain", "gmosampsGainBefore20060831")
        self.gmosampsRdnoise,self.gmosampsRdnoiseBefore20060831 = Lookups.getLookupTable("Gemini/GMOS/GMOSAmpTables", "gmosampsRdnoise", "gmosampsRdnoiseBefore20060831")
        
    def airmass(self, dataset):
        """
        Return the airmass value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the mean airmass for the observation
        """
        try:
            hdu = dataset.hdulist
            retairmassfloat = hdu[0].header[stdkeyDictGMOS["key_gmos_airmass"]]
        
        except KeyError:
            return None
        
        return float(retairmassfloat)
    
    def camera(self, dataset):
        """
        Return the camera value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retcamerastring = hdu[0].header[stdkeyDictGMOS["key_gmos_camera"]]
        
        except KeyError:
            return None
        
        return str(retcamerastring)
    
    def cwave(self, dataset):
        """
        Return the cwave value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        try:
            hdu = dataset.hdulist
            retcwavefloat = hdu[0].header[stdkeyDictGMOS["key_gmos_cwave"]]
        
        except KeyError:
            return None

        return float(retcwavefloat)
    
    def datasec(self, dataset):
        """
        Return the datasec value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: list
        @returns: an array of data section values, where the number of
        array elements equals the number of extensions in the image.
        Index 0 is the data section of the first extension containing pixels
        (equivalent to EXTNAME = "SCI") and so on.
        """
        retdataseclist = []
        for ext in dataset:
            try:
                # get the values - GMOS raw data can have up to 6 data extensions
                datasec = ext.header[stdkeyDictGMOS["key_gmos_datasec"]]
            
            except KeyError:
                datasec = None
            
            retdataseclist.append(datasec)
        
        return retdataseclist
    
    def detsec(self, dataset):
        """
        Return the detsec value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: list
        @returns: an array of detector section values, where the number of
        array elements equals the number of extensions in the image.
        Index 0 is the data section of the first extension containing pixels
        (equivalent to EXTNAME = "SCI") and so on.
        """
        retdetseclist = []
        for ext in dataset:
            try:
                # get the values - GMOS raw data can have up to 6 data extensions
                detsec = ext.header[stdkeyDictGMOS["key_gmos_detsec"]]

            except KeyError:
                detsec = None
            
            retdetseclist.append(detsec)
        
        return retdetseclist
    
    def disperser(self, dataset):
        """
        Return the disperser value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retdisperserstring = hdu[0].header[stdkeyDictGMOS["key_gmos_disperser"]]
        
        except KeyError:
            return None
        
        return str(retdisperserstring)
    
    def exptime(self, dataset):
        """
        Return the exptime value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            retexptimefloat = hdu[0].header[stdkeyDictGMOS["key_gmos_exptime"]]
        
        except KeyError:
            return None
        
        return float(retexptimefloat)
    
    def filterid(self, dataset):
        """
        Return the filterid value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter ID number string
        """
        try:
            hdu = dataset.hdulist
            filtid1 = str(hdu[0].header[stdkeyDictGMOS["key_gmos_filtid1"]])
            filtid2 = str(hdu[0].header[stdkeyDictGMOS["key_gmos_filtid2"]])
            
            filtsid = []
            filtsid.append(filtid1)
            filtsid.append(filtid2)
            filtsid.sort()
            retfilteridstring = "&".join(filtsid)
        
        except KeyError:
            return None
        
        return str(retfilteridstring)
    
    def filtername(self, dataset):
        """
        Return the filtername value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        try:
            hdu = dataset.hdulist
            filter1 = hdu[0].header[stdkeyDictGMOS["key_gmos_filter1"]]
            filter2 = hdu[0].header[stdkeyDictGMOS["key_gmos_filter2"]]
            filter1 = GemCalcUtil.removeComponentID(filter1)
            filter2 = GemCalcUtil.removeComponentID(filter2)
            
            filters = []
            if not "open" in filter1:
                filters.append(filter1)
            if not "open" in filter2:
                filters.append(filter2)
            
            if len(filters) == 0:
                retfilternamestring = "open"
            else:
                filters.sort()
                retfilternamestring = "&".join(filters)
        
        except KeyError:
            return None
        
        return str(retfilternamestring)
    
    def fpmask(self, dataset):
        """
        Return the fpmask value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            fpmask = hdu[0].header[stdkeyDictGMOS["key_gmos_fpmask"]]

            if fpmask == "None":
                retfpmaskstring = "Imaging"
            else:
                retfpmaskstring = fpmask
        
        except KeyError:
            return None
        
        return str(retfpmaskstring)
    
    def gain(self, dataset):
        """
        Return the gain value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: list
        @returns: an array of gain values (electrons/ADU), where the number
        of array elements equals the number of extensions in the image.
        Index 0 is the gain of the first extension containing pixels
        (equivalent to EXTNAME = "SCI") and so on.
        """
        try:
            hdu = dataset.hdulist
            ampinteg = hdu[0].header[stdkeyDictGMOS["key_gmos_ampinteg"]]
            utdate = hdu[0].header[stdkeyDictGMOS["key_gmos_utdate"]]
            obsutdate = datetime(*strptime(utdate, "%Y-%m-%d")[0:6])
            oldutdate = datetime(2006,8,31,0,0)
            
            retgainlist = []
            for ext in dataset:
                # get the values - GMOS raw data can have up to 6 data extensions
                headergain = ext.header[stdkeyDictGMOS["key_gmos_gain"]]
                ampname = ext.header[stdkeyDictGMOS["key_gmos_ampname"]]
                
                # gmode
                if (headergain > 3.0):
                    gmode = "high"
                else:
                    gmode = "low"
                
                # rmode
                if (ampinteg == None):
                    rmode = "slow"
                else:
                    if (ampinteg == 1000):
                        rmode = "fast"
                    else:
                        rmode = "slow"
                
                gainkey = (rmode, gmode, ampname)
                
                try:
                    if (obsutdate > oldutdate):
                        gain = self.gmosampsGain[gainkey]
                    else:
                        gain = self.gmosampsGainBefore20060831[gainkey]
                
                except KeyError:
                    gain = None
                
                retgainlist.append(gain)
        
        except KeyError:
            return None
        
        return retgainlist
    
    gmosampsGain = None
    gmosampsGainBefore20060831 = None
    
    def instrument(self, dataset):
        """
        Return the instrument value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the instrument used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retinstrumentstring = hdu[0].header[stdkeyDictGMOS["key_gmos_instrument"]]
        
        except KeyError:
            return None
                        
        return str(retinstrumentstring)
    
    def mdfrow(self, dataset):
        """
        Return the mdfrow value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the corresponding reference row in the MDF
        """
        retmdfrowint = None
        
        return retmdfrowint
    
    def nonlinear(self, dataset):
        """
        Return the nonlinear value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        retnonlinearint = None
        
        return retnonlinearint
    
    def nsciext(self, dataset):
        """
        Return the nsciext value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        retnsciextint = dataset.countExts(None)

        return int(retnsciextint)
    
    def object(self, dataset):
        """
        Return the object value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the name of the object acquired
        """
        try:
            hdu = dataset.hdulist
            retobjectstring = hdu[0].header[stdkeyDictGMOS["key_gmos_object"]]
        
        except KeyError:
            return None
                        
        return str(retobjectstring)
    
    def obsmode(self, dataset):
        """
        Return the obsmode value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        try:
            hdu = dataset.hdulist
            masktype = hdu[0].header[stdkeyDictGMOS["key_gmos_masktype"]]
            maskname = hdu[0].header[stdkeyDictGMOS["key_gmos_maskname"]]
            grating = hdu[0].header[stdkeyDictGMOS["key_gmos_grating"]]
            
            if masktype == 0:
                retobsmodestring = "IMAGE"
            
            elif masktype == -1:
                retobsmodestring = "IFU"
            
            elif masktype == 1:
                
                if re.search("arcsec", maskname) != None and re.search("NS", maskname) == None:
                    retobsmodestring = "LONGSLIT"
                else:
                    retobsmodestring = "MOS"
            
            else:
                # if obsmode cannot be determined, set it equal to IMAGE instead of crashing
                retobsmodestring = "IMAGE"

            # mask or IFU cannot be used without grating
            if grating == "MIRROR" and masktype != 0:
                retobsmodestring == "IMAGE" 
        
        except KeyError:
            return None
        
        return str(retobsmodestring)
    
    def pixscale(self, dataset):
        """
        Return the pixscale value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        try:
            hdu = dataset.hdulist
            instrument = hdu[0].header[stdkeyDictGMOS["key_gmos_instrument"]]
            
            for ext in dataset:
                # get the values - GMOS raw data can have up to 6 data extensions
                ccdsum = ext.header[stdkeyDictGMOS["key_gmos_ccdsum"]]
            
            if instrument == "GMOS-N":
                scale = 0.0727
            if instrument == "GMOS-S":
                scale = 0.073
            
            if ccdsum != None:
                xccdbin, yccdbin = ccdsum.split()
                retpixscalefloat = float(yccdbin) * scale
            else:
                retpixscalefloat = scale
        
        except KeyError:
            return None
        
        return float(retpixscalefloat)
    
    def pupilmask(self, dataset):
        """
        Return the pupilmask value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        retpupilmaskstring = None
        
        return str(retpupilmaskstring)
    
    def rdnoise(self, dataset):
        """
        Return the rdnoise value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: list
        @returns: an array of estimated readout noise values (electrons),
        where the number of array elements equals the number of extensions
        in the image. Index 0 is the estimated readout noise of the first
        extension containing pixels (equivalent to EXTNAME = "SCI") and so on.
        """
        try:
            hdu = dataset.hdulist
            ampinteg = hdu[0].header[stdkeyDictGMOS["key_gmos_ampinteg"]]
            utdate = hdu[0].header[stdkeyDictGMOS["key_gmos_utdate"]]
            obsutdate = datetime(*strptime(utdate, "%Y-%m-%d")[0:6])
            oldutdate = datetime(2006,8,31,0,0)
            
            retrdnoiselist = []
            for ext in dataset:
                # get the values - GMOS raw data can have up to 6 data extensions
                headergain = ext.header[stdkeyDictGMOS["key_gmos_gain"]]
                ampname = ext.header[stdkeyDictGMOS["key_gmos_ampname"]]
                
                # gmode
                if (headergain > 3.0):
                    gmode = "high"
                else:
                    gmode = "low"
                
                # rmode
                if (ampinteg == None):
                    rmode = "slow"
                else:
                    if (ampinteg == 1000):
                        rmode = "fast"
                    else:
                        rmode = "slow"
                
                rdnoisekey = (rmode, gmode, ampname)
                
                try:
                    if (obsutdate > oldutdate):
                        rdnoise = self.gmosampsRdnoise[rdnoisekey]
                    else:
                        rdnoise = self.gmosampsRdnoiseBefore20060831[rdnoisekey]
                
                except KeyError:
                    rdnoise = None
                
                retrdnoiselist.append(rdnoise)
        
        except KeyError:
            return None
        
        return retrdnoiselist
    
    gmosampsRdnoise = None
    gmosampsRdnoiseBefore20060831 = None
    
    def satlevel(self, dataset):
        """
        Return the satlevel value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        retsaturationint = 65000
        
        return int(retsaturationint)
    
    def utdate(self, dataset):
        """
        Return the utdate value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT date of the observation (YYYY-MM-DD)
        """
        try:
            hdu = dataset.hdulist
            retutdatestring = hdu[0].header[stdkeyDictGMOS["key_gmos_utdate"]]
        
        except KeyError:
            return None
        
        return str(retutdatestring)
    
    def uttime(self, dataset):
        """
        Return the uttime value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT at the start of the observation (HH:MM:SS.S)
        """
        try:
            hdu = dataset.hdulist
            retuttimestring = hdu[0].header[stdkeyDictGMOS["key_gmos_uttime"]]
        
        except KeyError:
            return None
        
        return str(retuttimestring)
    
    def wdelta(self, dataset):
        """
        Return the wdelta value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: list
        @returns: an array of dispersion values (angstroms/pixel), where the
        number of array elements equals the number of extensions in the
        image. Index 0 is the data section of the first extension containing
        pixels (equivalent to EXTNAME = "SCI") and so on.
        """
        retwdeltalist = None
        
        return retwdeltalist
    
    def wrefpix(self, dataset):
        """
        Return the wrefpix value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the reference pixel of the central wavelength
        """
        retwrefpixfloat = None
        
        return retwrefpixfloat
    
    def xccdbin(self, dataset):
        """
        Return the xccdbin value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector x-axis
        """
        try:
            for ext in dataset:
                # get the values - GMOS raw data can have up to 6 data extensions
                ccdsum = ext.header[stdkeyDictGMOS["key_gmos_ccdsum"]]
            
            retxccdbinint, yccdbin = ccdsum.split()
        
        except KeyError:
            return None
        
        return int(retxccdbinint)
    
    def yccdbin(self, dataset):
        """
        Return the yccdbin value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector y-axis
        """
        try:
            for ext in dataset:
                # get the values - GMOS raw data can have up to 6 data extensions
                ccdsum = ext.header[stdkeyDictGMOS["key_gmos_ccdsum"]]
            
            xccdbin, retyccdbinint = ccdsum.split()
        
        except KeyError:
            return None
        
        return int(retyccdbinint)
