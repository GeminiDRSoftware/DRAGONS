from astrodata import Lookups
from astrodata import Descriptors
from astrodata import Errors
import math

import astrodata
from astrodata.Calculator import Calculator

import GemCalcUtil 
from StandardNIRIKeyDict import stdkeyDictNIRI
import sys
import re

class NIRI_RAWDescriptorCalc(Calculator):
    niriFilternameMapConfig = None

    niriFilternameMap = {}
    niriSpecDict = None
    
    def __init__(self):
        self.niriSpecDict = Lookups.getLookupTable("Gemini/NIRI/NIRISpecDict", "niriSpecDict")
        self.niriFilternameMapConfig = Lookups.getLookupTable("Gemini/NIRI/NIRIFilterMap", "niriFilternameMapConfig")
        self.nsappwave = Lookups.getLookupTable("Gemini/IR/nsappwavepp.fits", 1)
        self.makeFilternameMap()
        
    def airmass(self, dataset):
        """
        Return the airmass value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the mean airmass for the observation
        """
        try:
            hdu = dataset.hdulist
            retairmassfloat = hdu[0].header[stdkeyDictNIRI["key_niri_airmass"]]
        
        except KeyError:
            return None
        
        return float(retairmassfloat)
    
    def camera(self, dataset):
        """
        Return the camera value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retcamerastring = hdu[0].header[stdkeyDictNIRI["key_niri_camera"]]
        
        except KeyError:
            return None
        
        return str(retcamerastring)
    
    def cwave(self, dataset):
        """
        Return the cwave value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        retcwavefloat = None

        fpmask = self.fpmask(dataset)
        dname  = self.disperser(dataset)
        
        for row in self.nsappwave.data:
            if fpmask == row.field("MASK") and dname == row.field("GRATING"):
                retcwavefloat = float(row.field("LAMBDA"))
        
        retcwavefloat /= 10000 # in header in ansgrtoms, convert to microns, cwaves unit
        return retcwavefloat
    
    def datasec(self, dataset):
        """
        Return the datasec value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the data section
        """
        retdatasecstring = None
        
        return str(retdatasecstring)
    
    def detsec(self, dataset):
        """/home/rtfuser/demo/gemini_python/trunk/emma.descriptors/files
        Return the detsec value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the detector section
        """
        retdetsecstring = None
        
        return str(retdetsecstring)
    
    def disperser(self, dataset, stripID=False, pretty=False):
        """
        Return the disperser value for NIRI
        @param dataset: the data set
        @param stripID: set to True to remove the component ID from the returned disperser name
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        # No specific pretty names, just stripID
        if(pretty):
          stripID = True

        # This seems overkill to me - dispersers can only ever be in filter3
        # becasue the other two wheels are in an uncollimated beam... - PH

        retdisperserstring = None
        try:
            hdu = dataset.hdulist
            filter1 = hdu[0].header[stdkeyDictNIRI["key_niri_filter1"]]
            filter2 = hdu[0].header[stdkeyDictNIRI["key_niri_filter2"]]
            filter3 = hdu[0].header[stdkeyDictNIRI["key_niri_filter3"]]
            #filter1 = GemCalcUtil.removeComponentID(filter1)
            #filter2 = GemCalcUtil.removeComponentID(filter2)
            #filter3 = GemCalcUtil.removeComponentID(filter3)
        except KeyError:
            return None
        diskey = "grism"
        if diskey in filter1:
            retdisperserstring = filter1
        if diskey in filter2:
            retdisperserstring = filter2
        if diskey in filter3:
            retdisperserstring = filter3

        if (stripID):
          retdisperserstring = GemCalcUtil.removeComponentID(retdisperserstring)

        return str(retdisperserstring)
        
    def exptime(self, dataset):
        """
        Return the exptime value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exptime = hdu[0].header[stdkeyDictNIRI["key_niri_exptime"]]
            coadds = hdu[0].header[stdkeyDictNIRI["key_niri_coadds"]]
            if dataset.isType("NIRI_RAW") == True:
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
    
    def filterid(self, dataset):
        """
        Return the filterid value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter ID number string
        """
        retfilteridstring = None
        
        return str(retfilteridstring)
    
    def filtername(self, dataset, pretty = False, stripID = False):
        """
        Return the filtername value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        # To match against the LUT to get the pretty name, we need the component IDs attached
        if(pretty):
            stripID=False

        try:
            hdu = dataset.hdulist
            filter1 = hdu[0].header[stdkeyDictNIRI["key_niri_filter1"]]
            filter2 = hdu[0].header[stdkeyDictNIRI["key_niri_filter2"]]
            filter3 = hdu[0].header[stdkeyDictNIRI["key_niri_filter3"]]
            if (stripID):
                filter1 = GemCalcUtil.removeComponentID(filter1)
                filter2 = GemCalcUtil.removeComponentID(filter2)
                filter3 = GemCalcUtil.removeComponentID(filter3)
            
            # create list of filter values
            filters = [filter1,filter2,filter3]
            retfilternamestring = self.filternameFrom(filters)
           
        except KeyError:
            return None
        
        # If pretty output, map to science name of filtername in table
        # To match against the LUT, the filter list must be sorted
        if (pretty):
            filters.sort()
            retfilternamestring = self.filternameFrom(filters)
            if retfilternamestring in self.niriFilternameMap:
                retfilternamestring = self.niriFilternameMap[retfilternamestring]
            
        return str(retfilternamestring)
    
    def fpmask(self, dataset):
        """
        Return the fpmask value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retfpmaskstring = hdu[0].header[stdkeyDictNIRI["key_niri_fpmask"]]
        
        except KeyError:
            return None
                        
        return str(retfpmaskstring)
    
    def gain(self, dataset):
        """
        Return the gain value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        try:
            retgainfloat = self.niriSpecDict["gain"]
        
        except KeyError:
            return None
        
        return float(retgainfloat)
    
    niriSpecDict = None
    
    def instrument(self, dataset):
        """
        Return the instrument value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the instrument used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retinstrumentstring = hdu[0].header[stdkeyDictNIRI["key_niri_instrument"]]
        
        except KeyError:
            return None
                        
        return str(retinstrumentstring)
    
    def mdfrow(self, dataset):
        """
        Return the mdfrow value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the corresponding reference row in the MDF
        """
        retmdfrowint = None
        
        return retmdfrowint
    
    def nonlinear(self, dataset):
        """
        Return the nonlinear value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        try:
            hdu = dataset.hdulist
            avdduc = hdu[0].header[stdkeyDictNIRI["key_niri_avdduc"]]
            avdet = hdu[0].header[stdkeyDictNIRI["key_niri_avdet"]]
            coadds = hdu[0].header[stdkeyDictNIRI["key_niri_coadds"]]
            
            gain = self.niriSpecDict["gain"]
            shallowwell = self.niriSpecDict["shallowwell"]
            deepwell = self.niriSpecDict["deepwell"]
            shallowbias = self.niriSpecDict["shallowbias"]
            deepbias = self.niriSpecDict["deepbias"]
            linearlimit = self.niriSpecDict["linearlimit"]
            
            biasvolt = avdduc - avdet
            #biasvolt = 100

            if abs(biasvolt - shallowbias) < 0.05:
                saturation = int(shallowwell * coadds / gain)
            
            elif abs(biasvolt - deepbias) < 0.05:
                saturation = int(deepwell * coadds / gain)
            
            else:
                raise Errors.CalcError()
        
        except Errors.CalcError, c:
            return c.message
            
        except KeyError, k:
            if k.message[0:3]=="Key":
                return k.message
            else:
                return "%s not found." % k.message

        else:
            retnonlinearint = saturation * linearlimit
            return int(retnonlinearint)
    
    niriSpecDict = None
    
    def nsciext(self, dataset):
        """
        Return the nsciext value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        retnsciextint = dataset.countExts(None)
        
        return int(retnsciextint)
    
    def object(self, dataset):
        """
        Return the object value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the name of the object acquired
        """
        try:
            hdu = dataset.hdulist
            retobjectstring = hdu[0].header[stdkeyDictNIRI["key_niri_object"]]
        
        except KeyError:
            return None
                        
        return str(retobjectstring)
    
    def obsmode(self, dataset):
        """
        Return the obsmode value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the observing mode
        """
        try:
            raise Errors.ExistError()

        except Errors.ExistError, e:
            return e.message
    
    def pixscale(self, dataset):
        """
        Return the pixscale value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        try:
            hdu = dataset.hdulist
            cd11 = hdu[0].header[stdkeyDictNIRI["key_niri_cd11"]]
            cd12 = hdu[0].header[stdkeyDictNIRI["key_niri_cd12"]]
            cd21 = hdu[0].header[stdkeyDictNIRI["key_niri_cd21"]]
            cd22 = hdu[0].header[stdkeyDictNIRI["key_niri_cd22"]]
            
            retpixscalefloat = 3600 * (math.sqrt(math.pow(cd11,2) + math.pow(cd12,2)) + math.sqrt(math.pow(cd21,2) + math.pow(cd22,2))) / 2
        
        except KeyError:
            return None
        
        return float(retpixscalefloat)
    
    def pupilmask(self, dataset):
        """
        Return the pupilmask value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        try:
            hdu = dataset.hdulist
            filter3 = hdu[0].header[stdkeyDictNIRI["key_niri_filter3"]]
            
            if filter3[:3] == "pup":
                pupilmask = filter3
                
                if pupilmask[-6:-4] == "_G":
                    retpupilmaskstring = pupilmask[:-6]
                else:
                    retpupilmaskstring = pupilmask
            
            else:
                return None
        
        except KeyError:
            return None
        
        return str(retpupilmaskstring)
    
    def rdnoise(self, dataset):
        """
        Return the rdnoise value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise (electrons)
        """
        try:
            hdu = dataset.hdulist
            lnrs = hdu[0].header[stdkeyDictNIRI["key_niri_lnrs"]]
            ndavgs = hdu[0].header[stdkeyDictNIRI["key_niri_ndavgs"]]
            coadds = hdu[0].header[stdkeyDictNIRI["key_niri_coadds"]]
            
            readnoise = self.niriSpecDict["readnoise"]
            medreadnoise = self.niriSpecDict["medreadnoise"]
            lowreadnoise = self.niriSpecDict["lowreadnoise"]
            
            if lnrs == 1 and ndavgs == 1:
                retrdnoisefloat = readnoise * math.sqrt(coadds)
            elif lnrs == 1 and ndavgs == 16:
                retrdnoisefloat = medreadnoise * math.sqrt(coadds)
            elif lnrs == 16 and ndavgs == 16:
                retrdnoisefloat = lowreadnoise * math.sqrt(coadds)
            else:
                retrdnoisefloat = medreadnoise * math.sqrt(coadds)
        
        except KeyError:
            return None
        
        return float(retrdnoisefloat)
    
    niriSpecDict = None
    
    def satlevel(self, dataset):
        """
        Return the satlevel value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        try:
            hdu = dataset.hdulist
            
            avdduc = hdu[0].header[stdkeyDictNIRI["key_niri_avdduc"]]
            avdet = hdu[0].header[stdkeyDictNIRI["key_niri_avdet"]]
            coadds = hdu[0].header[stdkeyDictNIRI["key_niri_coadds"]]
            
            gain = self.niriSpecDict["gain"]
            shallowwell = self.niriSpecDict["shallowwell"]
            deepwell = self.niriSpecDict["deepwell"]
            shallowbias = self.niriSpecDict["shallowbias"]
            deepbias = self.niriSpecDict["deepbias"]
            linearlimit = self.niriSpecDict["linearlimit"]
            
            biasvolt = avdduc - avdet
            
            if abs(biasvolt - shallowbias) < 0.05:
                retsaturationint = int(shallowwell * coadds / gain)
            
            elif abs(biasvolt - deepbias) < 0.05:
                retsaturationint = int(deepwell * coadds / gain)
            
            else:
                return None
            
        except KeyError:
            return None
        
        return int(retsaturationint)
    
    niriSpecDict = None
    
    def utdate(self, dataset):
        """
        Return the utdate value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT date of the observation (YYYY-MM-DD)
        """
        try:
            hdu = dataset.hdulist
            retutdatestring = hdu[0].header[stdkeyDictNIRI["key_niri_utdate"]]
        
        except KeyError:
            return None
        
        return str(retutdatestring)
    
    def uttime(self, dataset):
        """
        Return the uttime value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT at the start of the observation (HH:MM:SS.S)
        """
        try:
            hdu = dataset.hdulist
            retuttimestring = hdu[0].header[stdkeyDictNIRI["key_niri_uttime"]]
        
        except KeyError:
            return None
        
        return str(retuttimestring)
    
    def wdelta(self, dataset):
        """
        Return the wdelta value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the dispersion (angstroms/pixel)
        """
        retwdeltafloat = None
        
        return retwdeltafloat
    
    def wrefpix(self, dataset):
        """
        Return the wrefpix value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the reference pixel of the central wavelength
        """
        retwrefpixfloat = None
        
        return retwrefpixfloat
    
    def xccdbin(self, dataset):
        """
        Return the xccdbin value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector x-axis
        """
        retxccdbinint = None
        
        return retxccdbinint
    
    def yccdbin(self, dataset):
        """
        Return the yccdbin value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the binning of the detector y-axis
        """
        retyccdbinint = None
        
        return retyccdbinint

    ## UTILITY MEMBER FUNCTIONS (NOT DESCRIPTORS)
    
    def filternameFrom(self, filters):
        
        # reject "open" "grism" and "pupil"
        filters2 = []
        for filt in filters:
            filtlow = filt.lower()
            if ("open" in filtlow) or ("grism" in filtlow) or ("pupil" in filtlow):
                pass
            else:
                filters2.append(filt)

        filters = filters2

        # blank means an opaque mask was in place, which of course
        # blocks any other in place filters
        if "blank" in filters:
            retfilternamestring = "blank"
        elif len(filters) == 0:
            retfilternamestring = "open"
        else:
            filters.sort()
            retfilternamestring = "&".join(filters)
        return retfilternamestring
            
    def makeFilternameMap(self):
        filternamemap = {}
        for line in self.niriFilternameMapConfig:
            linefiltername = self.filternameFrom( [line[1], line[2], line[3]])
            filternamemap.update({linefiltername:line[0] })
        self.niriFilternameMap = filternamemap
