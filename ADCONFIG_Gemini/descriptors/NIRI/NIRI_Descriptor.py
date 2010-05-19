from astrodata import Lookups
from astrodata import Descriptors
from astrodata import Errors
import math

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardNIRIKeyDict import stdkeyDictNIRI
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class NIRI_DescriptorCalc(GEMINI_DescriptorCalc):
    
    niriFilternameMapConfig = None
    niriFilternameMap = {}
    niriSpecDict = None
    
    def __init__(self):
        self.niriSpecDict = \
            Lookups.getLookupTable('Gemini/NIRI/NIRISpecDict',
                                   'niriSpecDict')
        self.niriFilternameMapConfig = \
            Lookups.getLookupTable('Gemini/NIRI/NIRIFilterMap',
                                   'niriFilternameMapConfig')
        
        self.nsappwave =
            Lookups.getLookupTable('Gemini/IR/nsappwavepp.fits', 1)

        self.makeFilternameMap()
        
    def camera(self, dataset, **args):
        """
        Return the camera value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retcamerastring = hdu[0].header[stdkeyDictNIRI['key_niri_camera']]
        
        except KeyError:
            return None
        
        return str(retcamerastring)
    
    def cwave(self, dataset, **args):
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
            if fpmask == row.field('MASK') and dname == row.field('GRATING'):
                retcwavefloat = float(row.field('LAMBDA'))
        
        retcwavefloat /= 10000
        # in header in angstroms, convert to microns, cwaves unit
        return retcwavefloat
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
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
            filter1 = hdu[0].header[stdkeyDictNIRI['key_niri_filter1']]
            filter2 = hdu[0].header[stdkeyDictNIRI['key_niri_filter2']]
            filter3 = hdu[0].header[stdkeyDictNIRI['key_niri_filter3']]
            #filter1 = GemCalcUtil.removeComponentID(filter1)
            #filter2 = GemCalcUtil.removeComponentID(filter2)
            #filter3 = GemCalcUtil.removeComponentID(filter3)
        except KeyError:
            return None
        diskey = 'grism'
        if diskey in filter1:
            retdisperserstring = filter1
        if diskey in filter2:
            retdisperserstring = filter2
        if diskey in filter3:
            retdisperserstring = filter3

        if (stripID):
          retdisperserstring = GemCalcUtil.removeComponentID(retdisperserstring)

        return str(retdisperserstring)
        
    def exptime(self, dataset, **args):
        """
        Return the exptime value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exptime = hdu[0].header[stdkeyDictNIRI['key_niri_exptime']]
            coadds = hdu[0].header[stdkeyDictNIRI['key_niri_coadds']]
            if dataset.isType('NIRI_RAW') == True:
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
    
    def coadds(self, dataset, **args):
        """
        Return the number of coadds for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of coadds
        """
        try:
            hdu = dataset.hdulist
            retcoaddsint = hdu[0].header[stdkeyDictNIRI["key_niri_coadds"]]
        
        except KeyError:
            return None

        return int(retcoaddsint)
    
    def filtername(self, dataset, pretty = False, stripID = False, **args):
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
            filter1 = hdu[0].header[stdkeyDictNIRI['key_niri_filter1']]
            filter2 = hdu[0].header[stdkeyDictNIRI['key_niri_filter2']]
            filter3 = hdu[0].header[stdkeyDictNIRI['key_niri_filter3']]
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
    
    def fpmask(self, dataset, **args):
        """
        Return the fpmask value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retfpmaskstring = hdu[0].header[stdkeyDictNIRI['key_niri_fpmask']]
        
        except KeyError:
            return None
                        
        return str(retfpmaskstring)
    
    def gain(self, dataset, **args):
        """
        Return the gain value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the gain (electrons/ADU)
        """
        try:
            retgainfloat = self.niriSpecDict['gain']
        
        except KeyError:
            return None
        
        return float(retgainfloat)
    
    niriSpecDict = None
    
    def nonlinear(self, dataset, **args):
        """
        Return the nonlinear value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the non-linear level in the raw images (ADU)
        """
        try:
            hdu = dataset.hdulist
            avdduc = hdu[0].header[stdkeyDictNIRI['key_niri_avdduc']]
            avdet = hdu[0].header[stdkeyDictNIRI['key_niri_avdet']]
            coadds = hdu[0].header[stdkeyDictNIRI['key_niri_coadds']]
            
            gain = self.niriSpecDict['gain']
            shallowwell = self.niriSpecDict['shallowwell']
            deepwell = self.niriSpecDict['deepwell']
            shallowbias = self.niriSpecDict['shallowbias']
            deepbias = self.niriSpecDict['deepbias']
            linearlimit = self.niriSpecDict['linearlimit']
            
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
            if k.message[0:3]=='Key':
                return k.message
            else:
                return '%s not found.' % k.message

        else:
            retnonlinearint = saturation * linearlimit
            return int(retnonlinearint)
    
    niriSpecDict = None
    
    def nsciext(self, dataset, **args):
        """
        Return the nsciext value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the number of science extensions
        """
        retnsciextint = dataset.countExts('SCI')
        
        return int(retnsciextint)
    
    def obsmode(self, dataset, **args):
        """
        Return the NIRI datasec. This has to be derived from the
        first extension, but pertains to the whole data file
        """
        try:
            hdu = dataset.hdulist
            x_start = hdu[1].header["LOWROW"]
            x_end = hdu[1].header["HIROW"]
            y_start = hdu[1].header["LOWCOL"]
            y_end = hdu[1].header["HICOL"]

            retdetsec = '[%d:%d,%d:%d]' % (x_start, x_end, y_start, y_end)

        except KeyError:
            return None

        return retdetsec
 

    def pixscale(self, dataset, **args):
        """
        Return the pixscale value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the pixel scale (arcsec/pixel)
        """
        try:
            hdu = dataset.hdulist
            cd11 = hdu[0].header[stdkeyDictNIRI['key_niri_cd11']]
            cd12 = hdu[0].header[stdkeyDictNIRI['key_niri_cd12']]
            cd21 = hdu[0].header[stdkeyDictNIRI['key_niri_cd21']]
            cd22 = hdu[0].header[stdkeyDictNIRI['key_niri_cd22']]
            
            retpixscalefloat = 3600 * (math.sqrt(math.pow(cd11,2) + math.pow(cd12,2)) + math.sqrt(math.pow(cd21,2) + math.pow(cd22,2))) / 2
        
        except KeyError:
            return None
        
        return float(retpixscalefloat)
    
    def pupilmask(self, dataset, **args):
        """
        Return the pupilmask value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the pupil mask used to acquire data
        """
        try:
            hdu = dataset.hdulist
            filter3 = hdu[0].header[stdkeyDictNIRI['key_niri_filter3']]
            
            if filter3[:3] == 'pup':
                pupilmask = filter3
                
                if pupilmask[-6:-4] == '_G':
                    retpupilmaskstring = pupilmask[:-6]
                else:
                    retpupilmaskstring = pupilmask
            
            else:
                return None
        
        except KeyError:
            return None
        
        return str(retpupilmaskstring)
    
    def rdnoise(self, dataset, **args):
        """
        Return the rdnoise value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the estimated readout noise (electrons)
        """
        try:
            hdu = dataset.hdulist
            lnrs = hdu[0].header[stdkeyDictNIRI['key_niri_lnrs']]
            ndavgs = hdu[0].header[stdkeyDictNIRI['key_niri_ndavgs']]
            coadds = hdu[0].header[stdkeyDictNIRI['key_niri_coadds']]
            
            readnoise = self.niriSpecDict['readnoise']
            medreadnoise = self.niriSpecDict['medreadnoise']
            lowreadnoise = self.niriSpecDict['lowreadnoise']
            
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
    
    def readmode(self, dataset):
	"""
        Returns the Read Mode for NIRI. This is either "Low Background",
        "Medium Background" or "High Background" as in the OT.
        Returns 'Invalid' if the headers don't make sense wrt these
        defined modes.
	"""
	try:
	    hdu = dataset.hdulist

            lnrs = hdu[0].header[stdkeyDictNIRI["key_niri_lnrs"]]
            ndavgs = hdu[0].header[stdkeyDictNIRI["key_niri_ndavgs"]]

            readmode = "Invalid"

            if((lnrs==16) and (ndavgs==16)):
                readmode = "Low Background"

            if((lnrs==1) and (ndavgs==16)):
                readmode = "Medium Background"

            if((lnrs==1) and (ndavgs==1)):
                readmode = "High Background"

            return readmode

	except KeyError:
	    return None

    def satlevel(self, dataset, **args):
        """
        Return the satlevel value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @returns: the saturation level in the raw images (ADU)
        """
        try:
            hdu = dataset.hdulist
            
            avdduc = hdu[0].header[stdkeyDictNIRI['key_niri_avdduc']]
            avdet = hdu[0].header[stdkeyDictNIRI['key_niri_avdet']]
            coadds = hdu[0].header[stdkeyDictNIRI['key_niri_coadds']]
            
            gain = self.niriSpecDict['gain']
            shallowwell = self.niriSpecDict['shallowwell']
            deepwell = self.niriSpecDict['deepwell']
            shallowbias = self.niriSpecDict['shallowbias']
            deepbias = self.niriSpecDict['deepbias']
            linearlimit = self.niriSpecDict['linearlimit']
            
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
    
    def wdelta(self, dataset, **args):
        """
        Return the wdelta value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the dispersion (angstroms/pixel)
        """
        retwdeltafloat = None
        
        return retwdeltafloat
    
    def welldepthmode(self, dataset):
	"""
        Returns the well depth mode for NIRI. This is either "Deep" or
        "Shallow" as in the OT
        Returns 'Invalid' if the bias numbers aren't what we normally use
        Uses parameters in the niriSpecDict dictionary
	"""
        try:
            hdu = dataset.hdulist

            vdduc = hdu[0].header[stdkeyDictNIRI["key_niri_avdduc"]]
            vdet = hdu[0].header[stdkeyDictNIRI["key_niri_avdet"]]

            biasvolt = vdduc - vdet

            shallowbias = self.niriSpecDict["shallowbias"]
            deepbias = self.niriSpecDict["deepbias"]

            welldepthmode = 'Invalid'

            if abs(biasvolt - shallowbias) < 0.05:
		welldepthmode = 'Shallow'

            if abs(biasvolt - deepbias) < 0.05:
		welldepthmode = 'Deep'

	    return welldepthmode

        except KeyError:
            return None

    def wrefpix(self, dataset, **args):
        """
        Return the wrefpix value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @returns: the reference pixel of the central wavelength
        """
        retwrefpixfloat = None
        
        return retwrefpixfloat
    
    ## UTILITY MEMBER FUNCTIONS (NOT DESCRIPTORS)
    
    def filternameFrom(self, filters, **args):
        
        # reject 'open' 'grism' and 'pupil'
        filters2 = []
        for filt in filters:
            filtlow = filt.lower()
            if ('open' in filtlow) or ('grism' in filtlow) or ('pupil' in filtlow):
                pass
            else:
                filters2.append(filt)

        filters = filters2

        # blank means an opaque mask was in place, which of course
        # blocks any other in place filters
        if 'blank' in filters:
            retfilternamestring = 'blank'
        elif len(filters) == 0:
            retfilternamestring = 'open'
        else:
            filters.sort()
            retfilternamestring = '&'.join(filters)
        return retfilternamestring
            
    def makeFilternameMap(self, **args):
        filternamemap = {}
        for line in self.niriFilternameMapConfig:
            linefiltername = self.filternameFrom( [line[1], line[2], line[3]])
            filternamemap.update({linefiltername:line[0] })
        self.niriFilternameMap = filternamemap
