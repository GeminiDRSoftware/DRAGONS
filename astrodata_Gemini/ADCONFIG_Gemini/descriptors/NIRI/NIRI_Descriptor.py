from astrodata import Lookups
from astrodata import Descriptors
from astrodata import Errors
import math

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardNIRIKeyDict import stdkeyDictNIRI
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class NIRI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dict with the local dict of this descriptor class
    globalStdkeyDict.update(stdkeyDictNIRI)
    
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
        self.makeFilternameMap()
        
        self.nsappwave = \
            Lookups.getLookupTable('Gemini/IR/nsappwavepp.fits', 1)
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        focal_plane_mask = dataset.focal_plane_mask()
        disperser = dataset.disperser()
        
        # This doesn't work yet ... need to sort out nsappwave LUT
        #for row in self.nsappwave.data:
        #    if focal_plane_mask == row.field('MASK') and \
        #        disperser == row.field('GRATING'):
        #    central_wavelength = float(row.field('LAMBDA'))
        
        # Angstroms in header, convert to central_wavelength units
        # (i.e., nanometers)
        #ret_central_wavelength = central_wavelength / 10.
        
        # This doesn't work until we define ret_central_wavelength above
        #return float(ret_central_wavelength)
        
        return None
    
    def data_section(self, dataset, **args):
        """
        Return the data_section value for NIRI
        This has to be derived from the first extension, but pertains to the
        whole data file. The pixels are numbered starting from 1, not 0.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the data section
        """
        hdu = dataset.hdulist
        x_start = hdu[1].header[stdkeyDictNIRI['key_lowrow']]
        x_end = hdu[1].header[stdkeyDictNIRI['key_hirow']]
        y_start = hdu[1].header[stdkeyDictNIRI['key_lowcol']]
        y_end = hdu[1].header[stdkeyDictNIRI['key_hicol']]
        
        # The convention is that we start counting pixels from 1 in this case.
        data_section = \
            '[%d:%d,%d:%d]' % (x_start+1, x_end+1, y_start+1, y_end+1)
        
        ret_data_section = str(data_section)
        
        return ret_data_section
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the disperser value for NIRI
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
            stripID = True
        
        # This seems overkill to me - dispersers can only ever be in
        # filter3 becasue the other two wheels are in an uncollimated
        # beam... - PH
        
        hdu = dataset.hdulist
        filter1 = hdu[0].header[stdkeyDictNIRI['key_filter1']]
        filter2 = hdu[0].header[stdkeyDictNIRI['key_filter2']]
        filter3 = hdu[0].header[stdkeyDictNIRI['key_filter3']]
        
        disperser_key = 'grism'
        if disperser_key in filter1:
            disperser = filter1
        elif disperser_key in filter2:
            disperser = filter2
        elif disperser_key in filter3:
            disperser = filter3
        else:
            disperser = None
        
        if stripID and (disperser != None):
            ret_disperser = str(GemCalcUtil.removeComponentID(disperser))
        else:
            ret_disperser = str(disperser)
        
        return ret_disperser
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        hdu = dataset.hdulist
        exposure_time = hdu[0].header[globalStdkeyDict['key_exposure_time']]
        coadds = dataset.coadds()
        
        if dataset.isType('NIRI_RAW') == True and coadds != 1:
            ret_exposure_time = float(exposure_time * coadds)
        else:
            ret_exposure_time = float(exposure_time)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the filter_name value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID from the
        returned filter name
        @param pretty: set to True to return a meaningful filter name
        @rtype: string
        @return: the unique filter identifier string
        """
        # To match against the LUT to get the pretty name, we need the
        # component IDs attached
        if pretty:
            stripID=False
        
        hdu = dataset.hdulist
        filter1 = hdu[0].header[stdkeyDictNIRI['key_filter1']]
        filter2 = hdu[0].header[stdkeyDictNIRI['key_filter2']]
        filter3 = hdu[0].header[stdkeyDictNIRI['key_filter3']]
        
        if stripID:
            filter1 = GemCalcUtil.removeComponentID(filter1)
            filter2 = GemCalcUtil.removeComponentID(filter2)
            filter3 = GemCalcUtil.removeComponentID(filter3)
        
        # Create list of filter values
        filters = [filter1,filter2,filter3]
        
        # If pretty output, map to science name of filtername in table
        # To match against the LUT, the filter list must be sorted
        if pretty:
            filters.sort()
            filter_name = self.filternameFrom(filters)
            if filter_name in self.niriFilternameMap:
                ret_filter_name = str(self.niriFilternameMap[filter_name])
            else:
                ret_filter_name = filter_name
        else:
            ret_filter_name = str(self.filternameFrom(filters))
        
        return ret_filter_name
    
    def gain(self, dataset, **args):
        """
        Return the gain value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the gain (electrons/ADU)
        """
        gain = self.niriSpecDict['gain']
        
        ret_gain = float(gain)
        
        return ret_gain
    
    niriSpecDict = None
    
    def non_linear_level(self, dataset, **args):
        """
        Return the non_linear_level value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the non-linear level in the raw images (ADU)
        """
        saturation_level = dataset.saturation_level()
        linearlimit = self.niriSpecDict['linearlimit']
        
        ret_non_linear_level = int(saturation_level * linearlimit)
        
        return ret_non_linear_level
    
    niriSpecDict = None
    
    def pixel_scale(self, dataset, **args):
        """
        Return the pixel_scale value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the pixel scale (arcsec/pixel)
        """
        hdu = dataset.hdulist
        cd11 = hdu[0].header[stdkeyDictNIRI['key_cd11']]
        cd12 = hdu[0].header[stdkeyDictNIRI['key_cd12']]
        cd21 = hdu[0].header[stdkeyDictNIRI['key_cd21']]
        cd22 = hdu[0].header[stdkeyDictNIRI['key_cd22']]
        
        pixel_scale = 3600 * \
            (math.sqrt(math.pow(cd11,2) + math.pow(cd12,2)) + \
            math.sqrt(math.pow(cd21,2) + math.pow(cd22,2))) / 2
        
        ret_pixel_scale = float(pixel_scale)
        
        return ret_pixel_scale
    
    def pupil_mask(self, dataset, **args):
        """
        Return the pupil_mask value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the pupil mask used to acquire data
        """
        hdu = dataset.hdulist
        filter3 = hdu[0].header[stdkeyDictNIRI['key_filter3']]
        
        if filter3[:3] == 'pup':
            pupil_mask = filter3
            
            if pupil_mask[-6:-4] == '_G':
                ret_pupil_mask = str(pupil_mask[:-6])
            else:
                ret_pupil_mask = str(pupil_mask)
        else:
            ret_pupil_mask = None
        
        return ret_pupil_mask
    
    def read_mode(self, dataset, **args):
        """
        Return the read_noise value for NIRI
        This is either 'Low Background', 'Medium Background' or
        'High Background', as in the OT. Returns 'Invalid' if the headers
        don't make sense w.r.t. these defined modes.
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the readout mode
        """
        hdu = dataset.hdulist
        lnrs = hdu[0].header[stdkeyDictNIRI['key_lnrs']]
        ndavgs = hdu[0].header[stdkeyDictNIRI['key_ndavgs']]
        
        if lnrs == 16 and ndavgs == 16:
            ret_read_mode = 'Low Background'
        elif lnrs == 1 and ndavgs == 16:
            ret_read_mode = 'Medium Background'
        elif lnrs == 1 and ndavgs == 1:
            ret_read_mode = 'High Background'
        else:
            ret_read_mode = 'Invalid'
        
        return ret_read_mode
    
    def read_noise(self, dataset, **args):
        """
        Return the read_noise value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the estimated readout noise (electrons)
        """
        hdu = dataset.hdulist
        lnrs = hdu[0].header[stdkeyDictNIRI['key_lnrs']]
        ndavgs = hdu[0].header[stdkeyDictNIRI['key_ndavgs']]
        coadds = dataset.coadds()
        
        readnoise = self.niriSpecDict['readnoise']
        medreadnoise = self.niriSpecDict['medreadnoise']
        lowreadnoise = self.niriSpecDict['lowreadnoise']
        
        if lnrs == 1 and ndavgs == 1:
            read_noise = readnoise * math.sqrt(coadds)
        elif lnrs == 1 and ndavgs == 16:
            read_noise = medreadnoise * math.sqrt(coadds)
        elif lnrs == 16 and ndavgs == 16:
            read_noise = lowreadnoise * math.sqrt(coadds)
        else:
            read_noise = medreadnoise * math.sqrt(coadds)
        
        ret_read_noise = float(read_noise)
        
        return ret_read_noise
    
    niriSpecDict = None
    
    def saturation_level(self, dataset, **args):
        """
        Return the saturation_level value for NIRI
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the saturation level in the raw images (ADU)
        """
        hdu = dataset.hdulist
        coadds = dataset.coadds()
        gain = dataset.gain()
        well_depth_setting = dataset.well_depth_setting()
        
        shallowwell = self.niriSpecDict['shallowwell']
        deepwell = self.niriSpecDict['deepwell']
        
        if well_depth_setting == 'Shallow':
            ret_saturation_level = int(shallowwell * coadds / gain)
        elif well_depth_setting == 'Deep':
            ret_saturation_level = int(deepwell * coadds / gain)
        else:
            ret_saturation_level = None
        
        return ret_saturation_level
    
    niriSpecDict = None
    
    def well_depth_setting(self, dataset, **args):
        """
        Return the well_depth_setting value for NIRI
        This is either 'Deep' or 'Shallow' as in the OT. Returns 'Invalid' if
        the bias numbers aren't what we normally use. Uses parameters in the
        niriSpecDict dictionary
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the well depth mode
        """
        hdu = dataset.hdulist
        avdduc = hdu[0].header[stdkeyDictNIRI['key_avdduc']]
        avdet = hdu[0].header[stdkeyDictNIRI['key_avdet']]
        
        biasvolt = avdduc - avdet
        
        shallowbias = self.niriSpecDict['shallowbias']
        deepbias = self.niriSpecDict['deepbias']
        
        if abs(biasvolt - shallowbias) < 0.05:
            ret_well_depth_setting = 'Shallow'
        elif abs(biasvolt - deepbias) < 0.05:
            ret_well_depth_setting = 'Deep'
        else:
            ret_well_depth_setting = 'Invalid'
        
        return ret_well_depth_setting
    
    ## UTILITY MEMBER FUNCTIONS (NOT DESCRIPTORS)
    
    def filternameFrom(self, filters, **args):
        
        # reject 'open' 'grism' and 'pupil'
        filters2 = []
        for filt in filters:
            filtlow = filt.lower()
            if 'open' in filtlow or 'grism' in filtlow or 'pupil' in filtlow:
                pass
            else:
                filters2.append(filt)
        
        filters = filters2
        
        # blank means an opaque mask was in place, which of course
        # blocks any other in place filters
        
        if 'blank' in filters:
            filtername = 'blank'
        elif len(filters) == 0:
            filtername = 'open'
        else:
            filters.sort()
            filtername = str('&'.join(filters))
        
        return filtername
    
    def makeFilternameMap(self, **args):
        filternamemap = {}
        for line in self.niriFilternameMapConfig:
            linefiltername = self.filternameFrom([line[1], line[2], line[3]])
            filternamemap.update({linefiltername:line[0]})
        self.niriFilternameMap = filternamemap
