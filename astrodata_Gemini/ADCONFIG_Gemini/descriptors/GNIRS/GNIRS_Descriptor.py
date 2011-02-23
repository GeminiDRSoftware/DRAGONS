from astrodata import Lookups
from astrodata import Descriptors
import math
import re

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardGNIRSKeyDict import stdkeyDictGNIRS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class GNIRS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dict with the local dict of this descriptor class
    globalStdkeyDict.update(stdkeyDictGNIRS)
    
    gnirsArrayDict = None
    gnirsConfigDict = None
    
    def __init__(self):
        self.gnirsArrayDict = \
            Lookups.getLookupTable('Gemini/GNIRS/GNIRSArrayDict',
                                   'gnirsArrayDict')
        self.gnirsConfigDict = \
            Lookups.getLookupTable('Gemini/GNIRS/GNIRSConfigDict',
                                   'gnirsConfigDict')
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        hdu = dataset.hdulist
        central_wavelength = \
            hdu[0].header[stdkeyDictGNIRS['key_central_wavelength']]
        
        ret_central_wavelength = float(central_wavelength)
        
        return ret_central_wavelength
    
    def decker(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the decker value for GNIRS
        In GNIRS, the decker is used to basically mask off the ends of the
        slit to create the short slits used in the cross dispersed modes.
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID from the
        returned decker names
        @param pretty: set to True to return a human meaningful decker name
        @rtype: string
        @return: the decker postition used to acquire the data
        """
        hdu = dataset.hdulist
        decker = hdu[0].header[globalStdkeyDict['key_decker']]
        
        if pretty:
            stripID=True
        
        if stripID:
            decker = GemCalcUtil.removeComponentID(decker)
        
        return decker
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the disperser value for GNIRS
        Note that GNIRS contains two dispersers - the grating and the prism.
        This descriptor will combine the two with '&'. Sometimes the 'prism'
        is a mirror, in which case we don't list it in the human readable
        pretty string.
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to strip the component ID from the
        returned disperser names
        @param pretty: set to True to return a human meaningful disperser name
        @rtype: string
        @return: the dispersers used to acquire the data
        """
        if pretty:
            stripID=True
        
        grating = self.grating(dataset=dataset, stripID=stripID, pretty=pretty)
        prism = self.prism(dataset=dataset, stripID=stripID, pretty=pretty)
        
        if (pretty and prism[0:3]=='MIR'):
            disperser = grating
        else:
            disperser = grating + '&' + prism
        
        return disperser
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        hdu = dataset.hdulist
        exposure_time = hdu[0].header[globalStdkeyDict['key_exposure_time']]
        coadds = dataset.coadds()
        
        if dataset.isType('GNIRS_RAW') == True and coadds != 1:
            ret_exposure_time = float(exposure_time * coadds)
        else:
            ret_exposure_time = float(exposure_time)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the filter_name value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID from the
        returned filter name
        @param pretty: set to True to return a meaningful filter name
        @rtype: string
        @return: the unique filter identifier string
        """
        # No specific pretty names, just use stripID
        if pretty:
            stripID=True
        
        hdu = dataset.hdulist
        filter1 = hdu[0].header[stdkeyDictGNIRS['key_filter1']]
        filter2 = hdu[0].header[stdkeyDictGNIRS['key_filter2']]
        
        if stripID:
            filter1 = GemCalcUtil.removeComponentID(filter1)
            filter2 = GemCalcUtil.removeComponentID(filter2)
        
        # Create list of filter values
        filters = [filter1,filter2]
        
        # reject 'Open'
        filters2 = []
        for filt in filters:
            if 'Open' in filt:
                pass
            else:
                filters2.append(filt)
        
        filters = filters2
        
        if len(filters) == 0:
            ret_filter_name = 'open'
        else:
            ret_filter_name = str('&'.join(filters))
        
        if 'Dark' in filters:
            ret_filter_name = 'blank'
        
        return ret_filter_name
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the focal_plane_mask value for GNIRS
        Note that in GNIRS, the focal plane mask is the combination of the slit
        mechanism and the decker mechanism. 
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component IDs from the
        returned focal_plane_mask names
        @param pretty: set to True to return a human meaningful
        focal_plane_mask name
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        if pretty:
            stripID=True
        
        slit = self.slit(dataset=dataset, stripID=stripID, pretty=pretty)
        decker = self.decker(dataset=dataset, stripID=stripID, pretty=pretty)
        
        fpmask = slit + '&' + decker
        
        if pretty:
            # For pretty output, disregard the decker if it's in long slit mode
            if decker.count('Long'):
                fpmask = slit
            # For pretty output, simply append XD to the slit name if the
            # decker is in XD
            if decker.count('XD'):
                fpmask = slit + 'XD'
        
        return fpmask
    
    def gain(self, dataset, **args):
        """
        Return the gain value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the gain (electrons/ADU)
        """
        hdu = dataset.hdulist
        headerbias = hdu[0].header[stdkeyDictGNIRS['key_bias']]
        
        biasvalues = self.gnirsArrayDict.keys()
        
        for bias in biasvalues:
            if abs(float(bias) - abs(headerbias)) < 0.1:
                array = self.gnirsArrayDict[bias]
            else:
                array = None
        
        if array != None:
            ret_gain = float(array[2])
        else:
            ret_gain = None
        
        return ret_gain
    
    gnirsArrayDict = None
    
    def grating(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the grating value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to strip the component ID from the
        returned grating name
        @param pretty: set to True to return a human meaningful grating name.
        In this case, this is the same as stripID
        @rtype: string
        @return: the grating used to acquire the data

        Note. A CC software change approx July 2010 changed the grating names
        to also include the camera, eg 32/mmSB_G5533 indicates the 32/mm
        grating with the Short Blue camera. This is unhelpful as if we wanted
        to know the camera, we'd call the camera descriptor. Thus, this
        descriptor function repairs the header values to only list the grating.
        """
        hdu = dataset.hdulist
        string = hdu[0].header[globalStdkeyDict['key_grating']]
        
        # The format of the grating string is currently (2011) nnn/mmCAM_Gnnnn
        # nnn is a 2 or 3 digit number (lines per mm)
        # /mm is literally '/mm'
        # CAM is the camera: {L|S}{B|R}[{L|S}[X]}
        # _G is literally '_G'
        # nnnn is the 4 digit component ID.
        
        cre = re.compile('([\d/m]+)([A-Z]*)(_G)(\d+)')
        m = cre.match(string)
        
        grating = string
        if m:
            parts = m.groups()
            grating = parts[0] + parts[2] + parts[3]
        
        if (stripID or pretty):
            grating = str(GemCalcUtil.removeComponentID(grating))
        
        return grating
    
    def non_linear_level(self, dataset, **args):
        """
        Return the non_linear_level value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the non-linear level in the raw images (ADU)
        """
        # non_linear_level depends on whether data has been corrected for
        # non-linearity ... need to check this ...
        hdu = dataset.hdulist
        headerbias = hdu[0].header[stdkeyDictGNIRS['key_bias']]
        coadds = dataset.coadds()
        
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
            saturation = int(well * coadds)
            ret_non_linear_level = int(saturation * linearlimit)
        else:
            ret_non_linear_level = None
        
        return ret_non_linear_level
    
    gnirsArrayDict = None
    
    def pixel_scale(self, dataset, **args):
        """
        Return the pixel_scale value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the pixel scale (arcsec/pixel)
        """
        hdu = dataset.hdulist
        prism = hdu[0].header[globalStdkeyDict['key_prism']]
        decker = hdu[0].header[globalStdkeyDict['key_decker']]
        disperser = hdu[0].header[globalStdkeyDict['key_grating']]
        camera = dataset.camera()
        
        pixel_scale_key = (prism, decker, disperser, camera)
        
        array = self.gnirsConfigDict[pixel_scale_key]
        
        ret_pixel_scale = float(array[2])
        
        return ret_pixel_scale
    
    gnirsConfigDict = None
    
    def prism(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the prism value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to strip the component ID from the
        returned prism name
        @param pretty: set to True to return a human meaningful prism name. In
        this case, this is the same as stripID
        @rtype: string
        @return: the prism used to acquire the data

        Note. A CC software change approx July 2010 changed the prism names to
        also include the camera, eg 32/mmSB_G5533 indicates the 32/mm grating
        with the Short Blue camera. This is unhelpful as if we wanted to know
        the camera, we'd call the camera descriptor. Thus, this descriptor
        function repairs the header values to only list the prism.
        """
        hdu = dataset.hdulist
        string = hdu[0].header[globalStdkeyDict['key_prism']]
        
        # The format of the prism string is currently (2011) [CAM+]prism_Gnnnn
        # CAM is the camera: {L|S}{B|R}[{L|S}[X]}
        # + is a literal '+'
        # prism is the actual prism name
        # nnnn is the 4 digit component ID.
        
        cre = re.compile('([LBSR]*\+)*([A-Z]*)(_G)(\d+)')
        m = cre.match(string)
        
        prism = string
        if m:
            parts = m.groups()
            prism = parts[1] + parts[2] + parts[3]
        
        if (stripID or pretty):
            prism = str(GemCalcUtil.removeComponentID(prism))
        
        return prism
    
    def read_mode(self, dataset, **args):
        """
        Return the read_mode value for GNIRS
        This is either 'Very Bright Objects', 'Bright Objects',
        'Faint Objects' or 'Very Faint Objects' in the OT. Returns 'Invalid'
        if the headers don't make sense wrt these defined modes
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the read mode used to acquire the data
        """
        hdu = dataset.hdulist
        lnrs = hdu[0].header[stdkeyDictGNIRS['key_lnrs']]
        ndavgs = hdu[0].header[stdkeyDictGNIRS['key_ndavgs']]
        
        read_mode = 'Invalid'
        
        if lnrs == 32 and ndavgs == 16:
            read_mode = 'Very Faint Objects'
        
        if lnrs == 16 and ndavgs == 16:
            read_mode = 'Faint Objects'
        
        if lnrs == 1 and ndavgs == 16:
            read_mode = 'Bright Objects'
        
        if lnrs == 1 and ndavgs == 1:
            read_mode = 'Very Bright Objects'
        
        return read_mode
    
    def read_noise(self, dataset, **args):
        """
        Return the read_noise value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the estimated readout noise (electrons)
        """
        hdu = dataset.hdulist
        headerbias = hdu[0].header[stdkeyDictGNIRS['key_bias']]
        lnrs = hdu[0].header[stdkeyDictGNIRS['key_lnrs']]
        ndavgs = hdu[0].header[stdkeyDictGNIRS['key_ndavgs']]
        coadds = dataset.coadds()
        
        biasvalues = self.gnirsArrayDict.keys()
        for bias in biasvalues:
            if abs(float(bias) - abs(headerbias)) < 0.1:
                array = self.gnirsArrayDict[bias]
            else:
                array = None
        
        if array != None:
            read_noise = float(array[1])
            ret_read_noise = float((read_noise * math.sqrt(coadds)) \
                / (math.sqrt(lnrs) * math.sqrt(ndavgs)))
        else:
            ret_read_noise = None
        
        return ret_read_noise
    
    gnirsArrayDict = None
    
    def saturation_level(self, dataset, **args):
        """
        Return the saturation_level value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the saturation level in the raw images (ADU)
        """
        hdu = dataset.hdulist
        headerbias = hdu[0].header[stdkeyDictGNIRS['key_bias']]
        coadds = dataset.coadds()
        
        biasvalues = self.gnirsArrayDict.keys()
        for bias in biasvalues:
            if abs(float(bias) - abs(headerbias)) < 0.1:
                array = self.gnirsArrayDict[bias]
            else:
                array = None
        
        if array != None:
            well = array[3]
            ret_saturation_level = int(well * coadds)
        else:
            ret_saturation_level = None
        
        return ret_saturation_level
    
    gnirsArrayDict = None
    
    def slit(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the slit value for GNIRS
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID
        @param pretty: set to True to return a human readable
        @rtype: string
        @return: the slit used to acquire the data

        Note that in GNIRS all the slits are machined into one physical piece
        of metal, which is on a slide - the mechanism simply slides the slide
        along to put the right slit in the beam. Thus all the slits have the
        same componenet ID as they're they same physical compononet.
        """
        hdu = dataset.hdulist
        slit = hdu[0].header[globalStdkeyDict['key_slit']]
        
        if pretty:
            stripID=True
        
        if stripID:
            slit = GemCalcUtil.removeComponentID(slit)
        
        return slit
    
    def well_depth_setting(self, dataset, **args):
        """
        Return the well_depth_setting value for GNIRS
        This is either 'Shallow' or 'Deep' in the OT. Returns 'Invalid' if the
        headers don't make sense wrt these defined modes
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the well depth mode used to acquire the data
        """
        hdu = dataset.hdulist
        biasvoltage = hdu[0].header[stdkeyDictGNIRS['key_bias']]
        
        well_depth_setting = 'Invalid'
        
        if abs(biasvoltage + 0.3) < 0.1:
            well_depth_setting = 'Deep'
        
        if abs(biasvoltage + 0.6) < 0.1:
            well_depth_setting = 'Shallow'
        
        return well_depth_setting
