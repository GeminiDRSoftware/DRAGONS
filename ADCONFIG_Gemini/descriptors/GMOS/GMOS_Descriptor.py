from astrodata import Lookups
from astrodata import Descriptors
import re
import traceback

from astrodata.Calculator import Calculator

from datetime import datetime
from time import strptime

import GemCalcUtil

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardGMOSKeyDict import stdkeyDictGMOS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class GMOS_DescriptorCalc(GEMINI_DescriptorCalc):
    
    gmosampsGain = None
    gmosampsGainBefore20060831 = None
    gmosampsRdnoise = None
    gmosampsRdnoiseBefore20060831 = None
    
    def __init__(self):
        # self.gmosampsGain = \
        #     Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
        #                            'gmosampsGain')
        # self.gmosampsGainBefore20060831 = \
        #     Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
        #                            'gmosampsGainBefore20060831')
        
        # slightly more efficiently, we can get both at once since they are in
        # the same lookup space
        self.gmosampsGain, self.gmosampsGainBefore20060831 = \
            Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
                                   'gmosampsGain',
                                   'gmosampsGainBefore20060831')
        self.gmosampsRdnoise, self.gmosampsRdnoiseBefore20060831 = \
            Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
                                   'gmosampsRdnoise',
                                   'gmosampsRdnoiseBefore20060831')
    
    def amp_read_area(self, dataset, asList=False, **args):
        """
        Return the amp_read_area value for GMOS
        This is a composite string containing the name of the detector
        amplifier (ampname) and the readout area of that ccd (detsec).
        @param dataset: the data set
        @type dataset: AstroData
        @param asList: set to True to return a list, where the number of array
        elements equals the number of pixel data extensions in the image.
        @rtype: string or list (if asList = True)
        @return: the combined detector amplifier name and readout area
        """
        try:
            if asList:
                ret_amp_read_area = []
                if dataset.countExts('SCI') <= 1:
                    hdu = dataset.hdulist
                    ampname = \
                        hdu[1].header[stdkeyDictGMOS['key_ampname']]
                    detsec = \
                        hdu[1].header[globalStdkeyDict['key_detector_section']]
                    ret_amp_read_area.append("'%s':%s" % (ampname, detsec))
                else:
                    for ext in dataset:
                        ampname = ext.header[stdkeyDictGMOS['key_ampname']]
                        detsec = \
                            ext.header[globalStdkeyDict['key_detector_section']]
                        ret_amp_read_area.append("'%s':%s" % (ampname, detsec))
            else:
                try:
                    if dataset.countExts('SCI') <= 1:
                        hdu = dataset.hdulist
                        ampname = \
                            hdu[1].header[stdkeyDictGMOS['key_ampname']]
                        detsec = \
                            hdu[1].header[globalStdkeyDict['key_detector_section']]
                        ret_amp_read_area = ("'%s':%s" % (ampname, detsec))
                    else:
                        msg = 'Please use asList=True to obtain a list'
                        raise Exception(msg)
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return ret_amp_read_area
    
    def camera(self, dataset, **args):
        """
        Return the camera value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the camera used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            ret_camera = hdu[0].header[stdkeyDictGMOS['key_camera']]
        
        except KeyError:
            return None
        
        return str(ret_camera)
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (micrometers)
        """
        try:
            hdu = dataset.hdulist
            central_wavelength = \
                hdu[0].header[stdkeyDictGMOS['key_central_wavelength']]
            ret_central_wavelength = float(central_wavelength) / 1000.
        
        except KeyError:
            return None
        
        return float(ret_central_wavelength)
    
    def data_section(self, dataset, asList=False, **args):
        """
        Return the data_section value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @param asList: set to True to return a list, where the number of array
        elements equals the number of pixel data extensions in the image.
        @rtype: string or list (if asList = True)
        @return: the data section
        """
        try:
            if asList:
                ret_data_section = []
                if dataset.countExts('SCI') <= 1:
                    hdu = dataset.hdulist
                    data_section = \
                        hdu[1].header[globalStdkeyDict['key_data_section']]
                    ret_data_section.append(data_section)
                else:
                    for ext in dataset:
                        data_section = \
                            ext.header[globalStdkeyDict['key_data_section']]
                        ret_data_section.append(data_section)
            else:
                try:
                    if dataset.countExts('SCI') <= 1:
                        hdu = dataset.hdulist
                        ret_data_section = \
                            hdu[1].header[globalStdkeyDict['key_data_section']]
                    else:
                        msg = 'Please use asList=True to obtain a list'
                        raise Exception(msg)
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return ret_data_section
    
    def detector_section(self, dataset, asList=False, **args):
        """
        Return the detector_section value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @param asList: set to True to return a list, where the number of array
        elements equals the number of pixel data extensions in the image.
        @rtype: string or list (if asList = True)
        @return: the detector section
        """
        try:
            if asList:
                ret_detector_section = []
                if dataset.countExts('SCI') <= 1:
                    hdu = dataset.hdulist
                    detector_section = \
                        hdu[1].header[globalStdkeyDict['key_detector_section']]
                    ret_detector_section.append(detector_section)
                else:
                    for ext in dataset:
                        detector_section = \
                            ext.header[globalStdkeyDict['key_detector_section']]
                        ret_detector_section.append(detector_section)
            else:
                try:
                    if dataset.countExts('SCI') <= 1:
                        hdu = dataset.hdulist
                        ret_detector_section = \
                            hdu[1].header[globalStdkeyDict['key_detector_section']]
                    else:
                        msg = 'Please use asList=True to obtain a list'
                        raise Exception(msg)
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return ret_detector_section
    
    def detector_x_bin(self, dataset, **args):
        """
        Return the detector_x_bin value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the binning of the detector x-axis
        """
        try:
            hdu = dataset.hdulist
            # Assume ccdsum is the same in all extensions
            ccdsum = hdu[1].header[stdkeyDictGMOS['key_ccdsum']]
            
            if ccdsum != None:
                ret_detector_x_bin, detector_y_bin = ccdsum.split()
            else:
                return None
        
        except KeyError:
            return None
        
        return int(ret_detector_x_bin)
    
    def detector_y_bin(self, dataset, **args):
        """
        Return the detector_y_bin value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the binning of the detector y-axis
        """
        try:
            hdu = dataset.hdulist
            # Assume ccdsum is the same in all extensions            
            ccdsum = hdu[1].header[stdkeyDictGMOS['key_ccdsum']]
            
            if ccdsum != None:
                detector_x_bin, ret_detector_y_bin = ccdsum.split()
            else:
                return None
        
        except KeyError:
            return None
        
        return int(ret_detector_y_bin)
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the disperser value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the disperser / grating used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            disperser = hdu[0].header[stdkeyDictGMOS['key_disperser']]
            
            if pretty:
                # In the case of GMOS, pretty is stripID with additionally the
                # '+' removed from the string
                stripID = True
            
            if stripID:
                if pretty:
                    ret_disperser = \
                        GemCalcUtil.removeComponentID(disperser).strip('+')
                else:
                    ret_disperser = \
                        GemCalcUtil.removeComponentID(disperser)
            else:
                ret_disperser = disperser
        
        except KeyError:
            return None
        
        return str(ret_disperser)
    
    def dispersion(self, dataset, asList=False, **args):
        """
        Return the dispersion value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @param asList: set to True to return a list, where the number of array
        elements equals the number of pixel data extensions in the image.
        @rtype: float or list (if asList = True)
        @return: the dispersion value (angstroms/pixel)
        """
        try:
            if asList:
                ret_dispersion = []
                if dataset.countExts('SCI') <= 1:
                    hdu = dataset.hdulist
                    ret_dispersion = \
                        hdu[1].header[stdkeyDictGMOS['key_dispersion']]
                    ret_dispersion.append(dispersion)
                else:
                    for ext in dataset:
                        dispersion = \
                            ext.header[stdkeyDictGMOS['key_dispersion']]
                        ret_dispersion.append(dispersion)
            else:
                try:
                    if dataset.countExts('SCI') <= 1:
                        hdu = dataset.hdulist
                        ret_dispersion = \
                            hdu[1].header[stdkeyDictGMOS['key_dispersion']]
                    else:
                        msg = 'Please use asList=True to obtain a list'
                        raise Exception(msg)
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return ret_dispersion
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        try:
            hdu = dataset.hdulist
            exposure_time = \
                hdu[0].header[globalStdkeyDict['key_exposure_time']]
            
            # Sanity check for times when the GMOS DC is stoned
            if exposure_time > 10000. or exposure_time < 0.:
                return None
            else:
                ret_exposure_time = exposure_time
        
        except KeyError:
            return None
        
        return float(ret_exposure_time)
    
    def filter_id(self, dataset, **args):
        """
        Return the filter_id value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter ID number string
        """
        try:
            hdu = dataset.hdulist
            filtid1 = str(hdu[0].header[stdkeyDictGMOS['key_filtid1']])
            filtid2 = str(hdu[0].header[stdkeyDictGMOS['key_filtid2']])
            
            filtsid = []
            filtsid.append(filtid1)
            filtsid.append(filtid2)
            filtsid.sort()
            ret_filter_id = '&'.join(filtsid)
        
        except KeyError:
            return None
        
        return str(ret_filter_id)
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the filter_name value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the unique filter identifier string
        """
        try:
            hdu = dataset.hdulist
            filter1 = hdu[0].header[stdkeyDictGMOS['key_filter1']]
            filter2 = hdu[0].header[stdkeyDictGMOS['key_filter2']]
            
            if pretty:
                stripID = True
            
            if stripID:
                filter1 = GemCalcUtil.removeComponentID(filter1)
                filter2 = GemCalcUtil.removeComponentID(filter2)
            
            filters = []
            if not 'open' in filter1:
                filters.append(filter1)
            if not 'open' in filter2:
                filters.append(filter2)
            
            if len(filters) == 0:
                ret_filter_name = 'open'
            else:
                ret_filter_name = '&'.join(filters)
        
        except KeyError:
            return None
        
        return str(ret_filter_name)
    
    def focal_plane_mask(self, dataset, **args):
        """
        Return the focal_plane_mask value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the focal plane mask used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            focal_plane_mask = \
                hdu[0].header[stdkeyDictGMOS['key_focal_plane_mask']]
            
            if focal_plane_mask == 'None':
                ret_focal_plane_mask = 'Imaging'
            else:
                ret_focal_plane_mask = focal_plane_mask
        
        except KeyError:
            return None
        
        return str(ret_focal_plane_mask)
    
    def gain(self, dataset, asList=False, **args):
        """
        Return the gain value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @param asList: set to True to return a list, where the number of array
        elements equals the number of pixel data extensions in the image.
        @rtype: float or list (if asList = True)
        @return: the gain in electrons/ADU
        """
        try:
            hdu = dataset.hdulist
            ampinteg = hdu[0].header[stdkeyDictGMOS['key_ampinteg']]
            ut_date = hdu[0].header[globalStdkeyDict['key_ut_date']]
            obs_ut_date = datetime(*strptime(ut_date, '%Y-%m-%d')[0:6])
            old_ut_date = datetime(2006, 8, 31, 0, 0)
            
            if asList:
                ret_gain = []
                if dataset.countExts('SCI') <= 1:
                    # Descriptors must work for all AstroData Types so
                    # check if the original gain keyword exists to use for
                    # the look-up table
                    if hdu[1].header.has_key(stdkeyDictGMOS['key_gainorig']):
                        headergain = \
                            hdu[1].header[stdkeyDictGMOS['key_gainorig']]
                    else:
                        headergain = \
                            hdu[1].header[globalStdkeyDict['key_gain']]
                    
                    ampname = hdu[1].header[stdkeyDictGMOS['key_ampname']]
                    gmode = dataset.gain_mode()
                    rmode = dataset.read_speed_mode()
                    
                    gainkey = (rmode, gmode, ampname)
                    
                    try:
                        if obs_ut_date > old_ut_date:
                            gain = self.gmosampsGain[gainkey]
                        else:
                            gain = self.gmosampsGainBefore20060831[gainkey]
                    
                    except KeyError:
                        gain = None
                    
                    ret_gain.append(gain)
                else:
                    for ext in dataset:
                        # Descriptors must work for all AstroData Types so
                        # check if the original gain keyword exists to use for
                        # the look-up table
                        if ext.header.has_key(stdkeyDictGMOS['key_gainorig']):
                            headergain = \
                                ext.header[stdkeyDictGMOS['key_gainorig']]
                        else:
                            headergain = \
                                ext.header[globalStdkeyDict['key_gain']]
                        
                        ampname = ext.header[stdkeyDictGMOS['key_ampname']]
                        gmode = dataset.gain_mode()
                        rmode = dataset.read_speed_mode()
                        
                        gainkey = (rmode, gmode, ampname)
                        
                        try:
                            if obs_ut_date > old_ut_date:
                                gain = self.gmosampsGain[gainkey]
                            else:
                                gain = self.gmosampsGainBefore20060831[gainkey]
                        
                        except KeyError:
                            gain = None
                        
                        ret_gain.append(gain)
            else:
                try:
                    if dataset.countExts('SCI') <= 1:
                        # Descriptors must work for all AstroData Types so
                        # check if the original gain keyword exists to use for
                        # the look-up table
                        if hdu[1].header.has_key(stdkeyDictGMOS['key_gainorig']):
                            headergain = \
                                hdu[1].header[stdkeyDictGMOS['key_gainorig']]
                        else:
                            headergain = \
                                hdu[1].header[globalStdkeyDict['key_gain']]
                        
                        ampname = hdu[1].header[stdkeyDictGMOS['key_ampname']]
                        gmode = dataset.gain_mode()
                        rmode = dataset.read_speed_mode()
                        
                        gainkey = (rmode, gmode, ampname)
                        
                        try:
                            if obs_ut_date > old_ut_date:
                                ret_gain = self.gmosampsGain[gainkey]
                            else:
                                ret_gain = \
                                    self.gmosampsGainBefore20060831[gainkey]
                        
                        except KeyError:
                            return None
                    else:
                        msg = 'Please use asList=True to obtain a list'
                        raise Exception(msg)
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return ret_gain
    
    gmosampsGain = None
    gmosampsGainBefore20060831 = None
    
    def gain_mode(self, dataset, **args):
        """
        Return the gain_mode value for GMOS
        This is used in the gain descriptor for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the gain mode
        """
        try:
            hdu = dataset.hdulist
            # Descriptors must work for all AstroData Types so check
            # if the original gain keyword exists to use for the look-up table
            if hdu[1].header.has_key(stdkeyDictGMOS['key_gainorig']):
                headergain = \
                    hdu[1].header[stdkeyDictGMOS['key_gainorig']]
            else:
                headergain = \
                    hdu[1].header[globalStdkeyDict['key_gain']]
            
            if headergain > 3.0:
                ret_gain_mode = 'high'
            else:
                ret_gain_mode = 'low'
        
        except KeyError:
            return None
        
        return str(ret_gain_mode)
    
    def mdf_row_id(self, dataset, asList=False, **args):
        """
        Return the mdf_row_id value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @param asList: set to True to return a list, where the number of array
        elements equals the number of pixel data extensions in the image.
        @rtype: integer
        @return: the corresponding reference row in the MDF
        """
        try:
            # This descriptor function will only work on data that has been
            # reduced to a certain point (~gscut), so the descriptor function
            # should return None if the data is RAW, etc and the true value
            # when it is past the given data reduction point - TO BE DONE!
            if asList:
                ret_mdf_row_id = []
                if dataset.countExts('SCI') <= 1:
                    hdu = dataset.hdulist
                    mdf_row_id = \
                        hdu[1].header[globalStdkeyDict['key_mdf_row_id']]
                    ret_mdf_row_id.append(mdf_row_id)
                else:
                    for ext in dataset:
                        mdf_row_id = \
                            ext.header[globalStdkeyDict['key_mdf_row_id']]
                        ret_mdf_row_id.append(mdf_row_id)
            else:
                try:
                    if dataset.countExts('SCI') <= 1:
                        hdu = dataset.hdulist
                        ret_mdf_row_id = \
                            hdu[1].header[globalStdkeyDict['key_mdf_row_id']]
                    else:
                        msg = 'Please use asList=True to obtain a list'
                        raise Exception(msg)
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return ret_mdf_row_id
    
    def observation_mode(self, dataset, **args):
        """
        Return the observation_mode value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the observing mode
        """
        try:
            hdu = dataset.hdulist
            masktype = hdu[0].header[stdkeyDictGMOS['key_masktype']]
            maskname = hdu[0].header[stdkeyDictGMOS['key_maskname']]
            grating = hdu[0].header[stdkeyDictGMOS['key_disperser']]
            
            if masktype == 0:
                ret_observation_mode = 'IMAGE'
            
            elif masktype == -1:
                ret_observation_mode = 'IFU'
            
            elif masktype == 1:
                
                if re.search('arcsec', maskname) != None and \
                    re.search('NS', maskname) == None:
                    ret_observation_mode = 'LONGSLIT'
                else:
                    ret_observation_mode = 'MOS'
            else:
                # if obsmode cannot be determined, set it equal to IMAGE
                # instead of crashing
                ret_observation_mode = 'IMAGE'
            
            # mask or IFU cannot be used without grating
            if grating == 'MIRROR' and masktype != 0:
                ret_observation_mode == 'IMAGE' 
        
        except KeyError:
            return None
        
        return str(ret_observation_mode)
    
    def pixel_scale(self, dataset, **args):
        """
        Return the pixel_scale value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the pixel scale (arcsec/pixel)
        """
        try:
            hdu = dataset.hdulist
            instrument = \
                hdu[0].header[globalStdkeyDict['key_instrument']]
            detector_y_bin = dataset.detector_y_bin()
            
            if instrument == 'GMOS-N':
                scale = 0.0727
            if instrument == 'GMOS-S':
                scale = 0.073
            
            if detector_y_bin != None:
                ret_pixel_scale = float(detector_y_bin) * scale
            else:
                ret_pixel_scale = scale
        
        except KeyError:
            return None
        
        return float(ret_pixel_scale)
    
    def read_noise(self, dataset, asList=False, **args):
        """
        Return the read_noise value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @param asList: set to True to return a list, where the number of array
        elements equals the number of pixel data extensions in the image.
        @rtype: float or list (if asList = True)
        @return: the estimated readout noise values (electrons)
        """
        try:
            hdu = dataset.hdulist
            ampinteg = hdu[0].header[stdkeyDictGMOS['key_ampinteg']]
            ut_date = hdu[0].header[globalStdkeyDict['key_ut_date']]
            obs_ut_date = datetime(*strptime(ut_date, '%Y-%m-%d')[0:6])
            old_ut_date = datetime(2006, 8, 31, 0, 0)
            
            if asList:
                ret_read_noise = []
                if dataset.countExts('SCI') <= 1:
                    # Descriptors must work for all AstroData Types so
                    # check if the original gain keyword exists to use for
                    # the look-up table
                    if hdu[1].header.has_key(stdkeyDictGMOS['key_gainorig']):
                        headergain = \
                            hdu[1].header[stdkeyDictGMOS['key_gainorig']]
                    else:
                        headergain = \
                            hdu[1].header[globalStdkeyDict['key_gain']]
                    
                    ampname = hdu[1].header[stdkeyDictGMOS['key_ampname']]
                    gmode = dataset.gain_mode()
                    rmode = dataset.read_speed_mode()
                    
                    read_noise_key = (rmode, gmode, ampname)
                    
                    try:
                        if obs_ut_date > old_ut_date:
                            ret_read_noise = \
                                self.gmosampsRdnoise[read_noise_key]
                        else:
                            ret_read_noise = \
                                self.gmosampsRdnoiseBefore20060831[read_noise_key]
                    except KeyError:
                        return None
                    
                    ret_read_noise.append(read_noise)

                else:
                    for ext in dataset:
                        # Descriptors must work for all AstroData Types so
                        # check if the original gain keyword exists to use for
                        # the look-up table
                        if ext.header.has_key(stdkeyDictGMOS['key_gainorig']):
                            headergain = \
                                ext.header[stdkeyDictGMOS['key_gainorig']]
                        else:
                            headergain = \
                                ext.header[globalStdkeyDict['key_gain']]
                        
                        ampname = ext.header[stdkeyDictGMOS['key_ampname']]
                        gmode = dataset.gain_mode()
                        rmode = dataset.read_speed_mode()
                        
                        read_noise_key = (rmode, gmode, ampname)
                        
                        try:
                            if obs_ut_date > old_ut_date:
                                read_noise = \
                                    self.gmosampsRdnoise[read_noise_key]
                            else:
                                read_noise = \
                                    self.gmosampsRdnoiseBefore20060831[read_noise_key]
                        
                        except KeyError:
                            read_noise = None
                        
                        ret_read_noise.append(read_noise)
            else:
                try:
                    if dataset.countExts('SCI') <= 1:
                        # Descriptors must work for all AstroData Types so
                        # check if the original gain keyword exists to use for
                        # the look-up table
                        if hdu[1].header.has_key(stdkeyDictGMOS['key_gainorig']):
                            headergain = \
                                hdu[1].header[stdkeyDictGMOS['key_gainorig']]
                        else:
                            headergain = \
                                hdu[1].header[globalStdkeyDict['key_gain']]
                        
                        ampname = hdu[1].header[stdkeyDictGMOS['key_ampname']]
                        gmode = dataset.gain_mode()
                        rmode = dataset.read_speed_mode()
                        
                        read_noise_key = (rmode, gmode, ampname)
                        
                        try:
                            if obs_ut_date > old_ut_date:
                                ret_read_noise = self.gmosampsRdnoise[read_noise_key]
                            else:
                                ret_read_noise = \
                                    self.gmosampsRdnoiseBefore20060831[read_noise_key]
                        except KeyError:
                            return None
                    else:
                        msg = 'Please use asList=True to obtain a list'
                        raise Exception(msg)
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return ret_read_noise
    
    gmosampsRdnoise = None
    gmosampsRdnoiseBefore20060831 = None
    
    def read_speed_mode(self, dataset, **args):
        """
        Return the read_speed_mode value for GMOS
        This is used in the gain descriptor for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the read speed mode
        """
        try:
            hdu = dataset.hdulist
            ampinteg = hdu[0].header[stdkeyDictGMOS['key_ampinteg']]
            
            if ampinteg == 1000:
                ret_read_speed_mode = 'fast'
            else:
                ret_read_speed_mode = 'slow'
        
        except KeyError:
            return None
        
        return str(ret_read_speed_mode)
    
    def saturation_level(self, dataset, **args):
        """
        Return the saturation_level value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the saturation level in the raw images (ADU)
        """
        ret_saturation_level = 65000
        
        return int(ret_saturation_level)
    
    def wavelength_reference_pixel(self, dataset, asList=False, **args):
        """
        Return the wavelength_reference_pixel value for GMOS
        @param dataset: the data set
        @type dataset: AstroData
        @param asList: set to True to return a list, where the number of array
        elements equals the number of pixel data extensions in the image.
        @rtype: float or list (if asList = True)
        @return: the reference pixel of the central wavelength
        """
        try:
            if asList:
                ret_wavelength_reference_pixel = []
                if dataset.countExts('SCI') <= 1:
                    hdu = dataset.hdulist
                    wavelength_reference_pixel = \
                        hdu[1].header[stdkeyDictGMOS['key_wavelength_reference_pixel']]
                    ret_wavelength_reference_pixel.append(wavelength_reference_pixel)
                else:
                    for ext in dataset:
                        wavelength_reference_pixel = \
                            ext.header[stdkeyDictGMOS['key_wavelength_reference_pixel']]
                        ret_wavelength_reference_pixel.append(wavelength_reference_pixel)
            else:
                try:
                    if dataset.countExts('SCI') <= 1:
                        hdu = dataset.hdulist
                        ret_wavelength_reference_pixel = \
                            hdu[1].header[stdkeyDictGMOS['key_wavelength_reference_pixel']]
                    else:
                        msg = 'Please use asList=True to obtain a list'
                        raise Exception(msg)
                
                except Exception:
                    print traceback.format_exc()
        
        except KeyError:
            return None
        
        return ret_wavelength_reference_pixel
