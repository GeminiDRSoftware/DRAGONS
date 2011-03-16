from astrodata import Lookups
from astrodata import Descriptors
from astrodata import Errors

from astrodata.Calculator import Calculator

from datetime import datetime
from time import strptime

import GemCalcUtil

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardGMOSKeyDict import stdkeyDictGMOS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class GMOS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    globalStdkeyDict.update(stdkeyDictGMOS)
    
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
        
        # Slightly more efficient: we can get both at once since they are in
        # the same lookup space
        self.gmosampsGain, self.gmosampsGainBefore20060831 = \
            Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
                                   'gmosampsGain',
                                   'gmosampsGainBefore20060831')
        self.gmosampsRdnoise, self.gmosampsRdnoiseBefore20060831 = \
            Lookups.getLookupTable('Gemini/GMOS/GMOSAmpTables',
                                   'gmosampsRdnoise',
                                   'gmosampsRdnoiseBefore20060831')
    
    def amp_read_area(self, dataset, asDict=True, **args):
        if asDict:
            ret_amp_read_area = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Get the name of the detector amplifier (ampname) from the
                # header of each pixel data extension. The ampname keyword is
                # defined in the local key dictionary (stdkeyDictGMOS) but is
                # read from the updated global key dictionary
                # (globalStdkeyDict)
                ampname = ext.header[globalStdkeyDict['key_ampname']]
                # Get the readout area of the CCD (detsec) using the
                # appropriate descriptor
                detsec = ext.detector_section(pretty=True, asDict=False)
                # Create the composite amp_read_area string
                amp_read_area = "'%s':%s" % (ampname, detsec)

                # Return a dictionary with the composite amp_read_area string
                # as the value
                ret_amp_read_area.update({(ext.extname(), \
                    ext.extver()):str(amp_read_area)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Get the name of the detector amplifier (ampname) from the
                # header of the single pixel data extension. The ampname
                # keyword is defined in the local key dictionary
                # (stdkeyDictGMOS) but is read from the updated global key
                # dictionary (globalStdkeyDict)
                hdu = dataset.hdulist
                ampname = hdu[1].header[globalStdkeyDict['key_ampname']]
                # Get the readout area of the CCD (detsec) using the
                # appropriate descriptor
                detsec = hdu[1].detector_section(pretty=True, asDict=False)
                # Return the composite amp_read_area string
                ret_amp_read_area = "'%s':%s" % (ampname, detsec)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_amp_read_area
    
    def central_wavelength(self, dataset, asMicrometers=False, \
        asNanometers=False, asAngstroms=False, asDict=False, **args):        
        # Currently for GMOS data, the central wavelength is recorded in
        # nanometers
        input_units = 'nanometers'
        
        # Determine the output units to use
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # central wavelength in these units
            if asMicrometers:
                output_units = 'micrometers'
            if asNanometers:
                output_units = 'nanometers'
            if asAngstroms:
                output_units = 'angstroms'
        else:
            # Either none of the unit arguments were set to True or more than
            # one of the unit arguments was set to True. In either case,
            # return the central wavelength in the default units of meters
            output_units = 'meters'
        
        if asDict:
            # This is used when obtaining the central wavelength from processed
            # data (when the keyword will be in the pixel data extensions)
            return 'asDict for central_wavelength not yet implemented'
        else:
            # Get the central wavelength value from the header of the PHU. The
            # central wavelength keyword is defined in the local key
            # dictionary (stdkeyDictGMOS) but is read from the updated global
            # key dictionary (globalStdkeyDict)
            hdu = dataset.hdulist
            raw_central_wavelength = \
                hdu[0].header[globalStdkeyDict['key_central_wavelength']]
            # Use the utilities function convert_units to convert the central
            # wavelength value from the input units to the output units
            ret_central_wavelength = \
                GemCalcUtil.convert_units(input_units=input_units, \
                input_value=raw_central_wavelength, output_units=output_units)
        
        return ret_central_wavelength
    
    def detector_x_bin(self, dataset, asDict=True, **args):
        if asDict:
            ret_detector_x_bin = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Get the ccdsum value from the header of each pixel data
                # extension. The ccdsum keyword is defined in the local key
                # dictionary (stdkeyDictGMOS) but is read from the updated
                # global key dictionary (globalStdkeyDict)
                ccdsum = ext.header[globalStdkeyDict['key_ccdsum']]
                detector_x_bin, detector_y_bin = ccdsum.split()
                # Return a dictionary with the binning of the x-axis integer
                # as the value
                ret_detector_x_bin.update({(ext.extname(), \
                    ext.extver()):int(detector_x_bin)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Get the ccdsum value from the header of the single pixel data
                # extension. The ccdsum keyword is defined in the local key
                # dictionary (stdkeyDictGMOS) but is read from the updated
                # global key dictionary (globalStdkeyDict)
                hdu = dataset.hdulist
                ccdsum = hdu[1].header[globalStdkeyDict['key_ccdsum']]
                detector_x_bin, detector_y_bin = ccdsum.split()
                # Return the binning of the x-axis integer
                ret_detector_x_bin = int(detector_x_bin)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_detector_x_bin
    
    def detector_y_bin(self, dataset, asDict=True, **args):
        if asDict:
            ret_detector_y_bin = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Get the ccdsum value from the header of each pixel data
                # extension. The ccdsum keyword is defined in the local key
                # dictionary (stdkeyDictGMOS) but is read from the updated
                # global key dictionary (globalStdkeyDict)
                ccdsum = ext.header[globalStdkeyDict['key_ccdsum']]
                detector_x_bin, detector_y_bin = ccdsum.split()
                # Return a dictionary with the binning of the y-axis integer
                # as the value
                ret_detector_y_bin.update({(ext.extname(), \
                    ext.extver()):int(detector_y_bin)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Get the ccdsum value from the header of the single pixel data
                # extension. The ccdsum keyword is defined in the local key
                # dictionary (stdkeyDictGMOS) but is read from the updated
                # global key dictionary (globalStdkeyDict)
                hdu = dataset.hdulist
                ccdsum = hdu[1].header[globalStdkeyDict['key_ccdsum']]
                detector_x_bin, detector_y_bin = ccdsum.split()
                # Return the binning of the y-axis integer
                ret_detector_y_bin = int(detector_y_bin)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_detector_y_bin
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        # Get the disperser value from the header of the PHU. The disperser
        # keyword is defined in the local key dictionary (stdkeyDictGMOS) but
        # is read from the updated global key dictionary (globalStdkeyDict)
        hdu = dataset.hdulist
        disperser = hdu[0].header[globalStdkeyDict['key_disperser']]
        
        if pretty:
            # If pretty=True, use stripID then additionally remove the
            # trailing '+' from the string
            stripID = True
        
        if stripID:
            if pretty:
                # Return the stripped and pretty disperser string
                ret_disperser = \
                    GemCalcUtil.removeComponentID(disperser).strip('+')
            else:
                # Return the stripped disperser string
                ret_disperser = GemCalcUtil.removeComponentID(disperser)
        else:
            # Return the disperser string
            ret_disperser = str(disperser)
        
        return ret_disperser
    
    def dispersion(self, dataset, asMicrometers=False, asNanometers=False, \
        asAngstroms=False, asDict=True, **args):
        
        # I have no idea what the units the dispersion is recorded in, so
        # defaulting to meters for now ...
        input_units = 'meters'
        
        # Determine the output units to use
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # dispersion in these units
            if asMicrometers:
                output_units = 'micrometers'
            if asNanometers:
                output_units = 'nanometers'
            if asAngstroms:
                output_units = 'angstroms'
        else:
            # Either none of the unit arguments were set to True or more than
            # one of the unit arguments was set to True. In either case,
            # return the dispersion in the default units of meters
            output_units = 'meters'
        
        if asDict:
            ret_dispersion = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Get the dispersion value from the header of each pixel data
                # extension. The dispersion keyword is defined in the local
                # key dictionary (stdkeyDictGMOS) but is read from the updated
                # global key dictionary (globalStdkeyDict)
                raw_dispersion = \
                    ext.header[globalStdkeyDict['key_dispersion']]
                # Use the utilities function convert_units to convert the
                # dispersion wavelength value from the input units to the
                # output units                
                dispersion = \
                    GemCalcUtil.convert_units(input_units=input_units, \
                    input_value=raw_dispersion, output_units=output_units)
                # Return a dictionary with the dispersion float as the value
                ret_dispersion.update({(ext.extname(), \
                    ext.extver()):float(dispersion)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Get the dispersion value from the header of the single pixel
                # data extension. The dispersion keyword is defined in the
                # local key dictionary (stdkeyDictGMOS) but is read from the
                # updated global key dictionary (globalStdkeyDict)
                hdu = dataset.hdulist
                raw_dispersion = \
                    hdu[1].header[globalStdkeyDict['key_dispersion']]
                # Use the utilities function convert_units to convert the
                # dispersion wavelength value from the input units to the
                # output units                
                dispersion = \
                    GemCalcUtil.convert_units(input_units=input_units, \
                    input_value=raw_dispersion, output_units=output_units)
                # Return the dispersion float
                ret_dispersion = float(dispersion)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_dispersion
    
    def exposure_time(self, dataset, **args):
        # Get the exposure time from the header of the PHU
        hdu = dataset.hdulist
        exposure_time = hdu[0].header[globalStdkeyDict['key_exposure_time']]
        
        # Sanity check for times when the GMOS DC is stoned
        if exposure_time > 10000. or exposure_time < 0.:
            raise Errors.InvalidValueError()
        else:
            # Return the exposure time float
            ret_exposure_time = float(exposure_time)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Get the two filter name values from the header of the PHU. The two
        # filter name keywords are defined in the local key dictionary
        # (stdkeyDictGMOS) but are read from the updated global key dictionary
        # (globalStdkeyDict)
        hdu = dataset.hdulist
        filter1 = hdu[0].header[globalStdkeyDict['key_filter1']]
        filter2 = hdu[0].header[globalStdkeyDict['key_filter2']]
        
        if pretty:
            stripID = True
        
        if stripID:
            # Strip the component ID from the two filter name values
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
            # Return a unique, sorted filter name identifier string with an
            # ampersand separating each filter name
            filters.sort
            ret_filter_name = str('&'.join(filters))
        
        return ret_filter_name
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        # Get the focal plane mask value from the header of the PHU. The focal
        # plane mask keyword is defined in the local key dictionary
        # (stdkeyDictGMOS) but is read from the updated global key dictionary
        # (globalStdkeyDict)
        hdu = dataset.hdulist
        focal_plane_mask = \
            hdu[0].header[globalStdkeyDict['key_focal_plane_mask']]
        
        if focal_plane_mask == 'None':
            ret_focal_plane_mask = 'Imaging'
        else:
            # Return the focal plane mask string
            ret_focal_plane_mask = str(focal_plane_mask)
        
        return ret_focal_plane_mask
    
    def gain(self, dataset, asDict=True, **args):
        # Get the amplifier integration time (ampinteg) and the UT date from
        # the header of the PHU. The ampinteg keyword is defined in the local
        # key dictionary (stdkeyDictGMOS) but is read from the updated global
        # key dictionary (globalStdkeyDict)
        hdu = dataset.hdulist
        ampinteg = hdu[0].header[globalStdkeyDict['key_ampinteg']]
        # Get the UT date using the appropriate descriptor
        ut_date = dataset.ut_date(asString=True)
        obs_ut_date = datetime(*strptime(ut_date, '%Y-%m-%d')[0:6])
        old_ut_date = datetime(2006, 8, 31, 0, 0)
        
        if asDict:
            ret_gain = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Check if the original gain (gainorig) keyword exists in the
                # header of the pixel data extension. The gainorig keyword is
                # defined in the local key dictionary (stdkeyDictGMOS) but is
                # read from the updated global key dictionary
                # (globalStdkeyDict)
                if ext.header.has_key(globalStdkeyDict['key_gainorig']):
                    headergain = ext.header[globalStdkeyDict['key_gainorig']]
                else:
                    headergain = ext.header[globalStdkeyDict['key_gain']]
                
                # Get the name of the detector amplifier (ampname) from the
                # header of each pixel data extension. The ampname keyword is
                # defined in the local key dictionary (stdkeyDictGMOS) but is
                # read from the updated global key dictionary
                # (globalStdkeyDict)
                ampname = ext.header[globalStdkeyDict['key_ampname']]
                # Get the gain setting and read speed setting values using the
                # appropriate descriptors
                gain_setting = dataset.gain_setting()
                read_speed_setting = dataset.read_speed_setting()
                
                gainkey = (read_speed_setting, gain_setting, ampname)
                
                if obs_ut_date > old_ut_date:
                    gain = self.gmosampsGain[gainkey]
                else:
                    gain = self.gmosampsGainBefore20060831[gainkey]
                
                # Return a dictionary with the gain float as the value
                ret_gain.update({(ext.extname(), ext.extver()):float(gain)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Check if the original gain (gainorig) keyword exists in the
                # header of the pixel data extension. The gainorig keyword is
                # defined in the local key dictionary (stdkeyDictGMOS) but is
                # read from the updated global key dictionary
                # (globalStdkeyDict)
                hdu = dataset.hdulist
                if hdu[1].header.has_key(globalStdkeyDict['key_gainorig']):
                    headergain = \
                        hdu[1].header[globalStdkeyDict['key_gainorig']]
                else:
                    headergain = hdu[1].header[globalStdkeyDict['key_gain']]
                
                # Get the name of the detector amplifier (ampname) from the
                # header of the single pixel data extension. The ampname
                # keyword is defined in the local key dictionary
                # (stdkeyDictGMOS) but is read from the updated global key
                # dictionary (globalStdkeyDict)
                ampname = hdu[1].header[globalStdkeyDict['key_ampname']]
                # Get the gain setting and read speed setting values using the
                # appropriate descriptors
                gain_setting = dataset.gain_setting()
                read_speed_setting = dataset.read_speed_setting()
                
                gainkey = (read_speed_setting, gain_setting, ampname)
                
                if obs_ut_date > old_ut_date:
                    gain = self.gmosampsGain[gainkey]
                else:
                    gain = self.gmosampsGainBefore20060831[gainkey]
                
                # Return the gain float
                ret_gain = float(gain)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_gain
    
    gmosampsGain = None
    gmosampsGainBefore20060831 = None
    
    def gain_setting(self, dataset, **args):
        # Check if the original gain (gainorig) keyword exists in the header
        # of the pixel data extension. The gainorig keyword is defined in the
        # local key dictionary (stdkeyDictGMOS) but is read from the updated
        # global key dictionary (globalStdkeyDict)
        hdu = dataset.hdulist
        if hdu[1].header.has_key(globalStdkeyDict['key_gainorig']):
            headergain = hdu[1].header[globalStdkeyDict['key_gainorig']]
        else:
            headergain = hdu[1].header[globalStdkeyDict['key_gain']]
        
        if headergain > 3.0:
            ret_gain_setting = 'high'
        else:
            ret_gain_setting = 'low'
        
        return ret_gain_setting
    
    def mdf_row_id(self, dataset, asDict=True, **args):
        # This descriptor function will only work on data that has been
        # reduced to a certain point (~gscut), so the descriptor function
        # should return None if the data is RAW, etc and the true value
        # when it is past the given data reduction point - TO BE DONE!
        
        # Check if the images is prepared, and not an image
        if 'IMAGE' not in dataset.types and 'PREPARED' in dataset.types:
            if asDict:
                ret_mdf_row_id = {}
                # Loop over the science extensions
                for ext in dataset['SCI']:
                    # Get the MDF row ID from the header of each pixel data
                    # extension
                    mdf_row_id = \
                        ext.header[globalStdkeyDict['key_mdf_row_id']]
                    # Return a dictionary with the MDF row ID integer as the
                    # value
                    ret_mdf_row_id.update({(ext.extname(), \
                        ext.extver()):int(mdf_row_id)})
            else:
                # Check to see whether the dataset has a single extension and
                # if it does, return a single value
                if dataset.countExts('SCI') <= 1:
                    # Get the MDF row ID from the header of each pixel data
                    # extension                    
                    hdu = dataset.hdulist
                    mdf_row_id = \
                        hdu[1].header[globalStdkeyDict['key_mdf_row_id']]
                    # Return the MDF row ID integer
                    ret_mdf_row_id = int(mdf_row_id)
                else:
                    raise Errors.DescriptorDictError()
        else:
            raise Errors.DescriptorTypeError()
        
        return ret_mdf_row_id

    def nod_count(self, dataset, **args):
        # The number of nod and shuffle cycles can only be obtained from nod
        # and shuffle data
        if 'GMOS_NODANDSHUFFLE' in dataset.types:
            # Get the number of nod and shuffle cycles from the header of the
            # PHU
            hdu = dataset.hdulist
            ret_nod_count = hdu[0].header[globalStdkeyDict['key_nod_count']]
        else:
            raise Errors.DescriptorTypeError()
        
        return ret_nod_count
    
    def nod_pixels(self, dataset, **args):
        # The number of pixel rows the charge is shuffled by can only be
        # obtained from nod and shuffle data
        if 'GMOS_NODANDSHUFFLE' in dataset.types:
            # Get the number of pixel rows the charge is shuffled by from the
            # header of the PHU
            hdu = dataset.hdulist
            ret_nod_pixels = hdu[0].header[globalStdkeyDict['key_nod_pixels']]
        else:
            raise Errors.DescriptorTypeError()

        return ret_nod_pixels
    
    def non_linear_level(self, dataset, **args):
        # Set the non linear level equal to the saturation level for GMOS
        ret_non_linear_level = dataset.saturation_level()
        
        return ret_non_linear_level
    
    def pixel_scale(self, dataset, **args):
        # Should pixel_scale use asDict?
        # Get the instrument and the binning of the y-axis values using the 
        # appropriate descriptors
        instrument = dataset.instrument()
        detector_y_bin = dataset.detector_y_bin()
        
        # Set the default pixel scales for GMOS-N and GMOS-S
        if instrument == 'GMOS-N':
            scale = 0.0727
        if instrument == 'GMOS-S':
            scale = 0.073
        
        ret_pixel_scale = {}
        # The binning of the y-axis is used to calculate the pixel scale
        for key, y_bin in detector_y_bin.iteritems():
            # Return a dictionary with the pixel scale float as the value
            ret_pixel_scale.update({key:float(y_bin * scale)})
        
        return ret_pixel_scale
    
    def read_noise(self, dataset, asDict=True, **args):
        # Get the amplifier integration time (ampinteg) and the UT date from
        # the header of the PHU. The ampinteg keyword is defined in the local
        # key dictionary (stdkeyDictGMOS) but is read from the updated global
        # key dictionary (globalStdkeyDict)
        hdu = dataset.hdulist
        ampinteg = hdu[0].header[globalStdkeyDict['key_ampinteg']]
        # Get the UT date using the appropriate descriptor
        ut_date = dataset.ut_date(asString=True)
        obs_ut_date = datetime(*strptime(ut_date, '%Y-%m-%d')[0:6])
        old_ut_date = datetime(2006, 8, 31, 0, 0)
        
        if asDict:
            ret_read_noise = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Check if the original gain (gainorig) keyword exists in the
                # header of the pixel data extension. The gainorig keyword is
                # defined in the local key dictionary (stdkeyDictGMOS) but is
                # read from the updated global key dictionary
                # (globalStdkeyDict)
                if ext.header.has_key(globalStdkeyDict['key_gainorig']):
                    headergain = ext.header[globalStdkeyDict['key_gainorig']]
                else:
                    headergain = ext.header[globalStdkeyDict['key_gain']]
                
                # Get the name of the detector amplifier (ampname) from the
                # header of each pixel data extension. The ampname keyword is
                # defined in the local key dictionary (stdkeyDictGMOS) but is
                # read from the updated global key dictionary
                # (globalStdkeyDict)
                ampname = ext.header[globalStdkeyDict['key_ampname']]
                
                # Get the gain setting and read speed setting values using the
                # appropriate descriptors
                gain_setting = dataset.gain_setting()
                read_speed_setting = dataset.read_speed_setting()
                
                read_noise_key = (read_speed_setting, gain_setting, ampname)
                
                if obs_ut_date > old_ut_date:
                    read_noise = self.gmosampsRdnoise[read_noise_key]
                else:
                    read_noise = \
                        self.gmosampsRdnoiseBefore20060831[read_noise_key]
                
                # Return a dictionary with the read noise float as the value
                ret_read_noise.update({(ext.extname(), \
                    ext.extver()):float(read_noise)})
        else:
            # Check to see whether the dataset has a single extension and if 
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Check if the original gain (gainorig) keyword exists in the
                # header of the pixel data extension. The gainorig keyword is
                # defined in the local key dictionary (stdkeyDictGMOS) but is
                # read from the updated global key dictionary
                # (globalStdkeyDict)
                hdu = dataset.hdulist
                if hdu[1].header.has_key(globalStdkeyDict['key_gainorig']):
                    headergain = \
                        hdu[1].header[globalStdkeyDict['key_gainorig']]
                else:
                    headergain = \
                        hdu[1].header[globalStdkeyDict['key_gain']]
                
                # Get the name of the detector amplifier (ampname) from the
                # header of the single pixel data extension. The ampname
                # keyword is defined in the local key dictionary
                # (stdkeyDictGMOS) but is read from the updated global key
                # dictionary (globalStdkeyDict)
                ampname = hdu[1].header[globalStdkeyDict['key_ampname']]
                # Get the gain setting and read speed setting values using the
                # appropriate descriptors
                gain_setting = dataset.gain_setting()
                read_speed_setting = dataset.read_speed_setting()
                
                read_noise_key = (read_speed_setting, gain_setting, ampname)
                
                if obs_ut_date > old_ut_date:
                    read_noise = self.gmosampsRdnoise[read_noise_key]
                else:
                    read_noise = \
                        self.gmosampsRdnoiseBefore20060831[read_noise_key]
                
                # Return the read noise float
                ret_read_noise = float(read_noise)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_read_noise
    
    gmosampsRdnoise = None
    gmosampsRdnoiseBefore20060831 = None
    
    def read_speed_setting(self, dataset, **args):
        # Get the amplifier integration time (ampinteg) from the header of the
        # PHU. The ampinteg keyword is defined in the local key dictionary
        # (stdkeyDictGMOS) but is read from the updated global key dictionary
        # (globalStdkeyDict)
        hdu = dataset.hdulist
        ampinteg = hdu[0].header[globalStdkeyDict['key_ampinteg']]
        
        if ampinteg == 1000:
            ret_read_speed_setting = 'fast'
        else:
            ret_read_speed_setting = 'slow'
        
        return ret_read_speed_setting
    
    def saturation_level(self, dataset, **args):
        # Return the saturation level integer
        ret_saturation_level = int(65536)
        
        return ret_saturation_level
    
    def wavelength_reference_pixel(self, dataset, asDict=True, **args):
        if asDict:
            ret_wavelength_reference_pixel = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Get the reference pixel of the central wavelength from the
                # header of each pixel data extension. The reference pixel of
                # the central wavelength keyword is defined in the local key
                # dictionary (stdkeyDictGMOS) but is read from the updated
                # global key dictionary (globalStdkeyDict)
                wavelength_reference_pixel = \
                    ext.header\
                    [globalStdkeyDict['key_wavelength_reference_pixel']]
                # Return a dictionary with the reference pixel of the central
                # wavelength float as the value
                ret_wavelength_reference_pixel.update({(ext.extname(), \
                    ext.extver()):float(wavelength_reference_pixel)})
        else:
            # Check to see whether the dataset has a single extension and if 
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Get the reference pixel of the central wavelength from the
                # header of the single pixel data extension. The reference
                # pixel of the central wavelength keyword is defined in the
                # local key dictionary (stdkeyDictGMOS) but is read from the
                # updated global key dictionary (globalStdkeyDict)
                hdu = dataset.hdulist
                wavelength_reference_pixel = \
                    hdu[1].header\
                    [globalStdkeyDict['key_wavelength_reference_pixel']]
                # Return the reference pixel of the central wavelength float
                ret_wavelength_reference_pixel = \
                    float(wavelength_reference_pixel)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_wavelength_reference_pixel
