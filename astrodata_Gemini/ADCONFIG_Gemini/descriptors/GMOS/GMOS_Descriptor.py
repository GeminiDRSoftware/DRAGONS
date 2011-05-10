from datetime import datetime
from time import strptime

from astrodata import Descriptors
from astrodata.Descriptors import DescriptorValue
from astrodata import Errors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy import string
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
    
    def amp_read_area(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_amp_read_area = {}
        # Loop over the science extensions in the dataset
        for ext in dataset['SCI']:
            # Get the name of the detector amplifier (ampname) from the header
            # of each pixel data extension. The ampname keyword is defined in
            # the local key dictionary (stdkeyDictGMOS) but is read from the
            # updated global key dictionary (globalStdkeyDict)
            ampname = ext.get_key_value(globalStdkeyDict['key_ampname'])
            if ampname is None:
                # The get_key_value() function returns None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(ext, 'exception_info'):
                    raise ext.exception_info
            # Get the pretty (1-based indexing) readout area of the CCD
            # (detsec) using the appropriate descriptor
            detsec = ext.detector_section(pretty=True)
            if detsec is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(ext, 'exception_info'):
                    raise ext.exception_info
                # Create the composite amp_read_area string
            amp_read_area = "'%s':%s" % (ampname, detsec)
            # Return a dictionary with the composite amp_read_area string as
            # the value
            ret_amp_read_area.update({(ext.extname(), \
                ext.extver()):str(amp_read_area)})
        
        return ret_amp_read_area
    
    def central_wavelength(self, dataset, asMicrometers=False, \
        asNanometers=False, asAngstroms=False, **args):
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
        # Get the central wavelength value from the header of the PHU. The
        # central wavelength keyword is defined in the local key dictionary
        # (stdkeyDictGMOS) but is read from the updated global key dictionary
        # (globalStdkeyDict)
        raw_central_wavelength = \
            dataset.phu_get_key_value(globalStdkeyDict['key_central_wavelength'])
        if raw_central_wavelength is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Use the utilities function convert_units to convert the central
        # wavelength value from the input units to the output units
        ret_central_wavelength = \
            GemCalcUtil.convert_units(input_units=input_units, \
            input_value=float(raw_central_wavelength), \
             output_units=output_units)
        
        return ret_central_wavelength
    
    def detector_x_bin(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_detector_x_bin = {}
        # Loop over the science extensions in the dataset
        for ext in dataset['SCI']:
            # Get the ccdsum value from the header of each pixel data
            # extension. The ccdsum keyword is defined in the local key
            # dictionary (stdkeyDictGMOS) but is read from the updated global
            # key dictionary (globalStdkeyDict)
            ccdsum = ext.get_key_value(globalStdkeyDict['key_ccdsum'])
            if ccdsum is None:
                # The get_key_value() function returns None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(ext, 'exception_info'):
                    raise ext.exception_info
            detector_x_bin, detector_y_bin = ccdsum.split()
            # Return a dictionary with the binning of the x-axis integer as the
            # value
            ret_detector_x_bin.update({(ext.extname(), \
                ext.extver()):int(detector_x_bin)})
        
        return ret_detector_x_bin
    
    def detector_y_bin(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_detector_y_bin = {}
        # Loop over the science extensions in the dataset
        for ext in dataset['SCI']:
            # Get the ccdsum value from the header of each pixel data
            # extension. The ccdsum keyword is defined in the local key
            # dictionary (stdkeyDictGMOS) but is read from the updated global
            # key dictionary (globalStdkeyDict)
            ccdsum = ext.get_key_value(globalStdkeyDict['key_ccdsum'])
            if ccdsum is None:
                # The get_key_value() function returns None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(ext, 'exception_info'):
                    raise ext.exception_info
            detector_x_bin, detector_y_bin = ccdsum.split()
            # Return a dictionary with the binning of the y-axis integer as the
            # value
            ret_detector_y_bin.update({(ext.extname(), \
                ext.extver()):int(detector_y_bin)})
        
        return ret_detector_y_bin
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        # Get the disperser value from the header of the PHU. The disperser
        # keyword is defined in the local key dictionary (stdkeyDictGMOS) but
        # is read from the updated global key dictionary (globalStdkeyDict)
        disperser = dataset.phu_get_key_value(globalStdkeyDict['key_disperser'])
        if disperser is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if pretty:
            # If pretty=True, use stripID then additionally remove the
            # trailing '+' from the string
            stripID = True
        if stripID:
            if pretty:
                # Return the stripped and pretty disperser string
                ret_disperser = \
                    string.removeComponentID(disperser).strip('+')
            else:
                # Return the stripped disperser string
                ret_disperser = string.removeComponentID(disperser)
        else:
            # Return the disperser string
            ret_disperser = str(disperser)
        
        return ret_disperser
    
    def dispersion(self, dataset, asMicrometers=False, asNanometers=False, \
        asAngstroms=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_dispersion = {}
        # Currently for GMOS data, the dispersion is recorded in meters (?)
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
        # Loop over the science extensions in the dataset
        for ext in dataset['SCI']:
            # Get the dispersion value from the header of each pixel data
            # extension. The dispersion keyword is defined in the local key
            # dictionary (stdkeyDictGMOS) but is read from the updated global
            # key dictionary (globalStdkeyDict)
            raw_dispersion = \
                ext.get_key_value(globalStdkeyDict['key_dispersion'])
            if raw_dispersion is None:
                # The get_key_value() function returns None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(ext, 'exception_info'):
                    raise ext.exception_info
            # Use the utilities function convert_units to convert the
            # dispersion value from the input units to the output units
            dispersion = \
                GemCalcUtil.convert_units(input_units=input_units, \
                input_value=float(raw_dispersion), output_units=output_units)
            # Return a dictionary with the dispersion float as the value
            ret_dispersion.update({(ext.extname(), \
                ext.extver()):float(dispersion)})
        
        return ret_dispersion
    
    def exposure_time(self, dataset, **args):
        # Get the exposure time from the header of the PHU
        exposure_time = \
            dataset.phu_get_key_value(globalStdkeyDict['key_exposure_time'])
        if exposure_time is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It 
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Sanity check for times when the GMOS DC is stoned
        if exposure_time > 10000. or exposure_time < 0.:
            raise Errors.InvalidValueError()
        else:
            # Return the exposure time float
            ret_exposure_time = float(exposure_time)
        
        return ret_exposure_time
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        # Get the focal plane mask value from the header of the PHU. The focal
        # plane mask keyword is defined in the local key dictionary
        # (stdkeyDictGMOS) but is read from the updated global key dictionary
        # (globalStdkeyDict)
        focal_plane_mask = \
            dataset.phu_get_key_value(globalStdkeyDict['key_focal_plane_mask'])
        if focal_plane_mask is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if focal_plane_mask == 'None':
            ret_focal_plane_mask = 'Imaging'
        else:
            # Return the focal plane mask string
            ret_focal_plane_mask = str(focal_plane_mask)
        
        return ret_focal_plane_mask
    
    def gain(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_gain = {}
        # If the data have been prepared, take the gain value directly from the
        # appropriate keyword. At some point, a check for the STDHDRSI header
        # keyword should be added, since the function that overwrites the gain
        # keyword also writes the STDHDRSI keyword
        if 'PREPARED' in dataset.types:
            # Loop over the science extensions in the dataset
            for ext in dataset['SCI']:
                gain = ext.get_key_value(globalStdkeyDict['key_gain'])
                if gain is None:
                    # The get_key_value() function returns None if a value
                    # cannot be found and stores the exception info. Re-raise
                    # the exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(ext, 'exception_info'):
                        raise ext.exception_info
                # Return a dictionary with the gain float as the value
                ret_gain.update({(ext.extname(), ext.extver()):float(gain)})
        else:
            # Get the amplifier integration time (ampinteg) and the UT date
            # from the header of the PHU. The ampinteg keyword is defined in
            # the local key dictionary (stdkeyDictGMOS) but is read from the
            # updated global key dictionary (globalStdkeyDict)
            ampinteg = dataset.phu_get_key_value(globalStdkeyDict['key_ampinteg'])
            if ampinteg is None:
                # The phu_get_key_value() function returns None if a value cannot
                # be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, 'exception_info'):
                    raise dataset.exception_info
            # Get the UT date using the appropriate descriptor
            ut_date = str(dataset.ut_date())
            if ut_date is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, 'exception_info'):
                    raise dataset.exception_info
            obs_ut_date = datetime(*strptime(ut_date, '%Y-%m-%d')[0:6])
            old_ut_date = datetime(2006, 8, 31, 0, 0)
            # Loop over the science extensions in the dataset
            for ext in dataset['SCI']:
                # Get the name of the detector amplifier (ampname) from the
                # header of each pixel data extension. The ampname keyword is
                # defined in the local key dictionary (stdkeyDictGMOS) but is
                # read from the updated global key dictionary
                # (globalStdkeyDict)
                ampname = ext.get_key_value(globalStdkeyDict['key_ampname'])
                if ampname is None:
                    # The get_key_value() function returns None if a value
                    # cannot be found and stores the exception info. Re-raise
                    # the exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(ext, 'exception_info'):
                        raise ext.exception_info
                # Get the gain setting and read speed setting values using the
                # appropriate descriptors. Use asPytype() to return the values
                # as the default python type, rather than an object
                read_speed_setting = dataset.read_speed_setting().asPytype()
                gain_setting = dataset.gain_setting().asPytype()
                if read_speed_setting is None or gain_setting is None:
                    # The descriptor functions return None if a value cannot be
                    # found and stores the exception info. Re-raise the
                    # exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(dataset, 'exception_info'):
                        raise dataset.exception_info
                gain_key = (read_speed_setting, gain_setting, ampname)
                if obs_ut_date > old_ut_date:
                    if gain_key in getattr(self, 'gmosampsGain'):
                        gain = self.gmosampsGain[gain_key]
                    else:
                        raise Errors.TableKeyError()
                else:
                    if gain_key in getattr(self, 'gmosampsGainBefore20060831'):
                        gain = self.gmosampsGainBefore20060831[gain_key]
                    else:
                        raise Errors.TableKeyError()
                # Return a dictionary with the gain float as the value
                ret_gain.update({(ext.extname(), ext.extver()):float(gain)})
        
        return ret_gain
    
    gmosampsGain = None
    gmosampsGainBefore20060831 = None
    
    def gain_setting(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_gain_setting = {}
        # If the data have not been prepared, take the raw gain value directly
        # from the appropriate keyword
        if 'PREPARED' not in dataset.types:
            # Loop over the science extensions in the dataset
            for ext in dataset['SCI']:
                # Get the gain from the header of each pixel data extension.
                gain = ext.get_key_value(globalStdkeyDict['key_gain'])
                if gain is None:
                    # The get_key_value() function returns None if a value cannot
                    # be found and stores the exception info. Re-raise the
                    # exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(ext, 'exception_info'):
                        raise ext.exception_info
                if gain > 3.0:
                    gain_setting = 'high'
                else:
                    gain_setting = 'low'
                # Return a dictionary with the gain setting string as the
                # value
                ret_gain_setting.update({(ext.extname(), \
                        ext.extver()):str(gain_setting)})
        else:
            # Can you return the gain_setting without using the raw gain value?
            return DescriptorValue("N/A", format = "db", name = "gain_setting", pytype = str)
            # OLD -> raise Errors.DescriptorTypeError()
        
        return ret_gain_setting
    
    def nod_count(self, dataset, **args):
        # The number of nod and shuffle cycles can only be obtained from nod
        # and shuffle data
        if 'GMOS_NODANDSHUFFLE' in dataset.types:
            # Get the number of nod and shuffle cycles from the header of the
            # PHU
            nod_count = \
                dataset.phu_get_key_value(globalStdkeyDict['key_nod_count'])
            if nod_count is None:
                # The phu_get_key_value() function returns None if a value cannot
                # be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, 'exception_info'):
                    raise dataset.exception_info
            # Return the nod count integer
            ret_nod_count = int(nod_count)
        else:
            raise Errors.DescriptorTypeError()
        
        return ret_nod_count
    
    def nod_pixels(self, dataset, **args):
        # The number of pixel rows the charge is shuffled by can only be
        # obtained from nod and shuffle data
        if 'GMOS_NODANDSHUFFLE' in dataset.types:
            # Get the number of pixel rows the charge is shuffled by from the
            # header of the PHU
            nod_pixels = \
                dataset.phu_get_key_value(globalStdkeyDict['key_nod_pixels'])
            if nod_pixels is None:
                # The phu_get_key_value() function returns None if a value cannot
                # be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, 'exception_info'):
                    raise dataset.exception_info
            # Return the nod pixels integer
            ret_nod_pixels = int(nod_pixels)
        else:
            raise Errors.DescriptorTypeError()
        
        return ret_nod_pixels
    
    def non_linear_level(self, dataset, **args):
        # Set the non linear level equal to the saturation level for GMOS
        ret_non_linear_level = dataset.saturation_level()
        if ret_non_linear_level is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        
        return ret_non_linear_level
    
    def pixel_scale(self, dataset, **args):
        # Get the instrument value and the binning of the y-axis value using
        # the appropriate descriptors
        instrument = dataset.instrument()
        detector_y_bin = dataset.detector_y_bin()
        if instrument is None or detector_y_bin is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Set the default pixel scales for GMOS-N and GMOS-S
        if instrument == 'GMOS-N':
            scale = 0.0727
        if instrument == 'GMOS-S':
            scale = 0.073
        # The binning of the y-axis is used to calculate the pixel scale.
        # Return the pixel scale float
        ret_pixel_scale = float(detector_y_bin * scale)
        
        return ret_pixel_scale
    
    def read_noise(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_read_noise = {}
        # If the data have been prepared, take the read noise value directly
        # from the appropriate keyword. At some point, a check for the STDHDRSI
        # header keyword should be added, since the function that overwrites
        # the read noise keyword also writes the STDHDRSI keyword
        if 'PREPARED' in dataset.types:
            # Loop over the science extensions in the dataset
            for ext in dataset['SCI']:
                read_noise = \
                    ext.get_key_value(globalStdkeyDict['key_read_noise'])
                if read_noise is None:
                    # The get_key_value() function returns None if a value cannot
                    # be found and stores the exception info. Re-raise the
                    # exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(ext, 'exception_info'):
                        raise ext.exception_info
                # Return a dictionary with the read noise float as the value
                ret_read_noise.update({(ext.extname(), \
                    ext.extver()):float(read_noise)})
        else:
            # Get the amplifier integration time (ampinteg) and the UT date
            # from the header of the PHU. The ampinteg keyword is defined in
            # the local key dictionary (stdkeyDictGMOS) but is read from the
            # updated global key dictionary (globalStdkeyDict)
            ampinteg = dataset.phu_get_key_value(globalStdkeyDict['key_ampinteg'])
            if ampinteg is None:
                # The phu_get_key_value() function returns None if a value cannot
                # be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, 'exception_info'):
                    raise dataset.exception_info
            # Get the UT date using the appropriate descriptor
            ut_date = str(dataset.ut_date())
            if ut_date is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, 'exception_info'):
                    raise dataset.exception_info
            obs_ut_date = datetime(*strptime(ut_date, '%Y-%m-%d')[0:6])
            old_ut_date = datetime(2006, 8, 31, 0, 0)
            # Loop over the science extensions of the dataset
            for ext in dataset['SCI']:
                # Get the name of the detector amplifier (ampname) from the
                # header of each pixel data extension. The ampname keyword is
                # defined in the local key dictionary (stdkeyDictGMOS) but is
                # read from the updated global key dictionary
                # (globalStdkeyDict)
                ampname = ext.get_key_value(globalStdkeyDict['key_ampname'])
                if ampname is None:
                    # The get_key_value() function returns None if a value
                    # cannot be found and stores the exception info. Re-raise
                    # the exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(ext, 'exception_info'):
                        raise ext.exception_info
                # Get the gain setting and read speed setting values using the
                # appropriate descriptors. Use asPytype() to return the values
                # as the default python type, rather than an object
                read_speed_setting = dataset.read_speed_setting().asPytype()
                gain_setting = dataset.gain_setting().asPytype()
                if read_speed_setting is None or gain_setting is None:
                    # The descriptor function returns None if a value cannot be
                    # found and stores the exception info. Re-raise the
                    # exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(dataset, 'exception_info'):
                        raise dataset.exception_info
                read_noise_key = (read_speed_setting, gain_setting, ampname)
                if obs_ut_date > old_ut_date:
                    if read_noise_key in getattr(self, 'gmosampsRdnoise'):
                        read_noise = self.gmosampsRdnoise[read_noise_key]
                    else:
                        raise Errors.TableKeyError()
                else:
                    if read_noise_key in getattr(self,
                        'gmosampsRdnoiseBefore20060831[read_noise_key]'):
                        read_noise = \
                            self.gmosampsRdnoiseBefore20060831[read_noise_key]
                    else:
                        raise Errors.TableKeyError()
                # Return a dictionary with the read noise float as the value
                ret_read_noise.update({(ext.extname(), \
                    ext.extver()):float(read_noise)})
        
        return ret_read_noise
    
    gmosampsRdnoise = None
    gmosampsRdnoiseBefore20060831 = None
    
    def read_speed_setting(self, dataset, **args):
        # Get the amplifier integration time (ampinteg) from the header of the
        # PHU. The ampinteg keyword is defined in the local key dictionary
        # (stdkeyDictGMOS) but is read from the updated global key dictionary
        # (globalStdkeyDict)
        ampinteg = dataset.phu_get_key_value(globalStdkeyDict['key_ampinteg'])
        if ampinteg is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if ampinteg == 1000:
            ret_read_speed_setting = 'fast'
        else:
            ret_read_speed_setting = 'slow'
        
        return ret_read_speed_setting
    
    def saturation_level(self, dataset, **args):
        # Return the saturation level integer
        ret_saturation_level = int(65536)
        
        return ret_saturation_level
