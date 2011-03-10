from astrodata import Lookups
from astrodata import Descriptors
from astrodata import Errors
import re

from astrodata.Calculator import Calculator

from datetime import datetime
from time import strptime

import GemCalcUtil

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardGMOSKeyDict import stdkeyDictGMOS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class GMOS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dict with the local dict of this descriptor class
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
    
    def amp_read_area(self, dataset, asDict=True, **args):
        if asDict:
            ret_amp_read_area = {}
            for ext in dataset:
                ampname = ext.header[stdkeyDictGMOS['key_ampname']]
                detsec = ext.header[globalStdkeyDict['key_detector_section']]
                amp_read_area = "'%s':%s" % (ampname, detsec)
                ret_amp_read_area.update({(ext.extname(), \
                    ext.extver()):str(amp_read_area)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single string
            if dataset.countExts('SCI') <= 1:
                hdu = dataset.hdulist
                ampname = hdu[1].header[stdkeyDictGMOS['key_ampname']]
                detsec = \
                    hdu[1].header[globalStdkeyDict['key_detector_section']]
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
            ret_central_wavelength = {}
            hdu = dataset.hdulist
            raw_central_wavelength = \
                hdu[0].header[stdkeyDictGMOS['key_central_wavelength']]
            ret_central_wavelength = \
                self.convert_units(input_units = input_units, \
                input_value = raw_central_wavelength, \
                output_units = output_units)
        
        return ret_central_wavelength
    
    def data_section(self, dataset, pretty=False, asDict=True, **args):
        if asDict:
            ret_data_section = {}
            for ext in dataset:
                raw_data_section = \
                    ext.header[globalStdkeyDict['key_data_section']]
                if pretty:
                    # Return a string that uses 1-based indexing
                    ret_data_section.update({(ext.extname(), \
                    ext.extver()):str(raw_data_section)})
                else:
                    # Return a tuple that uses 0-based indexing
                    data_section = self.section_to_tuple(raw_data_section)
                    ret_data_section.update({(ext.extname(), \
                        ext.extver()):data_section})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single string
            if dataset.countExts('SCI') <= 1:
                hdu = dataset.hdulist
                raw_data_section = \
                    hdu[1].header[globalStdkeyDict['key_data_section']]
                if pretty:
                    # Return a string that uses 1-based indexing
                    ret_data_section = raw_data_section
                else:
                    # Return a tuple that uses 0-based indexing
                    data_section = self.section_to_tuple(raw_data_section)
                    ret_data_section = data_section
            else:
                raise Errors.DescriptorDictError()
        
        return ret_data_section
    
    def detector_section(self, dataset, pretty=False, asDict=True, **args):
        if asDict:
            ret_detector_section = {}
            for ext in dataset:
                raw_detector_section = \
                    ext.header[globalStdkeyDict['key_detector_section']]
                if pretty:
                    # Return a string that uses 1-based indexing
                    ret_detector_section.update({(ext.extname(), \
                    ext.extver()):str(raw_detector_section)})
                else:
                    # Return a tuple that uses 0-based indexing
                    detector_section = \
                        self.section_to_tuple(raw_detector_section)
                    ret_detector_section.update({(ext.extname(), \
                        ext.extver()):detector_section})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single string
            if dataset.countExts('SCI') <= 1:
                hdu = dataset.hdulist
                raw_detector_section = \
                    hdu[1].header[globalStdkeyDict['key_detector_section']]
                if pretty:
                    # Return a string that uses 1-based indexing
                    ret_detector_section = raw_detector_section
                else:
                    # Return a tuple that uses 0-based indexing
                    detector_section = \
                        self.section_to_tuple(raw_detector_section)
                    ret_detector_section = detector_section
            else:
                raise Errors.DescriptorDictError()
        
        return ret_detector_section
    
    def detector_x_bin(self, dataset, asDict=True, **args):
        if asDict:
            ret_detector_x_bin = {}
            for ext in dataset:
                ccdsum = ext.header[stdkeyDictGMOS['key_ccdsum']]
                detector_x_bin, detector_y_bin = ccdsum.split()
                ret_detector_x_bin.update({(ext.extname(), \
                    ext.extver()):int(detector_x_bin)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single string
            if dataset.countExts('SCI') <= 1:
                hdu = dataset.hdulist
                ccdsum = hdu[1].header[stdkeyDictGMOS['key_ccdsum']]
                detector_x_bin, detector_y_bin = ccdsum.split()
                ret_detector_x_bin = int(detector_x_bin)
            else:
                raise Errors.DescriptorDictError
        
        return ret_detector_x_bin
    
    def detector_y_bin(self, dataset, asDict=True, **args):
        if asDict:
            ret_detector_y_bin = {}
            for ext in dataset:
                ccdsum = ext.header[stdkeyDictGMOS['key_ccdsum']]
                detector_x_bin, detector_y_bin = ccdsum.split()
                ret_detector_y_bin.update({(ext.extname(), \
                    ext.extver()):int(detector_y_bin)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single string
            if dataset.countExts('SCI') <= 1:
                hdu = dataset.hdulist
                ccdsum = hdu[1].header[stdkeyDictGMOS['key_ccdsum']]
                detector_x_bin, detector_y_bin = ccdsum.split()
                ret_detector_y_bin = int(detector_y_bin)
            else:
                raise Errors.DescriptorDictError
        
        return ret_detector_y_bin
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        hdu = dataset.hdulist
        disperser = hdu[0].header[stdkeyDictGMOS['key_disperser']]
        
        if pretty:
            # If pretty=True, use stripID then additionally remove the
            # trailing '+' from the string
            stripID = True
        
        if stripID:
            if pretty:
                ret_disperser = \
                    str(GemCalcUtil.removeComponentID(disperser).strip('+'))
            else:
                ret_disperser = str(GemCalcUtil.removeComponentID(disperser))
        else:
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
            for ext in dataset:
                raw_dispersion = \
                    ext.header[stdkeyDictGMOS['key_dispersion']]
                dispersion = \
                    self.convert_units(input_units = input_units, \
                    input_value = raw_dispersion, \
                    output_units = output_units)
                ret_dispersion.update({(ext.extname(), \
                    ext.extver()):float(dispersion)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single string
            if dataset.countExts('SCI') <= 1:
                hdu = dataset.hdulist
                raw_dispersion = \
                    hdu[1].header[stdkeyDictGMOS['key_dispersion']]
                dispersion = \
                    self.convert_units(input_units = input_units, \
                    input_value = raw_dispersion, \
                    output_units = output_units)
                ret_dispersion = float(dispersion)
            else:
                raise Errors.DescriptorDictError()

        return ret_dispersion
    
    def exposure_time(self, dataset, **args):
        hdu = dataset.hdulist
        exposure_time = hdu[0].header[globalStdkeyDict['key_exposure_time']]
        
        # Sanity check for times when the GMOS DC is stoned
        if exposure_time > 10000. or exposure_time < 0.:
            raise Errors.CalcError()
        else:
            ret_exposure_time = float(exposure_time)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
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
            ret_filter_name = str('&'.join(filters.sort))
        
        return ret_filter_name
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        hdu = dataset.hdulist
        focal_plane_mask = \
            hdu[0].header[stdkeyDictGMOS['key_focal_plane_mask']]
        
        if focal_plane_mask == 'None':
            ret_focal_plane_mask = 'Imaging'
        else:
            ret_focal_plane_mask = str(focal_plane_mask)
        
        return ret_focal_plane_mask

    def gain(self, dataset, asDict=True, **args):
        hdu = dataset.hdulist
        ampinteg = hdu[0].header[stdkeyDictGMOS['key_ampinteg']]
        ut_date = hdu[0].header[globalStdkeyDict['key_ut_date']]
        obs_ut_date = datetime(*strptime(ut_date, '%Y-%m-%d')[0:6])
        old_ut_date = datetime(2006, 8, 31, 0, 0)

        if asDict:
            ret_gain = {}
            for ext in dataset:
                # Descriptors must work for all AstroData Types so
                # check if the original gain keyword exists to use for
                # the look-up table
                if ext.header.has_key(stdkeyDictGMOS['key_gainorig']):
                    headergain = ext.header[stdkeyDictGMOS['key_gainorig']]
                else:
                    headergain = ext.header[globalStdkeyDict['key_gain']]
                
                ampname = ext.header[stdkeyDictGMOS['key_ampname']]
                gmode = dataset.gain_setting()
                rmode = dataset.read_speed_setting()
                
                gainkey = (rmode, gmode, ampname)
                
                if obs_ut_date > old_ut_date:
                    gain = self.gmosampsGain[gainkey]
                else:
                    gain = self.gmosampsGainBefore20060831[gainkey]
                
                ret_gain.update({(ext.extname(), ext.extver()):gain})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single string
            if dataset.countExts('SCI') <= 1:
                hdu = dataset.hdulist
                # Descriptors must work for all AstroData Types so
                # check if the original gain keyword exists to use for
                # the look-up table
                if hdu[1].header.has_key(stdkeyDictGMOS['key_gainorig']):
                    headergain = hdu[1].header[stdkeyDictGMOS['key_gainorig']]
                else:
                    headergain = hdu[1].header[globalStdkeyDict['key_gain']]
                
                ampname = hdu[1].header[stdkeyDictGMOS['key_ampname']]
                gmode = dataset.gain_setting()
                rmode = dataset.read_speed_setting()
                
                gainkey = (rmode, gmode, ampname)
                
                if obs_ut_date > old_ut_date:
                    gain = self.gmosampsGain[gainkey]
                else:
                    gain = self.gmosampsGainBefore20060831[gainkey]
                ret_gain = float(gain)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_gain
    
    gmosampsGain = None
    gmosampsGainBefore20060831 = None
    
    def gain_setting(self, dataset, **args):
        hdu = dataset.hdulist
        # Descriptors must work for all AstroData Types so check
        # if the original gain keyword exists to use for the look-up table
        if hdu[1].header.has_key(stdkeyDictGMOS['key_gainorig']):
            headergain = hdu[1].header[stdkeyDictGMOS['key_gainorig']]
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
        if ('IMAGE' not in dataset.types) and ('PREPARED' in dataset.types):
            if asDict:
                ret_mdf_row_id = {}
                for ext in dataset:
                    mdf_row_id = \
                        ext.header[globalStdkeyDict['key_mdf_row_id']]
                    ret_mdf_row_id.update({(ext.extname(), \
                        ext.extver()):int(mdf_row_id)})
            else:
                # Check to see whether the dataset has a single extension and
                # if it does, return a single string
                if dataset.countExts('SCI') <= 1:
                    hdu = dataset.hdulist
                    mdf_row_id = \
                        hdu[1].header[globalStdkeyDict['key_mdf_row_id']]
                    ret_mdf_row_id = int(mdf_row_id)
                else:
                    raise Errors.DescriptorDictError()
        else:
            ret_mdf_row_id = None
        
        return ret_mdf_row_id
    
    def non_linear_level(self, dataset, **args):
        # Set the non_linear_level to the saturation_level for GMOS
        ret_non_linear_level = dataset.saturation_level()
        
        return ret_non_linear_level
    
    def pixel_scale(self, dataset, **args):
        # Should pixel_scale use asDict?
        hdu = dataset.hdulist
        instrument = hdu[0].header[globalStdkeyDict['key_instrument']]
        detector_y_bin = dataset.detector_y_bin()
        
        if instrument == 'GMOS-N':
            scale = 0.0727
        if instrument == 'GMOS-S':
            scale = 0.073

        ret_pixel_scale = {}
        for key, y_bin in detector_y_bin.iteritems():
            ret_pixel_scale.update({key:float(y_bin * scale)})
        
        return ret_pixel_scale
    
    def read_noise(self, dataset, asDict=True, **args):
        hdu = dataset.hdulist
        ampinteg = hdu[0].header[stdkeyDictGMOS['key_ampinteg']]
        ut_date = hdu[0].header[globalStdkeyDict['key_ut_date']]
        obs_ut_date = datetime(*strptime(ut_date, '%Y-%m-%d')[0:6])
        old_ut_date = datetime(2006, 8, 31, 0, 0)
        if asDict:
            ret_read_noise = {}
            for ext in dataset:
                # Descriptors must work for all AstroData Types so
                # check if the original gain keyword exists to use for
                # the look-up table
                if ext.header.has_key(stdkeyDictGMOS['key_gainorig']):
                    headergain = ext.header[stdkeyDictGMOS['key_gainorig']]
                else:
                    headergain = ext.header[globalStdkeyDict['key_gain']]
                
                ampname = ext.header[stdkeyDictGMOS['key_ampname']]
                gmode = dataset.gain_setting()
                rmode = dataset.read_speed_setting()
                
                read_noise_key = (rmode, gmode, ampname)
                
                if obs_ut_date > old_ut_date:
                    read_noise = self.gmosampsRdnoise[read_noise_key]
                else:
                    read_noise = \
                        self.gmosampsRdnoiseBefore20060831[read_noise_key]
                
                ret_read_noise.update({(ext.extname(), \
                    ext.extver()):float(read_noise)})
        else:
            # Check to see whether the dataset has a single extension and if 
            # it does, return a single string
            if dataset.countExts('SCI') <= 1:
                hdu = dataset.hdulist
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
                gmode = dataset.gain_setting()
                rmode = dataset.read_speed_setting()
                
                read_noise_key = (rmode, gmode, ampname)
                
                if obs_ut_date > old_ut_date:
                    read_noise = self.gmosampsRdnoise[read_noise_key]
                else:
                    read_noise = \
                        self.gmosampsRdnoiseBefore20060831[read_noise_key]

                ret_read_noise = float(read_noise)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_read_noise
    
    gmosampsRdnoise = None
    gmosampsRdnoiseBefore20060831 = None
    
    def read_speed_setting(self, dataset, **args):
        hdu = dataset.hdulist
        ampinteg = hdu[0].header[stdkeyDictGMOS['key_ampinteg']]
        
        if ampinteg == 1000:
            ret_read_speed_setting = 'fast'
        else:
            ret_read_speed_setting = 'slow'
        
        return ret_read_speed_setting
    
    def saturation_level(self, dataset, **args):
        ret_saturation_level = int(65536)
        
        return ret_saturation_level
    
    def wavelength_reference_pixel(self, dataset, asDict=True, **args):
        if asDict:
            ret_wavelength_reference_pixel = {}
            for ext in dataset:
                wavelength_reference_pixel = \
                    ext.header\
                    [stdkeyDictGMOS['key_wavelength_reference_pixel']]
                ret_wavelength_reference_pixel.update({(ext.extname(), \
                    ext.extver()):float(wavelength_reference_pixel)})
        else:
            # Check to see whether the dataset has a single extension and if 
            # it does, return a single string
            if dataset.countExts('SCI') <= 1:
                hdu = dataset.hdulist
                wavelength_reference_pixel = \
                    hdu[1].header\
                    [stdkeyDictGMOS['key_wavelength_reference_pixel']]
                ret_wavelength_reference_pixel = \
                    float(wavelength_reference_pixel)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_wavelength_reference_pixel

    ## UTILITY MEMBER FUNCTIONS (NOT DESCRIPTORS)
    
    def convert_units(self, input_units, input_value, output_units):
        """
        :param input_units: the units of the value specified by input_value.
                            Possible values are 'meters', 'micrometers',
                            'nanometers' and 'angstroms'.
        :type input_units: string
        :param input_value: the input value to be converted from the
                            input_units to the output_units
        :type input_value: float
        :param output_units: the units of the returned value. Possible values
                             are 'meters', 'micrometers', 'nanometers' and
                             'angstroms'.
        :type output_units: string
        :rtype: float
        :return: the converted value of input_value from input_units to
                 output_units
        """
        # Determine the factor required to convert the input_value from the 
        # input_units to the output_units
        power = int(self.unitDict[input_units]) - \
            int(self.unitDict[output_units])
        factor = float('1e' + str(power))

        # Return the converted output value
        return input_value * factor
    
    # The unitDict dictionary defines the factors for the function
    # convert_units
    unitDict = {
        'meters':0,
        'micrometers':-6,
        'nanometers':-9,
        'angstroms':-10,
               }
    
    def section_to_tuple(self, section):
        """
        Convert the input section in the form [x1:x2,y1:y2] to a tuple in the
        form (x1 - 1, x2 - 1, y1 - 1, y2 - 1), where x1, x2, y1 and y2 are
        integers. The values in the output tuple are converted to use 0-based
        indexing, making it compatible with numpy.
        :param section: the section (in the form [x1:x2,y1:y2]) to be
                        converted to a tuple
        :type section: string
        :rtype: tuple
        :return: the converted section as a tuple that uses 0-based indexing
                 in the form (x1 - 1, x2 - 1, y1 - 1, y2 - 1)
        """
        # Strip the square brackets from the input section and then create a
        # list in the form ['x1:x2', 'y1:y2']
        xylist = section.strip('[]').split(',')
        
        # Create variables containing the single x1, x2, y1 and y2 values
        x1 = int(xylist[0].split(':')[0]) - 1
        x2 = int(xylist[0].split(':')[1]) - 1
        y1 = int(xylist[1].split(':')[0]) - 1
        y2 = int(xylist[1].split(':')[1]) - 1

        # Return the tuple in the form (x1, x2, y1, y2)
        return (x1, x2, y1, y2)
