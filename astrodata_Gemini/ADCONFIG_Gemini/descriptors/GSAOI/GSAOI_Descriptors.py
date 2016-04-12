import math

from GSAOI_Keywords import GSAOI_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc
from astrodata.interface.Descriptors import DescriptorValue
from astrodata.utils import Errors

from gempy.gemini import gemini_metadata_utils as gmu

import GemCalcUtil

from astrodata_Gemini.ADCONFIG_Gemini.lookups.GSAOI import GSAOIArrayDict
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GSAOI import Nominal_Zeropoints
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GSAOI import GSAOIFilterWavelength

class GSAOI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GSAOI_KeyDict

    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)

    def array_name(self, dataset, **args):

        # Based on GMOS array_name but with a different header keyword.

        key_array_name = self.get_descriptor_key("key_array_name")

        # Get the value of the name of the array keyword from the header of
        # each pixel data extension as a dictionary where the key of the
        # dictionary is an ("*", EXTVER) tuple
        ret_name = gmu.get_key_value_dict(adinput=dataset,
                                          keyword=key_array_name)

        if ret_name is None:

            # If the data have been mosaicked, the separate array identifiers
            # no longer apply; return the detector name instead, as for GMOS.
            ret_name = dataset.detector_name()

            if ret_name.is_none():
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface. 
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info

        return DescriptorValue(ret_name, name="array_name", ad=dataset)

    def central_wavelength(self, dataset, asMicrometers=False,
                           asNanometers=False, asAngstroms=False, **args):
        # Currently for GSAOI data, the central wavelength is recorded in
        # angstroms
        input_units = "angstroms"
        
        # Determine the output units to use
        unit_arg_list = [asMicrometers, asNanometers, asAngstroms]
        
        if unit_arg_list.count(True) == 1:
            # Just one of the unit arguments was set to True. Return the
            # central wavelength in these units
            if asMicrometers:
                output_units = "micrometers"
            if asNanometers:
                output_units = "nanometers"
            if asAngstroms:
                output_units = "angstroms"
        else:
            # Either none of the unit arguments were set to True or more than
            # one of the unit arguments was set to True. In either case,
            # return the central wavelength in the default units of meters
            output_units = "meters"
        
        # Determine the central wavelength keyword from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_central_wavelength")
        
        # Get the value of the central wavelength keyword from the header of
        # the PHU
        raw_central_wavelength = dataset.phu_get_key_value(keyword)
        if raw_central_wavelength is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        else:
            central_wavelength = float(raw_central_wavelength)
        
        # Validate the central wavelength value
        if central_wavelength < 0.0:
            raise Errors.InvalidValueError()
        else:
            # Use the utilities function convert_units to convert the central
            # wavelength value from the input units to the output units
            ret_central_wavelength = GemCalcUtil.convert_units(
              input_units=input_units, input_value=central_wavelength,
              output_units=output_units)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_central_wavelength,
                                 name="central_wavelength", ad=dataset)
        return ret_dv

    def gain(self, dataset, **args):

        # Gain is the second column (ie. index 1) of the look-up table. The
        # columns should probably live in their own sub-dict rather than a
        # tuple but this is how it works for the other instruments:
        ret_dict = self._look_up_arr_property(dataset, table_index=1,
                                              as_type=float)

        return DescriptorValue(ret_dict, name="gain", ad=dataset)

    def group_id(self, dataset, **args):
        # Adapted from the F2 group_id descriptor.
 
        # Descriptors used for all image types
        unique_id_descriptor_list_all = ["read_mode", "detector_section"]
        
        # List to format descriptor calls using 'pretty=True' argument
        call_pretty_version_list = ["filter_name"]
        
        # Descriptors to be returned as an ordered list using 
        # descriptor 'as_list' method
        convert_to_list_list = ["detector_section"]
        
        # Additional descriptors required for each frame type
        # Note: dark_id and flat_twilight are not in common use.
        #       Those are therefore place holder with initial guess
        #
        #       For flat_id, "on" and "off" domeflats are not taken
        #       with the same obsID, so that association cannot be
        #       made.  The only other sensible characteristic would
        #       be to require a timeframe check, eg. within X hours.
        #
        #       The UT date and local date change in the middle of the 
        #       night.  Can't reliably use that.  Thought for a while
        #       using the either the fake UT or equivalently the date
        #       string in the file would work, but found at least one
        #       case where the flat sequence is taken across the 2pm
        #       filename change.  The f-word does come to mind.
        #
        #       Because the group_id is a static string, I can't use
        #       if-tricks or or-tricks.  The only thing that doesn't
        #       change is the program ID.  That's a bit procedural though
        #       but that's the only thing left.
        #
        #dark_id = ["exposure_time", "coadds"]
        flat_id = ["filter_name", "exposure_time", "program_id"]
        #flat_twilight_id = ["filter_name"]
        science_id = ["observation_id", "filter_name", "exposure_time"]
        
        # non-descriptor strings to attach to the group_id
        #   Note: the local date will be added for flats below.
        additional_item_to_include = []

        # Associate rules with data type
        # Note: add darks and twilight if necessary later.
        data_types = dataset.types
        if "GSAOI_IMAGE_FLAT" in data_types:
            id_descriptor_list = flat_id
            # get the local date and save as additional item.
            #datestr = re.search('S([0-9]+)S.*', dataset.filename).group(1)
            #additional_item_to_include = [datestr]
            
        else:
            id_descriptor_list = science_id
        
        # Add in all the common descriptors required
        id_descriptor_list.extend(unique_id_descriptor_list_all)
        
        # Form the group_id
        descriptor_object_string_list = []
        for descriptor in id_descriptor_list:
            # Set 'pretty' argument to descriptor call 
            pretty = False
            if descriptor in call_pretty_version_list:
                pretty = True

            # Call the descriptor
            descriptor_object = getattr(dataset, descriptor)(pretty=pretty)
            
            # Check for a returned descriptor value object with a None value
            if descriptor_object.is_none():
                # The descriptor functions return None if a value cannot be 
                # found and stores the exception info. Re-raise the exception. 
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
                
            # In some cases require the information as a list
            if descriptor in convert_to_list_list:
                descriptor_object = descriptor_object.as_list()
            
            # Convert DV value to a string and store
            descriptor_object_string_list.append(str(descriptor_object))
        
        # Add in any non-descriptor related information
        for item in additional_item_to_include:
            descriptor_object_string_list.append(item)
       
        # Create the final group_id string
        ret_group_id = '_'.join(descriptor_object_string_list)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_group_id, name="group_id", ad=dataset)
        
        return ret_dv


    def is_coadds_summed(self, dataset, **args):
        # Returns True for observations where the pixel data represent the
        # sum over the total exposure time, which is the default
        # But is NOT the case for GSAOI
        return DescriptorValue(False, name="is_coadds_summed", ad=dataset)
    
    def nominal_photometric_zeropoint(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always construct a dictionary where the
        # key of the dictionary is an (EXTNAME, EXTVER) tuple
        ret_nominal_photometric_zeropoint = {}
        
        table = Nominal_Zeropoints.nominal_zeropoints
        
        # Get the values of the gain, detector name and filter name using the
        # appropriate descriptors. Use as_pytype() to return the values as the
        # default python type rather than an object.
        gain_dict = dataset.gain().as_dict()
        filter_name = dataset.filter_name(pretty=True).as_pytype()
        array_name_dict = dataset.array_name().as_dict()

        # # Get the value of the BUNIT keyword from each extension:
        # bunit_dict = gmu.get_key_value_dict(adinput=dataset, keyword="BUNIT")

        if filter_name is None or None in gain_dict.values() \
                               or None in array_name_dict.values():
            # The descriptor functions return None if a value cannot be
            # found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        # Look up & adjust the appropriate value for each extension:
        for ext_name_ver, array_name in array_name_dict.iteritems():
            
            nominal_zeropoint_key = (filter_name, array_name)
            
            if nominal_zeropoint_key in table:
                nominal_photometric_zeropoint = table[nominal_zeropoint_key]
            else:
                raise Errors.TableKeyError()

            # Get the units (ADU or e-) from the SCI extensions, ignoring case:
            sciext = ('SCI', ext_name_ver[1])
            bunit = dataset[sciext].get_key_value('BUNIT')
            if bunit is None:
                raise KeyError('missing BUNIT value(s) in SCI headers')
            else:
                bunit = bunit.upper()   # FITS std requires a string val

            # Adjust value by the appropriate factor for electrons, if needed:
            if bunit == "ADU":
                nominal_photometric_zeropoint -= \
                    2.5 * math.log10(gain_dict[ext_name_ver])
            
            # Update the dictionary with the nominal photometric zeropoint
            # value 
            ret_nominal_photometric_zeropoint[ext_name_ver] = \
                nominal_photometric_zeropoint
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_nominal_photometric_zeropoint,
                                 name="nominal_photometric_zeropoint",
                                 ad=dataset)
        return ret_dv


    def nonlinearity_coeffs(self, dataset, **args):
        
        coeff0 = self._look_up_arr_property(dataset, table_index=4,
                                            as_type=float)
        coeff1 = self._look_up_arr_property(dataset, table_index=5,
                                            as_type=float)
        coeff2 = self._look_up_arr_property(dataset, table_index=6,
                                            as_type=float)
        
        ret_dict = { key : [coeff0[key], coeff1[key], coeff2[key]] \
                    for key in coeff0 }
        
        return DescriptorValue(ret_dict, name="nonlinearity_coeffs", ad=dataset)

    def non_linear_level(self, dataset, **args):

        # Get the saturation levels by re-using their descriptor:
        saturation_dv = dataset.saturation_level()

        if saturation_dv.is_none() and hasattr(dataset, "exception_info"):
            raise dataset.exception_info

        # Convert to a dict only if successful (to facilitate above is_none):
        sat_dict = saturation_dv.as_dict()

        # Get linear limit as a fraction of the saturation level from 4th
        # column of the look-up table:
        thresh_dict = self._look_up_arr_property(dataset, table_index=3,
                                                 as_type=float)

        # Multiply saturation limit by non-linear fraction to get linear limit
        # in ADUs. This assumes the dict returned by saturation_level() has the
        # same keys as here, which should be guaranteed:
        ret_dict = { key : thresh_dict[key] * sat_dict[key] \
                     for key in thresh_dict }

        return DescriptorValue(ret_dict, name="non_linear_level", ad=dataset)

    def pixel_scale(self, dataset, **args):

        # This support method is inherited from the Gemini descriptor class:
        scale_dict = self._get_wcs_pixel_scale(dataset)

        # Convert the extension dict vals to an average float because that's
        # what this descriptor is defined to return (including for GMOS). NB.
        # the "format" param doesn't get passed along by the infrastructure.
        ret_scale = sum(val for val in scale_dict.itervalues()) \
                  / float(len(scale_dict))

        return DescriptorValue(ret_scale, name="pixel_scale", ad=dataset)

    def read_noise(self, dataset, **args):

        # Read noise is the first column (ie. index 0) of the look-up table:
        ret_dict = self._look_up_arr_property(dataset, table_index=0,
                                              as_type=float)
        
        # Adjusting for the number of coadds
        coadds = dataset.coadds()
        if coadds is None:
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        for key in ret_dict:
            ret_dict[key] = round(ret_dict[key] * math.sqrt(coadds), 2)
            
        return DescriptorValue(ret_dict, name="read_noise", ad=dataset)

    def read_speed_setting(self, dataset, **args):

        # Definitions based on LNRS, from the OT:
        read_modes = { 2 : 'Bright Objects',
                       8 : 'Faint Objects',
                      16 : 'Very Faint Objects' }

        # Get the number of non-destructive read pairs (LNRS) from the
        # primary header via the keyword dictionary:
        key_lnrs = self.get_descriptor_key("key_lnrs")
        n_lnrs = dataset.phu_get_key_value(key_lnrs)

        if n_lnrs is None:
            # This is copied from the GNIRS read mode descriptor, which says
            # that if the phu_get_key_value() function returns None and stores
            # exception info, which we re-raise, if a value cannot be found.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        if n_lnrs in read_modes:
            setting = read_modes[n_lnrs]
        else:
            setting = 'Invalid'

         # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(setting, name="read_speed_setting", ad=dataset)

        return ret_dv

    def saturation_level(self, dataset, **args):

        # The values looked up here are actually the limit at which linearity
        # is still accurate within 2% *after* correction, rather than the hard
        # capacitative well saturation level, which is just a bit (~3%) higher.
        # Unlike the IRAF package, the GSAOIArrayDict file does not list full
        # well values separately, but above 2% corrected linearity, the
        # response curve very quickly flattens to a constant value and is no
        # longer considered useful.

        ret_dict = self._look_up_arr_property(dataset, table_index=2,
                                              as_type=int)

        return DescriptorValue(ret_dict, name="saturation_level", ad=dataset)

    def wavelength_band(self, dataset, **args):
        
        if "IMAGE" in dataset.types:
            # If imaging, associate the filter name with a central wavelength
            filter_table = GSAOIFilterWavelength.filter_wavelength
            filter = str(dataset.filter_name(pretty=True))
#            print "filter = ", filter
#            print "filter_table = ", filter_table
            if filter in filter_table:
                ctrl_wave = filter_table[filter]
            else:
                raise Errors.TableKeyError()
        else:
            ctrl_wave = dataset.central_wavelength(asMicrometers=True)
                
        min_diff = None
        band = None
#        print "ctrl_wave = ", ctrl_wave
        for std_band, std_wave in self.std_wavelength_band.items():
            diff = abs(std_wave - ctrl_wave)
            if min_diff is None or diff < min_diff:
                min_diff = diff
                band = std_band
        
        if band is None:
            raise Errors.CalcError()
        else:
            ret_wavelength_band = band
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_wavelength_band, name="wavelength_band",
                                 ad=dataset)
        return ret_dv

    def wcs_ra(self, dataset, **args):
        # Return the RA derived from the WCS
        # This GMOS version simply returns the reference value for each coordinate
        # from the WCS. This is simplistic, but very robust and is good for GSAOI
        # Try the first science extension
        crval = dataset['SCI', 1].get_key_value('CRVAL1')
        ctype = dataset['SCI', 1].get_key_value('CTYPE1')

        # If None, try the PHU
        crval = dataset.phu_get_key_value('CRVAL1') if crval is None else crval
        ctype = dataset.phu_get_key_value('CTYPE1') if ctype is None else ctype

        if crval is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        if ctype != 'RA---TAN':
            raise Errors.InvalidValueError

        return DescriptorValue(crval, name="wcs_ra", ad=dataset)

    def wcs_dec(self, dataset, **args):
        # Return the DEC derived from the WCS
        # This generic version simply returns the REFERENCE VALUE
        # from the WCS. This is simplistic, but very robust and is good for GSAOI
        # Try the first science extension
        crval = dataset['SCI', 1].get_key_value('CRVAL2')
        ctype = dataset['SCI', 1].get_key_value('CTYPE2')

        # If None, try the PHU
        crval = dataset.phu_get_key_value('CRVAL2') if crval is None else crval
        ctype = dataset.phu_get_key_value('CTYPE2') if ctype is None else ctype

        if crval is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        if ctype != 'DEC--TAN':
            raise Errors.InvalidValueError

        return DescriptorValue(crval, name="wcs_dec", ad=dataset)

    def _look_up_arr_property(self, dataset, table_index, as_type=None):
        """
        Support function to look up values for detector characteristics (such
        as the gain for each quadrant) in the "array dict" file.

        :param     dataset: AstroData instance for which to perform the table
                            look-up.
        :type      dataset: AstroData

        :param table_index: The zero-indexed column number(s) of the value to
                            look up within the tuple of values. This may be a
                            single number or a range tuple (exclusive of the
                            upper limit, as per the usual Python convention).
        :type  table_index: int or sequence of int

        :param     as_type: Type to convert each constituent value to, if any.
        :type      as_type: type, optional

        """
        # If provided with a range tuple as the index (for looking up coeffs),
        # rather than a single column, convert it to a slice. For some reason,
        # AstroData mangles the returned dict into a single tuple for ranges
        # containing a single element, but there's no use case for that anyway.
        if hasattr(table_index, '__getitem__'):
            table_index = slice(*table_index)

        # The table value to look up depends on these descriptor values:
        read_speed_setting = dataset.read_speed_setting().as_pytype()
        array_name_dict = dataset.array_name().as_dict()

        if read_speed_setting is None:
            # This descriptor returns None if a value cannot be found and
            # stores the exception info, which we re-raise.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info

        ret_val_dict = {}

        # Loop over the array extension(s) & add each gain to the results dict:
        for ext_key, array_name in array_name_dict.iteritems():

            lookup_key = (read_speed_setting, array_name)

            if lookup_key in GSAOIArrayDict.GSAOIArrayDict:

                val = GSAOIArrayDict.GSAOIArrayDict[lookup_key][table_index]

                # Attempt to convert to a specified data type if requested:
                if as_type:
                    val = tuple(as_type(item) for item in val) \
                           if hasattr(val, '__iter__') else as_type(val)

            else:
                # As well as missing entries, this covers the cases where
                # read_speed_setting is 'Invalid' or array_name is None.
                # The failed look-up raises an exception for consistency with
                # the other instruments (passing through a default message).
                raise Errors.TableKeyError()

            ret_val_dict[ext_key] = val

        return ret_val_dict

