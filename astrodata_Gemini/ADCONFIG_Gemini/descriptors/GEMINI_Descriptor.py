import datetime, os, re
import dateutil.parser

from astrodata import AstroData
from astrodata import Descriptors
from astrodata import Errors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy import string
import GemCalcUtil

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardGEMINIKeyDict import stdkeyDictGEMINI
from Generic_Descriptor import Generic_DescriptorCalc

class GEMINI_DescriptorCalc(Generic_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    globalStdkeyDict.update(stdkeyDictGEMINI)
    
    def airmass(self, dataset, **args):
        # Get the airmass value from the header of the PHU
        airmass = dataset.phuGetKeyValue(globalStdkeyDict['key_airmass'])
        if airmass is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Validate the airmass value
        if airmass < 1.0:
            raise Errors.InvalidValueError()
        else:
            ret_airmass = float(airmass)
        
        return ret_airmass
    
    def amp_read_area(self, dataset, **args):
        # The amp_read_area descriptor is only specific to GMOS data. The code
        # below will be replaced with the GMOS specific amp_read_area
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of 'GMOS'. For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def cass_rotator_pa(self, dataset, **args):
        # Get the cassegrain rotator position angle from the header of the PHU
        cass_rotator_pa = \
            dataset.phuGetKeyValue(globalStdkeyDict['key_cass_rotator_pa'])
        if cass_rotator_pa is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Validate the cassegrain rotator position angle value
        if cass_rotator_pa < -360.0 or cass_rotator_pa > 360.0:
            raise Errors.InvalidValueError()
        else:
            ret_cass_rotator_pa = float(cass_rotator_pa)
        
        return ret_cass_rotator_pa
    
    def central_wavelength(self, dataset, asMicrometers=False, \
        asNanometers=False, asAngstroms=False, **args):
        # For most Gemini data, the central wavelength is recorded in
        # micrometers
        input_units = 'micrometers'
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
        # Get the central wavelength value from the header of the PHU.
        raw_central_wavelength = dataset.phuGetKeyValue\
            (globalStdkeyDict['key_central_wavelength'])
        if raw_central_wavelength is None:
            # The phuGetKeyValue() function returns None if a value cannot be
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
    
    def coadds(self, dataset, **args):
        # Return the coadds integer (set to 1 as default for Gemini data)
        ret_coadds = int(1)
        
        return ret_coadds
    
    def data_section(self, dataset, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_data_section = {}
        # Loop over the science extensions in the dataset
        for ext in dataset['SCI']:
            # Get the data section from the header of each pixel data extension
            raw_data_section = \
                ext.getKeyValue(globalStdkeyDict['key_data_section'])
            if raw_data_section is None:
                # The getKeyValue() function returns None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(ext, 'exception_info'):
                    raise ext.exception_info
            if pretty:
                # Return a dictionary with the data section string that uses
                # 1-based indexing as the value
                ret_data_section.update({(ext.extname(), \
                    ext.extver()):str(raw_data_section)})
            else:
                # Return a dictionary with the data section tuple that uses
                # 0-based indexing as the value
                data_section = \
                    string.section_to_tuple(raw_data_section)
                ret_data_section.update({(ext.extname(), \
                    ext.extver()):data_section})
        
        return ret_data_section
    
    def decker(self, dataset, stripID=False, pretty=False, **args):
        """
        In GNIRS, the decker is used to basically mask off the ends of the
        slit to create the short slits used in the cross dispersed modes.
        """
        # Get the decker position from the header of the PHU
        decker = dataset.phuGetKeyValue(globalStdkeyDict['key_decker'])
        if decker is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if pretty:
            stripID = True
        if stripID:
            # Return the decker string with the component ID stripped
            ret_decker = string.removeComponentID(decker)
        else:
            # Return the decker string
            ret_decker = str(decker)
        
        return ret_decker
    
    def detector_section(self, dataset, pretty=False, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_detector_section = {}
        # Loop over the science extensions in the dataset
        for ext in dataset['SCI']:
            # Get the detector section from the header of each pixel data
            # extension
            raw_detector_section = \
                ext.getKeyValue(globalStdkeyDict['key_detector_section'])
            if raw_detector_section is None:
                # The getKeyValue() function returns None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(ext, 'exception_info'):
                    raise ext.exception_info
            if pretty:
                # Return a dictionary with the detector section string that 
                # uses 1-based indexing as the value
                ret_detector_section.update({(ext.extname(), \
                    ext.extver()):str(raw_detector_section)})
            else:
                # Return a dictionary with the detector section tuple that 
                # uses 0-based indexing as the value
                detector_section = \
                    string.section_to_tuple(raw_detector_section)
                ret_detector_section.update({(ext.extname(), \
                    ext.extver()):detector_section})
        
        return ret_detector_section
    
    def detector_x_bin(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_detector_x_bin = {}
        # Loop over the science extensions in the dataset
        for ext in dataset['SCI']:
            # Return a dictionary with the binning of the x-axis integer (set
            # to 1 as default for Gemini data) as the value
            ret_detector_x_bin.update({(ext.extname(), \
                ext.extver()):int(1)})
        
        return ret_detector_x_bin
    
    def detector_y_bin(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_detector_y_bin = {}
        # Loop over the science extensions in the dataset
        for ext in dataset['SCI']:
            # Return a dictionary with the binning of the y-axis integer (set
            # to 1 as default for Gemini data) as the value
            ret_detector_y_bin.update({(ext.extname(), \
                ext.extver()):int(1)})
        
        return ret_detector_y_bin
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        # Get the disperser value from the header of the PHU. The disperser
        # keyword may be defined in a local key dictionary
        # (stdkeyDict<INSTRUMENT>) but is read from the updated global key
        # dictionary (globalStdkeyDict)
        disperser = dataset.phuGetKeyValue(globalStdkeyDict['key_disperser'])
        if disperser is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if pretty:
            stripID = True
        if stripID:
            # Return the disperser string with the component ID stripped
            ret_disperser = string.removeComponentID(disperser)
        else:
            # Return the disperser string
            ret_disperser = str(disperser)
        
        return ret_disperser
    
    def dispersion_axis(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_dispersion_axis = {}
        # The dispersion axis can only be obtained from data that does not
        # have an AstroData Type of IMAGE and that has been prepared (since
        # the dispersion axis keyword is written during the prepare step)
        if 'IMAGE' not in dataset.types and 'PREPARED' in dataset.types:
            # Loop over the science extensions in the dataset
            for ext in dataset['SCI']:
                # Get the dispersion axis from the header of each pixel data
                # extension
                dispersion_axis = \
                    ext.getKeyValue(globalStdkeyDict['key_dispersion_axis'])
                if dispersion_axis is None:
                    # The getKeyValue() function returns None if a value
                    # cannot be found and stores the exception info. Re-raise
                    # the exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(ext, 'exception_info'):
                        raise ext.exception_info
                # Return a dictionary with the dispersion axis integer as
                # the value
                ret_dispersion_axis.update({(ext.extname(), \
                    ext.extver()):int(dispersion_axis)})
        else:
            raise Errors.DescriptorTypeError()
        
        return ret_dispersion_axis
    
    def exposure_time(self, dataset, **args):
        # Get the exposure time value from the header of the PHU
        exposure_time = \
            dataset.phuGetKeyValue(globalStdkeyDict['key_exposure_time'])
        if exposure_time is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # If the data have been prepared, take the (total) exposure time value
        # directly from the appropriate keyword
        if 'PREPARED' in dataset.types:
            # Get the total exposure time value from the header of the PHU
            ret_exposure_time = float(exposure_time)
        else:
            # Get the number of coadds using the appropriate descriptor
            coadds = dataset.coadds()
            if coadds is None:
                # The descriptor functions return None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, 'exception_info'):
                    raise dataset.exception_info
            ret_exposure_time = exposure_time * coadds
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Get the two filter name values from the header of the PHU. The two
        # filter name keywords are defined in the local key dictionary
        # (stdkeyDictGMOS) but are read from the updated global key dictionary
        # (globalStdkeyDict)
        key_filter1 = globalStdkeyDict['key_filter1']
        key_filter2 = globalStdkeyDict['key_filter2']
        filter1 = dataset.phuGetKeyValue(key_filter1)
        filter2 = dataset.phuGetKeyValue(key_filter2)
        if filter1 is None or filter2 is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if pretty:
            stripID = True
        if stripID:
            # Strip the component ID from the two filter name values
            filter1 = string.removeComponentID(filter1)
            filter2 = string.removeComponentID(filter2)
        # Return a dictionary with the keyword names as the key and the filter
        # name string as the value
        ret_filter_name = {}
        if pretty:
            # Remove any filters that have the value 'open' or 'Open'
            if 'open' not in filter1 and 'Open' not in filter1:
                ret_filter_name.update({key_filter1:str(filter1)})
            if 'open' not in filter2 and 'Open' not in filter2:
                ret_filter_name.update({key_filter2:str(filter2)})
            if len(ret_filter_name) == 0:
                ret_filter_name = 'open'
        else:
            # Return a dictionary with the filter name string as the value
            ret_filter_name.update({key_filter1:str(filter1), \
                key_filter2:str(filter2)})
        
        return ret_filter_name
    
    def focal_plane_mask(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            stripID = True
        # Get the focal plane mask value from the header of the PHU.
        focal_plane_mask = \
            dataset.phuGetKeyValue(globalStdkeyDict['key_focal_plane_mask'])
        if focal_plane_mask is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if stripID:
            # Return the focal plane mask string with the component ID stripped
            ret_focal_plane_mask = \
                string.removeComponentID(focal_plane_mask)
        else:
            # Return the focal plane mask string
            ret_focal_plane_mask = str(focal_plane_mask)
        
        return ret_focal_plane_mask
    
    def gain_setting(self, dataset, **args):
        # The gain_setting descriptor is only specific to GMOS data. The code
        # below will be replaced with the GMOS specific gain_setting
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of 'GMOS'. For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def local_time(self, dataset, **args):
        # Get the local time from the header of the PHU
        local_time = dataset.phuGetKeyValue(globalStdkeyDict['key_local_time'])
        if local_time is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Validate the local time value. The assumption is that the standard
        # mandates HH:MM:SS[.S]. We don't enforce the number of decimal places.
        # These are somewhat basic checks, it's not completely rigorous. Note
        # that seconds can be > 59 when leap seconds occurs
        if re.match('^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$', local_time):
            ret_local_time = str(local_time)
        else:
            raise Errors.InvalidValueError()
        
        return ret_local_time
    
    def mdf_row_id(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_mdf_row_id = {}
        # The MDF row ID can only be obtained from data that does not have an
        # AstroData Type of IMAGE and that has been cut (since the MDF row ID
        # keyword is written during the cut step). As there is no CUT type yet,
        # just check whether the dataset has been prepared.
        if 'IMAGE' not in dataset.types and 'PREPARED' in dataset.types:
            # Loop over the science extensions of the dataset
            for ext in dataset['SCI']:
                # Get the MDF row ID from the header of each pixel data
                # extension
                mdf_row_id = \
                    ext.getKeyValue(globalStdkeyDict['key_mdf_row_id'])
                if mdf_row_id is None:
                    # The getKeyValue() function returns None if a value
                    # cannot be found and stores the exception info. Re-raise
                    # the exception. It will be dealt with by the
                    # CalculatorInterface.
                    if hasattr(ext, 'exception_info'):
                        raise ext.exception_info
                # Return a dictionary with the MDF row ID integer as the
                # value
                ret_mdf_row_id.update({(ext.extname(), \
                    ext.extver()):int(mdf_row_id)})
        else:
            raise Errors.DescriptorTypeError()
        
        return ret_mdf_row_id
    
    def nod_count(self, dataset, **args):
        # The nod_count descriptor is only specific to GMOS data. The code
        # below will be replaced with the GMOS specific nod_count
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of 'GMOS'. For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def nod_pixels(self, dataset, **args):
        # The nod_pixels descriptor is only specific to GMOS data. The code
        # below will be replaced with the GMOS specific nod_pixels
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of 'GMOS'. For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def read_mode(self, dataset, **args):
        # The read_mode descriptor is only specific to GNIRS, NIFS and NIRI
        # data. The code below will be replaced with the GNIRS, NIFS or NIRI
        # specific read_mode descriptor function located in the instrument
        # specific descriptor files. For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def read_speed_setting(self, dataset, **args):
        # The read_speed_setting descriptor is only specific to GMOS data. The
        # code below will be replaced with the GMOS specific read_speed_setting
        # descriptor function located in GMOS/GMOS_Descriptor.py for data with
        # an AstroData Type of 'GMOS'. For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
    
    def qa_state(self, dataset, **args):
        # Get the value for whether the PI requirements were met (rawpireq)
        # and the value for the raw Gemini Quality Assessment (rawgemqa) from
        # the header of the PHU. The rawpireq and rawgemqa keywords are
        # defined in the local key dictionary (stdkeyDictGEMINI) but are read
        # from the updated global key dictionary (globalStdkeyDict)
        rawpireq = dataset.phuGetKeyValue\
            (globalStdkeyDict['key_raw_pi_requirements_met'])
        rawgemqa = dataset.phuGetKeyValue\
            (globalStdkeyDict['key_raw_gemini_qa'])
        if rawpireq is None or rawgemqa is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Calculate the derived QA state
        ret_qa_state = '%s:%s' % (rawpireq, rawgemqa)
        if rawpireq == 'UNKNOWN' or rawgemqa == 'UNKNOWN':
            ret_qa_state = 'Undefined'
        if rawpireq.upper() == 'YES' and rawgemqa.upper() == 'USABLE':
            ret_qa_state = 'Pass'
        if rawpireq.upper() == 'NO' and rawgemqa.upper() == 'USABLE':
            ret_qa_state = 'Usable'
        if rawgemqa.upper() == 'BAD':
            ret_qa_state = 'Fail'
        if rawpireq.upper() == 'CHECK' or rawgemqa.upper() == 'CHECK':
            ret_qa_state = 'CHECK'
        
        return ret_qa_state
    
    def ut_date(self, dataset, **args):
        # Call ut_datetime(strict=True, dateonly=True) to return a valid
        # ut_date, if possible.
        ret_ut_date = dataset.ut_datetime(strict=True, dateonly=True)
        if ret_ut_date is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        
        return ret_ut_date
    
    def ut_datetime(self, dataset, strict=False, dateonly=False, \
        timeonly=False, **args):
        # First, we try and figure out the date, looping through several
        # header keywords that might tell us. DATE-OBS can also give us a full
        # date-time combination, so we check for this too.
        hdu = dataset.hdulist
        for kw in ['DATE-OBS', globalStdkeyDict['key_ut_date'], 'DATE', \
            'UTDATE']:
            try:
                utdate_hdr = hdu[0].header[kw].strip()
            except KeyError:
                #print "Didn't get a utdate from keyword %s" % kw
                utdate_hdr = ''

            # Validate. The definition is taken from the FITS
            # standard document v3.0. Must be YYYY-MM-DD or
            # YYYY-MM-DDThh:mm:ss[.sss]. Here I also do some very basic checks
            # like ensuring the first digit of the month is 0 or 1, but I
            # don't do cleverer checks like 01<=M<=12. nb. seconds ss > 59 is
            # valid when leap seconds occur.

            # Did we get a full date-time string?
            if re.match('(\d\d\d\d-[01]\d-[0123]\d)(T)([012]\d:[012345]\d:\d\d.*\d*)', utdate_hdr):
                ut_datetime = dateutil.parser.parse(utdate_hdr)
                #print "Got a full date-time from %s: %s = %s" % (kw, utdate_hdr, ut_datetime)
                # If we got a full datetime, then just return it and we're done already!
                return ut_datetime

            # Did we get a date (only) string?
            match = re.match('\d\d\d\d-[01]\d-[0123]\d', utdate_hdr)
            if match:
                #print "got a date from %s: %s" % (kw, utdate_hdr)
                break
            else:
                #print "did not get a date from %s: %s" % (kw, utdate_hdr)
                pass

            # Did we get a *horrible* early niri style string DD/MM/YY[Y] - YYYY = 1900 + YY[Y]?
            match = re.match('(\d\d)/(\d\d)/(\d\d+)', utdate_hdr)
            if(match):
                #print "horrible niri format"
                d = int(match.groups()[0])
                m = int(match.groups()[1])
                y = 1900 + int(match.groups()[2])
                if((d > 0) and (d < 32) and (m > 0) and (m < 13) and (y>1990) and (y<2050)):
                    #print "got a date from horrible niri format"
                    utdate_hdr = "%d-%02d-%02d" % (y, m, d)
                    break
            else:
                utdate_hdr = ''


        # OK, at this point, utdate_hdr should contain either an empty string
        # or a valid date string, ie YYYY-MM-DD

        # If that's all we need, return it now
        if(utdate_hdr and dateonly):
            ut_datetime = dateutil.parser.parse(utdate_hdr+" 00:00:00")
            return ut_datetime.date()

        # Get and validate the ut time header, if present. We try several
        # header keywords that might contain a ut time.
        for kw in [globalStdkeyDict['key_ut_time'], 'UT', 'TIME-OBS', 'STARTUT']:
            try:
                uttime_hdr = hdu[0].header[kw].strip()
            except KeyError:
                #print "Didn't get a uttime from keyword %s" % kw
                uttime_hdr = ''
            # The standard mandates HH:MM:SS[.S...] 
            # OK, we allow single digits to cope with crap data
            # These are somewhat basic checks, it's not completely rigorous
            # Note that seconds can be > 59 when leap seconds occurs
            if re.match('^([012]?\d)(:)([012345]?\d)(:)(\d\d?\.?\d*)$', uttime_hdr):
                #print "Got UT time from keyword %s: %s" % (kw, uttime_hdr)
                break
            else:
                #print "Could not parse a UT time from keyword %s: %s" % (kw, uttime_hdr)
                uttime_hdr = ''
        # OK, at this point, uttime_hdr should contain either an empty string
        # or a valid UT time string, ie HH:MM:SS[.S...]
          
        # If that's all we need, parse it and return it now
        if(uttime_hdr and timeonly):
            ut_datetime = dateutil.parser.parse("2000-01-01 "+uttime_hdr)
            return ut_datetime.time()

        # OK, if we got both a date and a time, then we can go ahead
        # and stick them together then parse them into a datetime object
        if(utdate_hdr and uttime_hdr):
            datetime_str = "%sT%s" % (utdate_hdr, uttime_hdr)
            #print "Got both a utdate and a uttime and made: %s" % datetime_str
            ut_datetime = dateutil.parser.parse(datetime_str)
            return ut_datetime

        # OK, we didn't get good ut_date and ut_time headers, this is non-compliant
        # data, probably taken in engineering mode or something
 
        # Maybe there's an MJD_OBS header we can use
        try:
            #print "Trying to use MJD_OBS"
            mjd = hdu[0].header['MJD_OBS']
            if(mjd > 1):
                # MJD zero is 1858-11-17T00:00:00.000 UTC
                mjdzero = datetime.datetime(1858, 11, 17, 0, 0, 0, 0, None)
                mjddelta = datetime.timedelta(mjd)
                ut_datetime = mjdzero + mjddelta
                #print "determined ut_datetime from MJD_OBS"
                if(dateonly):
                    return ut_datetime.date()
                if(timeonly):
                    return ut_datetime.time()
                return ut_datetime
        except KeyError:
            pass

        # Maybe there's an OBSSTART header we can use
        try:
            #print "Trying to use OBSSTART"
            obsstart = hdu[0].header['OBSSTART'].strip()
            if(obsstart):
                ut_datetime = dateutil.parser.parse(obsstart)
                #print "Did it by OBSSTART"
                if(dateonly):
                    return ut_datetime.date()
                if(timeonly):
                    return ut_datetime.time()
                return ut_datetime
        except KeyError:
            pass

        # OK, now we're getting a desperate. If we're in strict mode, we give up now
        if(strict):
            return None

        # If we didn't get a utdate, can we parse it from the framename header if there is one, or the filename?
        if(not utdate_hdr):
            #print "Desperately trying FRMNAME, filename etc"
            try:
                frmname = hdu[1].header['FRMNAME']
            except (KeyError, ValueError, IndexError):
                frmname = ''
            for string in [frmname, os.path.basename(dataset.filename)]:
                try:
                    #print "... Trying to Parse: %s" % string
                    year = string[1:5]
                    y = int(year)
                    month = string[5:7]
                    m = int(month)
                    day = string[7:9]
                    d = int(day)
                    if(( y > 1999) and (m < 13) and (d < 32)):
                        utdate_hdr = "%s-%s-%s" % (year, month, day)
                        #print "Guessed utdate from %s: %s" % (string, utdate_hdr)
                        break
                except (KeyError, ValueError, IndexError):
                    pass

        # If we didn't get a uttime, assume midnight
        if(not uttime_hdr):
            #print "Assuming midnight. Bleah."
            uttime_hdr = "00:00:00"


        # OK, if all we wanted was a date, and we've got it, return it now
        if(utdate_hdr and dateonly):
            ut_datetime = dateutil.parser.parse(utdate_hdr+" 00:00:00")
            return ut_datetime.date()

        # OK, if all we wanted was a time, and we've got it, return it now
        if(uttime_hdr and timeonly):
            ut_datetime = dateutil.parser.parse("2000-01-01 "+uttime_hdr)
            return ut_datetime.time()

        # If we've got a utdate and a uttime, return it
        if(utdate_hdr and uttime_hdr):
            datetime_str = "%sT%s" % (utdate_hdr, uttime_hdr)
            #print "Got both a utdate and a uttime and made: %s" % datetime_str
            ut_datetime = dateutil.parser.parse(datetime_str)
            return ut_datetime

        # Well, if we get here, we really have no idea
        return None
    
    def ut_time(self, dataset, **args):
        # Call ut_datetime(strict=True, timeonly=True) to return a valid
        # ut_time, if possible.
        ret_ut_time = dataset.ut_datetime(strict=True, timeonly=True)
        if ret_ut_time is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        
        return ret_ut_time
    
    def wavefront_sensor(self, dataset, **args):
        # Get the AOWFS, OIWFS, PWFS1 and PWFS2 probe states (aowfs, oiwfs,
        # pwfs1 and pwfs2, respectively) from the header of the PHU. The probe
        # states keywords are defined in the local key dictionary
        # (stdkeyDictGEMINI) but are read from the updated global key
        # dictionary (globalStdkeyDict)
        aowfs = dataset.phuGetKeyValue(globalStdkeyDict['key_aowfs'])
        oiwfs = dataset.phuGetKeyValue(globalStdkeyDict['key_oiwfs'])
        pwfs1 = dataset.phuGetKeyValue(globalStdkeyDict['key_pwfs1'])
        pwfs2 = dataset.phuGetKeyValue(globalStdkeyDict['key_pwfs2'])
        if aowfs is None or oiwfs is None or pwfs1 is None or pwfs2 is None:
            # The phuGetKeyValue() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # If any of the probes are guiding, add them to the list
        wavefront_sensors = []
        if aowfs == 'guiding':
            wavefront_sensors.append('AOWFS')
        if oiwfs == 'guiding':
            wavefront_sensors.append('OIWFS')
        if pwfs1 == 'guiding':
            wavefront_sensors.append('PWFS1')
        if pwfs2 == 'guiding':
            wavefront_sensors.append('PWFS2')
        if len(wavefront_sensors) == 0:
            # If no probes are guiding, raise an exception
            raise Errors.CalcError()
        else:
            # Return a unique, sorted, wavefront sensor identifier string with
            # an ampersand separating each wavefront sensor name
            wavefront_sensors.sort
            ret_wavefront_sensor = str('&'.join(wavefront_sensors))
        
        return ret_wavefront_sensor
    
    def wavelength_reference_pixel(self, dataset, **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_wavelength_reference_pixel = {}
        # Loop over the science extensions in the dataset
        for ext in dataset['SCI']:
            # Get the reference pixel of the central wavelength from the header
            # of each pixel data extension. The reference pixel of the central
            # wavelength keyword may be defined in a local key dictionary
            # (stdkeyDict<INSTRUMENT>) but is read from the updated global key
            # dictionary (globalStdkeyDict)
            wavelength_reference_pixel = ext.getKeyValue\
                (globalStdkeyDict['key_wavelength_reference_pixel'])
            if wavelength_reference_pixel is None:
                # The getKeyValue() function returns None if a value cannot be
                # found and stores the exception info. Re-raise the exception.
                # It will be dealt with by the CalculatorInterface.
                if hasattr(ext, 'exception_info'):
                    raise ext.exception_info
            # Return a dictionary with the reference pixel of the central
            # wavelength float as the value
            ret_wavelength_reference_pixel.update({(ext.extname(), \
                ext.extver()):float(wavelength_reference_pixel)})
        
        return ret_wavelength_reference_pixel
    
    def well_depth_setting(self, dataset, **args):
        # The well_depth_setting descriptor is only specific to GNIRS and NIRI
        # data. The code below will be replaced with the GNIRS or NIRI specific
        # well_depth_setting descriptor function located in the instrument
        # specific descriptor files. For all other Gemini data, raise an
        # exception if this descriptor is called.
        raise Errors.ExistError()
