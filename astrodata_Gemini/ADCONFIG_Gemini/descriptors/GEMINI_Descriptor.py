from astrodata import Lookups
from astrodata import Descriptors
from astrodata import Errors

from astrodata.Calculator import Calculator

import GemCalcUtil

import re
import datetime
import dateutil.parser
import os

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardGEMINIKeyDict import stdkeyDictGEMINI
from Generic_Descriptor import Generic_DescriptorCalc

class GEMINI_DescriptorCalc(Generic_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    globalStdkeyDict.update(stdkeyDictGEMINI)

    def airmass(self, dataset, **args):
        # Get the airmass value from the header of the PHU
        hdu = dataset.hdulist
        airmass = hdu[0].header[globalStdkeyDict['key_airmass']]

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
        hdu = dataset.hdulist
        crpa = hdu[0].header[globalStdkeyDict['key_cass_rotator_pa']]

        # Validate the cassegrain rotator position angle value
        if (crpa < -360.0) or (crpa > 360.0):
            raise Errors.InvalidValueError()
        else:
            ret_crpa = float(crpa)

        return ret_crpa
    
    def central_wavelength(self, dataset, asMicrometers=False, \
        asNanometers=False, asAngstroms=False, asDict=False, **args):
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
        
        if asDict:
            # This is used when obtaining the central wavelength from processed
            # data (when the keyword will be in the pixel data extensions)
            return 'asDict for central_wavelength not yet implemented'
        else:
            # Get the central wavelength value from the header of the PHU.
            hdu = dataset.hdulist
            raw_central_wavelength = \
                float(hdu[0].header[globalStdkeyDict['key_central_wavelength']])
            # Use the utilities function convert_units to convert the central
            # wavelength value from the input units to the output units
            ret_central_wavelength = \
                GemCalcUtil.convert_units(input_units=input_units, \
                input_value=raw_central_wavelength, output_units=output_units)
        
        return ret_central_wavelength
    
    def data_section(self, dataset, pretty=False, asDict=True, **args):
        if asDict:
            ret_data_section = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Get the data section from the header of each pixel data
                # extension
                raw_data_section = \
                    ext.header[globalStdkeyDict['key_data_section']]
                if pretty:
                    # Return a dictionary with the data section string that
                    # uses 1-based indexing as the value
                    ret_data_section.update({(ext.extname(), \
                        ext.extver()):str(raw_data_section)})
                else:
                    # Return a dictionary with the data section tuple that
                    # uses 0-based indexing as the value
                    data_section = \
                        GemCalcUtil.section_to_tuple(raw_data_section)
                    ret_data_section.update({(ext.extname(), \
                        ext.extver()):data_section})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Get the data section from the header of the single pixel
                # data extension
                hdu = dataset.hdulist
                raw_data_section = \
                    hdu[1].header[globalStdkeyDict['key_data_section']]
                if pretty:
                    # Return the data section string that uses 1-based indexing
                    ret_data_section = raw_data_section
                else:
                    # Return the data section tuple that uses 0-based indexing
                    data_section = \
                        GemCalcUtil.section_to_tuple(raw_data_section)
                    ret_data_section = data_section
            else:
                raise Errors.DescriptorDictError()
        
        return ret_data_section
    
    def decker(self, dataset, stripID=False, pretty=False, **args):
        """
        In GNIRS, the decker is used to basically mask off the ends of the
        slit to create the short slits used in the cross dispersed modes.
        """
        # Get the decker position from the header of the PHU
        hdu = dataset.hdulist
        decker = hdu[0].header[globalStdkeyDict['key_decker']]
        
        if pretty:
            stripID = True
        
        if stripID:
            # Strip the component ID from the decker position value
            ret_decker = GemCalcUtil.removeComponentID(decker)
        
        return ret_decker
    
    def detector_section(self, dataset, pretty=False, asDict=True, **args):
        if asDict:
            ret_detector_section = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Get the detector section from the header of each pixel data
                # extension
                raw_detector_section = \
                    ext.header[globalStdkeyDict['key_detector_section']]
                if pretty:
                    # Return a dictionary with the detector section string
                    # that uses 1-based indexing as the value
                    ret_detector_section.update({(ext.extname(), \
                        ext.extver()):str(raw_detector_section)})
                else:
                    # Return a dictionary with the detector section tuple that
                    # uses 0-based indexing as the value
                    detector_section = \
                        GemCalcUtil.section_to_tuple(raw_detector_section)
                    ret_detector_section.update({(ext.extname(), \
                        ext.extver()):detector_section})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Get the detector section from the header of the single pixel
                # data extension
                hdu = dataset.hdulist
                raw_detector_section = \
                    hdu[1].header[globalStdkeyDict['key_detector_section']]
                if pretty:
                    # Return the detector section string that uses 1-based
                    # indexing
                    ret_detector_section = raw_detector_section
                else:
                    # Return the detector section tuple that uses 0-based
                    # indexing
                    detector_section = \
                        GemCalcUtil.section_to_tuple(raw_detector_section)
                    ret_detector_section = detector_section
            else:
                raise Errors.DescriptorDictError()
        
        return ret_detector_section
    
    def detector_x_bin(self, dataset, asDict=True, **args):
        if asDict:
            ret_detector_x_bin = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Return a dictionary with the binning of the x-axis integer
                # (set to 1 as default for Gemini data) as the value
                ret_detector_x_bin.update({(ext.extname(), \
                    ext.extver()):int(1)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Return the binning of the x-axis integer
                hdu = dataset.hdulist
                ret_detector_x_bin = int(1)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_detector_x_bin
    
    def detector_y_bin(self, dataset, asDict=True, **args):
        if asDict:
            ret_detector_y_bin = {}
            # Loop over the science extensions
            for ext in dataset['SCI']:
                # Return a dictionary with the binning of the x-axis integer
                # (set to 1 as default for Gemini data) as the value
                ret_detector_y_bin.update({(ext.extname(), \
                    ext.extver()):int(1)})
        else:
            # Check to see whether the dataset has a single extension and if
            # it does, return a single value
            if dataset.countExts('SCI') <= 1:
                # Return the binning of the x-axis integer
                hdu = dataset.hdulist
                ret_detector_y_bin = int(1)
            else:
                raise Errors.DescriptorDictError()
        
        return ret_detector_y_bin

    def dispersion_axis(self, dataset, asDict=True, **args):
        # The dispersion axis can only be obtained from data that does not
        # have an AstroData Type of IMAGE and that has been prepared (since
        # the dispersion axis keyword is written during the prepare step)
        if 'IMAGE' not in dataset.types and 'PREPARED' in dataset.types:
            if asDict:
                ret_dispersion_axis = {}
                # Loop over the science extensions
                for ext in dataset['SCI']:
                    # Get the dispersion axis from the header of each pixel
                    # data extension
                    dispersion_axis = \
                        ext.header[globalStdkeyDict['key_dispersion_axis']]
                    # Return a dictionary with the dispersion axis integer as
                    # the value
                    ret_dispersion_axis.update({(ext.extname(), \
                        ext.extver()):int(dispersion_axis)})
            else:
                # Check to see whether the dataset has a single extension and 
                # if it does, return a single value
                if dataset.countExts('SCI') <= 1:
                    # Get the dispersion axis from the header of the
                    # single pixel data extension
                    hdu = dataset.hdulist
                    dispersion_axis = \
                        hdu[1].header[globalStdkeyDict['key_dispersion_axis']]
                    # Return the dispersion axis integer
                    ret_dispersion_axis = int(dispersion_axis)
                else:
                    raise Errors.DescriptorDictError()
        else:
            raise Errors.DescriptorTypeError()
        
        return ret_dispersion_axis
    
    def local_time(self, dataset, **args):
        # Get the local time from the header of the PHU
        hdu = dataset.hdulist
        local_time = hdu[0].header[globalStdkeyDict['key_local_time']]
        
        # Validate the local time value. The assumption is that the standard
        # mandates HH:MM:SS[.S]. We don't enforce the number of decimal places.
        # These are somewhat basic checks, it's not completely rigorous. Note
        # that seconds can be > 59 when leap seconds occurs        
        if re.match('^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$', local_time):
            ret_local_time = str(local_time)
        else:
            raise Errors.InvalidValueError()
        
        return ret_local_time
    
    def qa_state(self, dataset, **args):
        # Get the value for whether the PI requirements were met (rawpireq)
        # and the value for the raw Gemini Quality Assessment (rawgemqa) from
        # the header of the PHU. The rawpireq and rawgemqa keywords are
        # defined in the local key dictionary (stdkeyDictGEMINI) but are read
        # from the updated global key dictionary (globalStdkeyDict)
        hdu = dataset.hdulist
        rawpireq = \
            hdu[0].header[globalStdkeyDict['key_raw_pi_requirements_met']]
        rawgemqa = hdu[0].header[globalStdkeyDict['key_raw_gemini_qa']]
        
        # Calculate the derived QA state
        qa_state = "%s:%s" % (rawpireq, rawgemqa)
        if rawpireq == 'UNKNOWN' or rawgemqa == 'UNKNOWN':
            qa_state = 'Undefined'
        if rawpireq.upper() == 'YES' and rawgemqa.upper() == 'USABLE':
            qa_state = 'Pass'
        if rawpireq.upper() == 'NO' and rawgemqa.upper() == 'USABLE':
            qa_state = 'Usable'
        if rawgemqa.upper() == 'BAD':
            qa_state = 'Fail'
        if rawpireq.upper() == 'CHECK' or rawgemqa.upper() == 'CHECK':
            qa_state = 'CHECK'
        
        return qa_state
    
    def ut_date(self, dataset, **args):
        # Call ut_datetime(strict=True, dateonly=True) to return a valid
        # ut_date, if possible.
        return dataset.ut_datetime(strict=True, dateonly=True)

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
        return self.ut_datetime(dataset, strict=True, timeonly=True)
    
    def wavefront_sensor(self, dataset, **args):
        # Get the AOWFS, OIWFS, PWFS1 and PWFS2 probe states (aowfs, oiwfs,
        # pwfs1 and pwfs2, respectively) from the header of the PHU
        hdu = dataset.hdulist
        aowfs = hdu[0].header[stdkeyDictGEMINI['key_aowfs']]
        oiwfs = hdu[0].header[stdkeyDictGEMINI['key_oiwfs']]
        pwfs1 = hdu[0].header[stdkeyDictGEMINI['key_pwfs1']]
        pwfs2 = hdu[0].header[stdkeyDictGEMINI['key_pwfs2']]
        
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

