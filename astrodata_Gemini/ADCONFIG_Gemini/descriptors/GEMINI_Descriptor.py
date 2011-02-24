from astrodata import Lookups
from astrodata import Descriptors

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
    # Updating the global key dict with the local dict of this descriptor class
    globalStdkeyDict.update(stdkeyDictGEMINI)

    def airmass(self, dataset, **args):
        """
        Return the airmass value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the mean airmass for the observation
        """
        hdu = dataset.hdulist
        airmass = hdu[0].header[globalStdkeyDict['key_airmass']]
        
        if airmass < 0.0:
            ret_airmass = None
        else:
            ret_airmass = float(airmass)
        
        return ret_airmass
    
    def cass_rotator_pa(self, dataset, **args):
        """
        Return the cassegrain rotator position angle value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the crpa for the observation
        """
        hdu = dataset.hdulist
        crpa = hdu[0].header[globalStdkeyDict['key_cass_rotator_pa']]

        if (crpa < -999.0) or (crpa > 999.0):
            ret_crpa = None
        else:
            ret_crpa = float(crpa)

        return ret_crpa
    
    def detector_x_bin(self, dataset, **args):
        """
        Return the detector_x_bin value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the binning of the detector x-axis
        """
        ret_detector_x_bin = int(1)
        
        return ret_detector_x_bin
    
    def detector_y_bin(self, dataset, **args):
        """
        Return the detector_y_bin value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the binning of the detector y-axis
        """
        ret_detector_y_bin = int(1)
        
        return ret_detector_y_bin
    
    def local_time(self, dataset, **args):
        """
        Return the local_time value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the local time at the start of the observation (HH:MM:SS.S)
        """
        hdu = dataset.hdulist
        local_time = hdu[0].header[globalStdkeyDict['key_local_time']]
        
        # Validate the result.
        # The assumption is that the standard mandates HH:MM:SS[.S] 
        # We don't enforce the number of decimal places
        # These are somewhat basic checks, it's not completely rigorous
        # Note that seconds can be > 59 when leap seconds occurs
        
        if re.match('^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$', local_time):
            ret_local_time = str(local_time)
        else:
            ret_local_time = None
        
        return ret_local_time
    
    def qa_state(self, dataset, **args):
        """
        Return the qa_state for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the QA state for the observation
        """
        hdu = dataset.hdulist
        rawpireq = hdu[0].header[stdkeyDictGEMINI['key_raw_pi_requirements_met']]
        rawgemqa = hdu[0].header[stdkeyDictGEMINI['key_raw_gemini_qa']]
        
        # Calculate the derived QA state
        qa_state = "%s:%s" % (rawpireq, rawgemqa)
        if rawpireq == 'UNKNOWN' and rawgemqa == 'UNKNOWN':
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
    
    def ut_datetime(self, dataset, **args):
        """
        Return the ut_datetime value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: datetime.datetime
        @returns: the UT date and time at the start of the observation

        This descriptor attempts to figure out the datetime
        even when headers are malformed or not present. It tries just
        about every header combination that could allow it to determine
        an appropriate datetime for the file in question.
        This makes it somewhat specific to Gemini data, in that the headers
        it looks at, and the assumptions it makes in trying to parse thier
        values, are those known to ocurr in Gemini data. Note that some of
        the early gemini data, and that taken from lower level engineering
        interfaces, lack standard headers. Also the format and occurence of
        various headers has changed over time, even on the same instrument.
        
        """
        hdu = dataset.hdulist
 
        # First, we try and figure out the date, looping through several
        # header keywords that might tell us. DATE-OBS can also give us a full
        # date-time combination, so we check for this too.
        for kw in ['DATE-OBS', globalStdkeyDict['key_ut_date'], 'DATE', 'UTDATE']:
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
                utdate_hdr = ''

        # OK, at this point, utdate_hdr should contain either an empty string
        # or a valid date string, ie YYYY-MM-DD

        # Get and validate the ut time header, if present. We try several
        # header keywords that might contain a ut time.
        for kw in [globalStdkeyDict['key_ut_time'], 'UT', 'TIME-OBS']:
            try:
                uttime_hdr = hdu[0].header[kw].strip()
            except KeyError:
                #print "Didn't get a uttime from keyword %s" % kw
                uttime_hdr = ''
            # The standard mandates HH:MM:SS[.S...] 
            # These are somewhat basic checks, it's not completely rigorous
            # Note that seconds can be > 59 when leap seconds occurs
            if re.match('^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$', uttime_hdr):
                #print "Got UT time from keyword %s: %s" % (kw, uttime_hdr)
                break
            else:
                #print "Could not parse a UT time from keyword %s: %s" % (kw, uttime_hdr)
                uttime_hdr = ''
        # OK, at this point, uttime_hdr should contain either an empty string
        # or a valid UT time string, ie HH:MM:SS[.S...]
          
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
            mjd = hdu[0].header['MJD_OBS']
            if(mjd > 1):
                mjdzero = datetime.datetime(1858, 11, 17, 0, 0, 0, 0, None)
                mjddelta = datetime.timedelta(mjd)
                ut_datetime = mjdzero + mjddelta
                #print "determined ut_datetime from MJD_OBS"
                return ut_datetime
        except KeyError:
            pass

        # Maybe there's an OBSSTART header we can use
        try:
            obsstart = hdu[0].header['OBSSTART'].strip()
            if(obsstart):
                ut_datetime = dateutil.parser.parse(obsstart)
                #print "Did it by OBSSTART"
                return ut_datetime
        except KeyError:
            pass

        # OK, now we're getting desperate

        # If we didn't get a utdate, can we parse it from the framename header if there is one, or the filename?
        if(not utdate_hdr):
            try:
                for string in [hdu[1].header['FRMNAME'], os.path.basename(dataset.filename)]:
                    year = string[1:5]
                    month = string[5:7]
                    day = string[7:9]
                    uddate_hdr = "%s-%s-%s" % (year, month, day)
                    #print "Guessed utdate from %s: %s" % (string, utdate_hdr)
                    break
            except (KeyError, ValueError, IndexError):
                pass

        # If we didn't get a uttime, assume midnight
        if(not uttime_hdr):
            #print "Assuming midnight. Bleah."
            uttime_hdr = "00:00:00"

        # Again, if we've got a utdate and a ut
        if(utdate_hdr and uttime_hdr):
            datetime_str = "%sT%s" % (utdate_hdr, uttime_hdr)
            #print "Got both a utdate and a uttime and made: %s" % datetime_str
            ut_datetime = dateutil.parser.parse(datetime_str)
            return ut_datetime

        # Well, if we get here, we really have no idea
        return None


    def ut_time(self, dataset, **args):
        """
        Return the ut_time value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT at the start of the observation (HH:MM:SS.S)
        """
        hdu = dataset.hdulist
        ut_time = hdu[0].header[globalStdkeyDict['key_ut_time']]
        
        # Validate the result.
        # The assumption is that the standard mandates HH:MM:SS[.S] 
        # We don't enforce the number of decimal places
        # These are somewhat basic checks, it's not completely rigorous
        # Note that seconds can be > 59 when leap seconds occurs
        
        if re.match('^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$', ut_time):
            ret_ut_time = str(ut_time)
        else:
            ret_ut_time = None
        
        return ret_ut_time
    
    def wavefront_sensor(self, dataset, **args):
        """
        Return the wavefront_sensor value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the wavefront sensor used for the observation
        """
        hdu = dataset.hdulist
        aowfs = hdu[0].header[stdkeyDictGEMINI['key_aowfs']]
        oiwfs = hdu[0].header[stdkeyDictGEMINI['key_oiwfs']]
        pwfs1 = hdu[0].header[stdkeyDictGEMINI['key_pwfs1']]
        pwfs2 = hdu[0].header[stdkeyDictGEMINI['key_pwfs2']]
        
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
            ret_wavefront_sensor = None
        else:
            ret_wavefront_sensor = str('&'.join(wavefront_sensors))
        
        return ret_wavefront_sensor

