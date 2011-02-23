from astrodata import Lookups
from astrodata import Descriptors
import re

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardMICHELLEKeyDict import stdkeyDictMICHELLE
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class MICHELLE_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dict with the local dict of this descriptor class
    globalStdkeyDict.update(stdkeyDictMICHELLE)
    
    def __init__(self):
        pass
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        hdu = dataset.hdulist
        central_wavelength = \
            hdu[0].header[stdkeyDictMICHELLE['key_central_wavelength']]
        
        ret_central_wavelength = float(central_wavelength)
        if(ret_central_wavelength < 0.0):
            ret_central_wavelength = None
        
        return ret_central_wavelength
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        hdu = dataset.hdulist
        exposure = float(hdu[0].header[stdkeyDictMICHELLE['key_exposure']])
        numexpos = float(hdu[0].header[stdkeyDictMICHELLE['key_coadds']])
        numext = float(hdu[0].header[stdkeyDictMICHELLE['key_numext']])
        
        ret_exposure_time = float(exposure * numexpos * numext)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the filter_name value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID from the
        returned filter name
        @param pretty: set to True to return a meaningful filter name
        @rtype: string
        @return: the unique filter identifier string
        """
        # The Michelle filters don't have ID strings, so we just ignore the
        # stripID and pretty options
        hdu = dataset.hdulist
        filter = hdu[0].header[stdkeyDictMICHELLE['key_filter']]
        
        if filter == 'NBlock':
            ret_filter_name = 'blank'
        else:
            ret_filter_name = str(filter)
        
        return ret_filter_name
    
    def ut_date(self, dataset, **args):
        """
        Return the ut_date value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT date of the observation (YYYY-MM-DD)
        """
        hdu = dataset.hdulist
        ut_date = hdu[0].header[stdkeyDictMICHELLE['key_ut_date']]
        
        # Validate the result. The definition is taken from the FITS
        # standard document v3.0. Must be YYYY-MM-DD or
        # YYYY-MM-DDThh:mm:ss[.sss]. Here I also do some very basic checks
        # like ensuring the first digit of the month is 0 or 1, but I
        # don't do cleverer checks like 01<=M<=12. nb. seconds ss > 59 is
        # valid when leap seconds occur.
        
        match1 = re.match('\d\d\d\d-[01]\d-[0123]\d', ut_date)
        match2 = re.match('(\d\d\d\d-[01]\d-[0123]\d)(T)([012]\d:[012345]\d:\d\d.*\d*)', ut_date)
        
        if match1:
            ret_ut_date = str(ut_date)
        elif match2:
            ret_ut_date = str(match2.group(1))
        else:
            ret_ut_date = None
        
        return ret_ut_date

