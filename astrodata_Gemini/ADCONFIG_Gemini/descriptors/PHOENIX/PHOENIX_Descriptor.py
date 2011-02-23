import re

from astrodata import Lookups
from astrodata import Descriptors

import astrodata
from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardPHOENIXKeyDict import stdkeyDictPHOENIX
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class PHOENIX_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dict with the local dict of this descriptor class
    globalStdkeyDict.update(stdkeyDictPHOENIX)
    
    def __init__(self):
        pass
    
    def dec(self, dataset, **args):
        """
        Return the dec value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the declination in decimal degrees
        """
        hdu = dataset.hdulist
        dec = hdu[0].header[globalStdkeyDict['key_dec']]
        ret_dec = float(GemCalcUtil.degsextodec(dec))
        
        return ret_dec
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        hdu = dataset.hdulist
        exptime = hdu[0].header[globalStdkeyDict['key_exposure_time']]
        coadds = dataset.coadds()
        
        ret_exposure_time = float(exptime * coadds)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the filter_name value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID from the
        returned filter name
        @param pretty: set to True to return a meaningful filter name
        @rtype: string
        @return: the unique filter identifier string
        """
        hdu = dataset.hdulist
        filter_name = hdu[0].header[stdkeyDictPHOENIX['key_filter']]
        
        ret_filter_name = str(filter_name)
        
        return ret_filter_name
    
    def ra(self, dataset, **args):
        """
        Return the ra value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the R.A. in decimal degrees
        """
        hdu = dataset.hdulist
        ra = hdu[0].header[globalStdkeyDict['key_ra']]
        ret_ra = float(GemCalcUtil.rasextodec(ra))
        
        return ret_ra
    
    def ut_date(self, dataset, **args):
        """
        Return the ut_date value for PHOENIX
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the UT date of the observation (YYYY-MM-DD)
        """
        hdu = dataset.hdulist
        ut_date = hdu[0].header[stdkeyDictPHOENIX['key_ut_date']]
        
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
