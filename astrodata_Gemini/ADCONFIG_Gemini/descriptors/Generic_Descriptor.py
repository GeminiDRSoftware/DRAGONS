from astrodata import Lookups
from astrodata import Descriptors

from astrodata.Calculator import Calculator

import GemCalcUtil
import re
from StandardGenericKeyDict import stdkeyDictGeneric

class Generic_DescriptorCalc(Calculator):
    # Updating the global key dict with the local dict of this descriptor class
    _udpate_stdkey_dict = stdkeyDictGeneric
    
    def ut_date(self, dataset, **args):
        """
        Return the ut_date value for generic data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the UT date of the observation (YYYY-MM-DD)
        """
        hdu = dataset.hdulist
        ut_date = hdu[0].header[self._specifickey_dict['key_ut_date']]
        
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
