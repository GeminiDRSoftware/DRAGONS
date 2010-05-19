from astrodata import Lookups
from astrodata import Descriptors

from astrodata.Calculator import Calculator

import GemCalcUtil
import re

from StandardGenericKeyDict import stdkeyDictGeneric

class Generic_DescriptorCalc(Calculator):
    
    def utdate(self, dataset, **args):
        """
        Return the UT date value for generic data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the UT date of the observation (YYYY-MM-DD)
        """
        try:
            hdu = dataset.hdulist
            utdatestring = hdu[0].header[stdkeyDictGeneric["key_generic_utdate"]]

        except KeyError:
            return None
        
        # Validate the result. The definition is taken from the FITS standard document v3.0
        # Must be YYYY-MM-DD or YYYY-MM-DDThh:mm:ss[.sss]

        # Here I also do some very basic checks like ensuring the first digit of the
        # month is 0 or 1, but I don't do cleverer checks like 01<=M<=12

        # nb. seconds ss > 59 is valid when leap seconds occur.

        retval = None

        if(re.match('\d\d\d\d-[01]\d-[0123]\d', utdatestring)):
            retval = utdatestring

        m=re.match('(\d\d\d\d-[01]\d-[0123]\d)(T)([012]\d:[012345]\d:\d\d.*\d*)', utdatestring)
        if(m):
            retval = m.group(1)

        return retval
    
    def instrument(self, dataset, **args):
        """
        Return the instrument value for generic data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the instrument used to acquire the data
        """
        try:
            hdu = dataset.hdulist
            retinstrumentstring = hdu[0].header[stdkeyDictGeneric["key_generic_instrument"]]
                    
        except KeyError:
            return None
        
        return str(retinstrumentstring)
    
    def object(self, dataset, **args):
        """
        Return the object value for generic data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the name of the object acquired
        """
        try:
            hdu = dataset.hdulist
            retobjectstring = hdu[0].header[stdkeyDictGeneric["key_generic_object"]]

        except KeyError:
            return None
        
        return str(retobjectstring)
    
    def observer(self, dataset, **args):
        """
        Return the observer value for generic data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the observer who acquired the observation
        """
        try:
            hdu = dataset.hdulist
            retobserverstring = hdu[0].header[stdkeyDictGeneric["key_generic_observer"]]

        except KeyError:
            return None
        
        return str(retobserverstring)
    
    def telescope(self, dataset, **args):
        """
        Return the telescope value for generic data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the telescope where the observation was taken
        """
        try:
            hdu = dataset.hdulist
            rettelescopestring = hdu[0].header[stdkeyDictGeneric["key_generic_telescope"]]

        except KeyError:
            return None
        
        return str(rettelescopestring)
