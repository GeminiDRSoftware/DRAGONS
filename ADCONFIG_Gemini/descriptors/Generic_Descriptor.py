from astrodata import Lookups
from astrodata import Descriptors

from astrodata.Calculator import Calculator

import GemCalcUtil

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
            retutdatestring = hdu[0].header[stdkeyDictGeneric["key_generic_utdate"]]

        except KeyError:
            return None
        
        return str(retutdatestring)
    
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
