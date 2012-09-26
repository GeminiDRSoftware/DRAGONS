import sys
from astrodata import Descriptors
from astrodata.Descriptors import DescriptorValue
from astrodata import Errors

class CalculatorInterface:

    descriptor_calculator = None

    def instrument(self, format=None, **args):
        """
        Return the instrument value
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the instrument used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_"+"instrument"
            #print "mkCI22:",key, repr(keydict)
            #print "mkCI23:", key in keydict
            if key in keydict.keys():
                keyword = keydict[key]
            else:
                keyword = None
            #print hasattr(self.descriptor_calculator, "instrument")
            if not hasattr(self.descriptor_calculator, "instrument"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info
                else:
                    msg = "Unable to find an appropriate descriptor function "
                    msg += "or a default keyword for instrument"
                    raise KeyError(msg)
            else:
                retval = self.descriptor_calculator.instrument(self, **args)
            
            
            ret = DescriptorValue( retval, 
                                   format = format, 
                                   name = "instrument",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = str )
            return ret
        except:
            if not hasattr(self, "exception_info"):
                setattr(self, "exception_info", sys.exc_info()[1])
            if (self.descriptor_calculator is None 
                or self.descriptor_calculator.throwExceptions == True):
                raise
            else:
                #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()[1]
                return None
    
    def object(self, format=None, **args):
        """
        Return the object value
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the name of the target object observed
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_"+"object"
            #print "mkCI22:",key, repr(keydict)
            #print "mkCI23:", key in keydict
            if key in keydict.keys():
                keyword = keydict[key]
            else:
                keyword = None
            #print hasattr(self.descriptor_calculator, "object")
            if not hasattr(self.descriptor_calculator, "object"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info
                else:
                    msg = "Unable to find an appropriate descriptor function "
                    msg += "or a default keyword for object"
                    raise KeyError(msg)
            else:
                retval = self.descriptor_calculator.object(self, **args)
            
            
            ret = DescriptorValue( retval, 
                                   format = format, 
                                   name = "object",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = str )
            return ret
        except:
            if not hasattr(self, "exception_info"):
                setattr(self, "exception_info", sys.exc_info()[1])
            if (self.descriptor_calculator is None 
                or self.descriptor_calculator.throwExceptions == True):
                raise
            else:
                #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()[1]
                return None
    
    def ut_date(self, format=None, **args):
        """
        Return the ut_date value
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: datetime as default (i.e., format=None)
        :return: the UT date at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_"+"ut_date"
            #print "mkCI22:",key, repr(keydict)
            #print "mkCI23:", key in keydict
            if key in keydict.keys():
                keyword = keydict[key]
            else:
                keyword = None
            #print hasattr(self.descriptor_calculator, "ut_date")
            if not hasattr(self.descriptor_calculator, "ut_date"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info
                else:
                    msg = "Unable to find an appropriate descriptor function "
                    msg += "or a default keyword for ut_date"
                    raise KeyError(msg)
            else:
                retval = self.descriptor_calculator.ut_date(self, **args)
            
            from datetime import datetime
            ret = DescriptorValue( retval, 
                                   format = format, 
                                   name = "ut_date",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = datetime )
            return ret
        except:
            if not hasattr(self, "exception_info"):
                setattr(self, "exception_info", sys.exc_info()[1])
            if (self.descriptor_calculator is None 
                or self.descriptor_calculator.throwExceptions == True):
                raise
            else:
                #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()[1]
                return None
    
# UTILITY FUNCTIONS, above are descriptor thunks            
    def _lazyloadCalculator(self, **args):
        '''Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed.'''
        if self.descriptor_calculator is None:
            self.descriptor_calculator = Descriptors.get_calculator(self, **args)

