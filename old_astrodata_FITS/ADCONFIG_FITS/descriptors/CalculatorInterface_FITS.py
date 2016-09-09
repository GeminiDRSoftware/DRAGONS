import sys

from astrodata.utils import Errors
from astrodata.interface import Descriptors
from astrodata.interface.Descriptors import DescriptorValue

class CalculatorInterface(object):
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
            key = "key_instrument"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "instrument"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'instrument'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.instrument(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "instrument",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = str )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "instrument",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

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
            key = "key_object"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "object"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'object'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.object(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "object",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = str )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "object",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def telescope(self, format=None, **args):
        """
        Return the telescope value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the telescope used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_telescope"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "telescope"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'telescope'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.telescope(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "telescope",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = str )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "telescope",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

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
            key = "key_ut_date"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "ut_date"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'ut_date'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.ut_date(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            from datetime import datetime
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "ut_date",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = datetime )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "ut_date",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    # UTILITY FUNCTIONS, above are descriptor function buffers
    def _lazyloadCalculator(self, **args):
        '''Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed.'''
        if self.descriptor_calculator is None:
            self.descriptor_calculator = Descriptors.get_calculator(self, **args)

