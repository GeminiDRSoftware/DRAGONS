import sys
from astrodata.interface import Descriptors
from astrodata.interface.Descriptors import DescriptorValue
from astrodata.utils import Errors

class CalculatorInterface(object):

    descriptor_calculator = None

    def airmass(self, format=None, **args):
        """
        Return the airmass value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the mean airmass of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_airmass"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "airmass"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'airmass'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.airmass(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "airmass",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "airmass",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def ao_seeing(self, format=None, **args):
        """
        Return the AO-estimated seeing
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the AO-estimated seeing of the observation in arcseconds       
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_ao_seeing"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "ao_seeing"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'ao_seeing'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.ao_seeing(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "ao_seeing",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "ao_seeing",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def amp_read_area(self, format=None, **args):
        """
        Return the amp_read_area value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: string as default (i.e., format=None)
        :rtype: dictionary containing one or more string(s) (format=as_dict)
        :return: the composite string containing the name of the array
                 amplifier and the readout area of the array used for the
                 observation 
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_amp_read_area"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "amp_read_area"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'amp_read_area'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.amp_read_area(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "amp_read_area",
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
                                       name = "amp_read_area",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def array_name(self, format=None, **args):
        """
        Return the array_name value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: string as default (i.e., format=None)
        :rtype: dictionary containing one or more string(s) (format=as_dict)
        :return: the name of each array used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_array_name"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "array_name"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'array_name'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.array_name(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "array_name",
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
                                       name = "array_name",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def array_section(self, format=None, **args):
        """
        Return the array_section value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful array_section 
                       value in the form [x1:x2,y1:y2] that uses 1-based 
                       indexing
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: list of integers that uses 0-based indexing in the form 
                [x1 - 1, x2 - 1, y1 - 1, y2 - 1] as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2]
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the unbinned section of the array that was used to observe the
                 data
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_array_section"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "array_section"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'array_section'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.array_section(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "array_section",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = list )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "array_section",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def azimuth(self, format=None, **args):
        """
        Return the azimuth value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the azimuth (in degrees between 0 and 360) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_azimuth"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "azimuth"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'azimuth'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.azimuth(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "azimuth",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "azimuth",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def camera(self, format=None, **args):
        """
        Return the camera value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the camera used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_camera"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "camera"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'camera'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.camera(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "camera",
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
                                       name = "camera",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def cass_rotator_pa(self, format=None, **args):
        """
        Return the cass_rotator_pa value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the cassegrain rotator position angle (in degrees between -360
                 and 360) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_cass_rotator_pa"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "cass_rotator_pa"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'cass_rotator_pa'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.cass_rotator_pa(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "cass_rotator_pa",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "cass_rotator_pa",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def central_wavelength(self, format=None, **args):
        """
        Return the central_wavelength value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param asMicrometers: set to True to return the central_wavelength 
                              value in units of Micrometers
        :type asMicrometers: Python boolean
        :param asNanometers: set to True to return the central_wavelength 
                             value in units of Nanometers
        :type asNanometers: Python boolean
        :param asAngstroms: set to True to return the central_wavelength 
                            value in units of Angstroms
        :type asAngstroms: Python boolean
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s)
        :return: the central wavelength (in meters as default) of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_central_wavelength"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "central_wavelength"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'central_wavelength'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.central_wavelength(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "central_wavelength",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "central_wavelength",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def coadds(self, format=None, **args):
        """
        Return the coadds value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the number of coadds used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_coadds"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "coadds"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'coadds'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.coadds(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "coadds",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "coadds",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def data_label(self, format=None, **args):
        """
        Return the data_label value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the unique identifying name (e.g., GN-2003A-C-2-52-003) of the
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_data_label"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "data_label"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'data_label'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.data_label(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "data_label",
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
                                       name = "data_label",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def data_section(self, format=None, **args):
        """
        Return the data_section value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful data_section 
                       value in the form [x1:x2,y1:y2] that uses 1-based 
                       indexing
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: list of integers that uses 0-based indexing in the form 
                [x1 - 1, x2 - 1, y1 - 1, y2 - 1] as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2]
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the section of the pixel data extensions that contains the
                 data observed
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_data_section"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "data_section"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'data_section'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.data_section(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "data_section",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = list )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "data_section",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def dec(self, format=None, **args):
        """
        Return the dec value, defined for most Gemini instruments at the central pixel
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the declination (in decimal degrees) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_dec"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "dec"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'dec'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.dec(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "dec",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "dec",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def decker(self, format=None, **args):
        """
        Return the decker value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned decker value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       decker value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the decker position used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_decker"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "decker"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'decker'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.decker(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "decker",
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
                                       name = "decker",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def detector_name(self, format=None, **args):
        """
        Return the detector_name value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: string as default (i.e., format=None)
        :rtype: dictionary containing one or more string(s) (format=as_dict)
        :return: the name of the detector used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_detector_name"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "detector_name"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'detector_name'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.detector_name(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "detector_name",
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
                                       name = "detector_name",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def detector_roi_setting(self, format=None, **args):
        """
        Return the detector_roi_setting value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the human-readable description of the detector Region Of
                 Interest (ROI) setting (either 'Full Frame', 'CCD2', 'Central
                 Spectrum', 'Central Stamp', 'Custom', 'Undefined' or 'Fixed'),
                 which corresponds to the name of the ROI in the OT
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_detector_roi_setting"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "detector_roi_setting"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'detector_roi_setting'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.detector_roi_setting(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "detector_roi_setting",
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
                                       name = "detector_roi_setting",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def detector_rois_requested(self, format=None, **args):
        """
        Return the detector_rois_requested value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: list containing a list of integers (corresponding to unbinned
                pixels) that uses 1-bases indexing in the form [x1, x2, y1, y2]
                as default (i.e., format=None) 
        :return: the requested detector Region Of Interest (ROI)s of the
                 observation 
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_detector_rois_requested"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "detector_rois_requested"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'detector_rois_requested'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.detector_rois_requested(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "detector_rois_requested",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = list )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "detector_rois_requested",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def detector_section(self, format=None, **args):
        """
        Return the detector_section value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful 
                       detector_section value in the form [x1:x2,y1:y2] that 
                       uses 1-based indexing
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: list of integers that uses 0-based indexing in the form 
                [x1 - 1, x2 - 1, y1 - 1, y2 - 1] as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2] 
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the unbinned section of the detector that was used to observe
                 the data
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_detector_section"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "detector_section"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'detector_section'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.detector_section(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "detector_section",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = list )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "detector_section",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def detector_x_bin(self, format=None, **args):
        """
        Return the detector_x_bin value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :rtype: dictionary containing one or more integer(s) (format=as_dict)
        :return: the binning of the x-axis of the detector used for the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_detector_x_bin"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "detector_x_bin"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'detector_x_bin'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.detector_x_bin(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "detector_x_bin",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "detector_x_bin",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def detector_y_bin(self, format=None, **args):
        """
        Return the detector_y_bin value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :rtype: dictionary containing one or more integer(s) (format=as_dict)
        :return: the binning of the y-axis of the detector used for the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_detector_y_bin"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "detector_y_bin"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'detector_y_bin'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.detector_y_bin(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "detector_y_bin",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "detector_y_bin",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def disperser(self, format=None, **args):
        """
        Return the disperser value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned disperser value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       disperser value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the disperser used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_disperser"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "disperser"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'disperser'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.disperser(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "disperser",
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
                                       name = "disperser",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def dispersion(self, format=None, **args):
        """
        Return the dispersion value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param asMicrometers: set to True to return the dispersion 
                              value in units of Micrometers
        :type asMicrometers: Python boolean
        :param asNanometers: set to True to return the dispersion 
                             value in units of Nanometers
        :type asNanometers: Python boolean
        :param asAngstroms: set to True to return the dispersion 
                            value in units of Angstroms
        :type asAngstroms: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the dispersion (in meters per pixel as default) of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_dispersion"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "dispersion"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'dispersion'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.dispersion(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "dispersion",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "dispersion",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def dispersion_axis(self, format=None, **args):
        """
        Return the dispersion_axis value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :rtype: dictionary containing one or more integer(s) (format=as_dict)
        :return: the dispersion axis (along rows, x = 1; along columns, y = 2;
                 along planes, z = 3) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_dispersion_axis"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "dispersion_axis"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'dispersion_axis'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.dispersion_axis(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "dispersion_axis",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "dispersion_axis",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def elevation(self, format=None, **args):
        """
        Return the elevation value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the elevation (in degrees) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_elevation"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "elevation"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'elevation'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.elevation(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "elevation",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "elevation",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def exposure_time(self, format=None, **args):
        """
        Return the exposure_time value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the total exposure time (in seconds) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_exposure_time"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "exposure_time"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'exposure_time'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.exposure_time(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "exposure_time",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "exposure_time",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def filter_name(self, format=None, **args):
        """
        Return the filter_name value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned filter_name value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       filter_name value
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: string as default (i.e., format=None)
        :rtype: dictionary containing one or more string(s) (format=as_dict)
        :return: the unique filter name identifier string used for the 
                 observation; when multiple filters are used, the filter names
                 are concatenated with an ampersand
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_filter_name"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "filter_name"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'filter_name'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.filter_name(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "filter_name",
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
                                       name = "filter_name",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def focal_plane_mask(self, format=None, **args):
        """
        Return the focal_plane_mask value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned focal_plane_mask value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       focal_plane_mask value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the focal plane mask used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_focal_plane_mask"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "focal_plane_mask"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'focal_plane_mask'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.focal_plane_mask(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "focal_plane_mask",
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
                                       name = "focal_plane_mask",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def gain(self, format=None, **args):
        """
        Return the gain value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the gain (in electrons per ADU) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_gain"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "gain"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'gain'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.gain(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "gain",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "gain",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def gain_setting(self, format=None, **args):
        """
        Return the gain_setting value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: string as default (i.e., format=None)
        :rtype: dictionary containing one or more string(s) (format=as_dict)
        :return: the gain setting (either 'high' or 'low') of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_gain_setting"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "gain_setting"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'gain_setting'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.gain_setting(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "gain_setting",
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
                                       name = "gain_setting",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def grating(self, format=None, **args):
        """
        Return the grating value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned grating value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       grating value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the grating used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_grating"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "grating"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'grating'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.grating(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "grating",
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
                                       name = "grating",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def gcal_lamp(self, format=None, **args):
        """
        Return the lamp from which GCAL is sending out light. This takes into
        account the fact that the IR lamp is behind a shutter.

        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: string as default (i.e., format=None)
        :rtype: dictionary containing one or more string(s) (format=as_dict)
        :return: the lamp from which gcal is sending out light
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_gcal_lamp"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "gcal_lamp"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'gcal_lamp'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.gcal_lamp(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "gcal_lamp",
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
                                       name = "gcal_lamp",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def group_id(self, format=None, **args):
        """
        Return the group_id value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the unique string that describes which stack a dataset belongs
                 to; it is based on the observation_id
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_group_id"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "group_id"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'group_id'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.group_id(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "group_id",
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
                                       name = "group_id",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def is_ao(self, format=None, **args):
        """
        Return True if the observation uses AO
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: boolean as default (i.e., format=None)
        :return: True if the observation uses adaptive optics, False otherwise        
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_is_ao"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "is_ao"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'is_ao'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.is_ao(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "is_ao",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = bool )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "is_ao",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def local_time(self, format=None, **args):
        """
        Return the local_time value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: datetime as default (i.e., format=None)
        :return: the local time (in HH:MM:SS.S) at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_local_time"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "local_time"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'local_time'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.local_time(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            from datetime import datetime
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "local_time",
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
                                       name = "local_time",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def lyot_stop(self, format=None, **args):
        """
        Return the lyot_stop value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the lyot stop used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_lyot_stop"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "lyot_stop"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'lyot_stop'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.lyot_stop(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "lyot_stop",
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
                                       name = "lyot_stop",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def mdf_row_id(self, format=None, **args):
        """
        Return the mdf_row_id value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :rtype: dictionary containing one or more integer(s) (format=as_dict)
        :return: the corresponding reference row in the Mask Definition File
                 (MDF)
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_mdf_row_id"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "mdf_row_id"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'mdf_row_id'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.mdf_row_id(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "mdf_row_id",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "mdf_row_id",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def nod_count(self, format=None, **args):
        """
        Return the nod_count value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the number of nod and shuffle cycles in the nod and shuffle 
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_nod_count"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "nod_count"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'nod_count'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.nod_count(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "nod_count",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "nod_count",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def nod_pixels(self, format=None, **args):
        """
        Return the nod_pixels value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the number of pixel rows the charge is shuffled by in the nod 
                 and shuffle observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_nod_pixels"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "nod_pixels"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'nod_pixels'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.nod_pixels(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "nod_pixels",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "nod_pixels",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def nominal_atmospheric_extinction(self, format=None, **args):
        """
        Return the nominal_atmospheric_extinction value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None) 
        :return: the nominal atmospheric extinction (defined as coeff *
                 (airmass - 1.0), where coeff is the site and filter specific
                 nominal atmospheric extinction coefficient) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_nominal_atmospheric_extinction"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "nominal_atmospheric_extinction"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'nominal_atmospheric_extinction'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.nominal_atmospheric_extinction(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "nominal_atmospheric_extinction",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "nominal_atmospheric_extinction",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def nominal_photometric_zeropoint(self, format=None, **args):
        """
        Return the nominal_photometric_zeropoint value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the nominal photometric zeropoint of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_nominal_photometric_zeropoint"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "nominal_photometric_zeropoint"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'nominal_photometric_zeropoint'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.nominal_photometric_zeropoint(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "nominal_photometric_zeropoint",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "nominal_photometric_zeropoint",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def non_linear_level(self, format=None, **args):
        """
        Return the non_linear_level value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the non linear level in the raw images (in ADU) of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_non_linear_level"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "non_linear_level"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'non_linear_level'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.non_linear_level(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "non_linear_level",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "non_linear_level",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def observation_class(self, format=None, **args):
        """
        Return the observation_class value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the class (either 'science', 'progCal', 'partnerCal', 'acq', 
                 'acqCal' or 'dayCal') of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_observation_class"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "observation_class"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'observation_class'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.observation_class(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "observation_class",
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
                                       name = "observation_class",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def observation_epoch(self, format=None, **args):
        """
        Return the observation_epoch value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the epoch (in years) at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_observation_epoch"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "observation_epoch"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'observation_epoch'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.observation_epoch(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "observation_epoch",
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
                                       name = "observation_epoch",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def observation_id(self, format=None, **args):
        """
        Return the observation_id value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the ID (e.g., GN-2011A-Q-123-45) of the observation; it is
                 used by group_id
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_observation_id"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "observation_id"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'observation_id'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.observation_id(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "observation_id",
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
                                       name = "observation_id",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def observation_type(self, format=None, **args):
        """
        Return the observation_type value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the type (either 'OBJECT', 'DARK', 'FLAT', 'ARC', 'BIAS' or 
                 'MASK') of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_observation_type"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "observation_type"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'observation_type'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.observation_type(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "observation_type",
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
                                       name = "observation_type",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def overscan_section(self, format=None, **args):
        """
        Return the overscan_section value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful
                       overscan_section value in the form [x1:x2,y1:y2] that
                       uses 1-based indexing
        :type pretty: Python boolean
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: list of integers that uses 0-based indexing in the form 
                [x1 - 1, x2 - 1, y1 - 1, y2 - 1] as default 
                (i.e., format=None, pretty=False)
        :rtype: string that uses 1-based indexing in the form [x1:x2,y1:y2] 
                (pretty=True)
        :rtype: dictionary containing one or more of the above return types 
                (format=as_dict)
        :return: the section of the pixel data extensions that contains the
                 overscan data
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_overscan_section"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "overscan_section"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'overscan_section'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.overscan_section(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "overscan_section",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = list )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "overscan_section",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def pixel_scale(self, format=None, **args):
        """
        Return the pixel_scale value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the pixel scale (in arcsec per pixel) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_pixel_scale"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "pixel_scale"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'pixel_scale'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.pixel_scale(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "pixel_scale",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "pixel_scale",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def prism(self, format=None, **args):
        """
        Return the prism value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned prism value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       prism value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the prism used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_prism"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "prism"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'prism'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.prism(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "prism",
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
                                       name = "prism",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def program_id(self, format=None, **args):
        """
        Return the program_id value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the Gemini program ID (e.g., GN-2011A-Q-123) of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_program_id"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "program_id"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'program_id'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.program_id(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "program_id",
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
                                       name = "program_id",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def pupil_mask(self, format=None, **args):
        """
        Return the pupil_mask value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned pupil mask value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful
                       pupil mask value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the pupil mask used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_pupil_mask"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "pupil_mask"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'pupil_mask'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.pupil_mask(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "pupil_mask",
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
                                       name = "pupil_mask",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def qa_state(self, format=None, **args):
        """
        Return the qa_state value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the quality assessment state (either 'Undefined', 'Pass', 
                 'Usable', 'Fail' or 'CHECK') of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_qa_state"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "qa_state"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'qa_state'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.qa_state(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "qa_state",
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
                                       name = "qa_state",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def ra(self, format=None, **args):
        """
        Return the ra value, defined for most Gemini instruments at the central pixel
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the Right Ascension (in decimal degrees) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_ra"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "ra"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'ra'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.ra(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "ra",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "ra",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def raw_bg(self, format=None, **args):
        """
        Return the raw_bg value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw background (as an integer percentile value) of the
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_raw_bg"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "raw_bg"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'raw_bg'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.raw_bg(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "raw_bg",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "raw_bg",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def raw_cc(self, format=None, **args):
        """
        Return the raw_cc value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw cloud cover (as an integer percentile value) of the
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_raw_cc"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "raw_cc"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'raw_cc'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.raw_cc(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "raw_cc",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "raw_cc",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def raw_iq(self, format=None, **args):
        """
        Return the raw_iq value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw image quality (as an integer percentile value) of the
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_raw_iq"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "raw_iq"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'raw_iq'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.raw_iq(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "raw_iq",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "raw_iq",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def raw_wv(self, format=None, **args):
        """
        Return the raw_wv value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the raw water vapour (as an integer percentile value) of the
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_raw_wv"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "raw_wv"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'raw_wv'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.raw_wv(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "raw_wv",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "raw_wv",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def read_mode(self, format=None, **args):
        """
        Return the read_mode value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: GNIRS/NIFS: one of
                 'Very Faint Object(s)', 
                 'Faint Object(s)', 
                 'Medium Object(s)', 
                 'Bright Object(s)', 
                 'Very Bright Object(s)', 

                 NIRI: one of 
                 'Low Background', 
                 'Medium Background', 
                 'High Background',
                 'Invalid'

                 GMOS: one of 
                 'Normal',
                 'Bright',
                 'Acquisition',
                 'Engineering'

        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_read_mode"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "read_mode"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'read_mode'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.read_mode(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "read_mode",
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
                                       name = "read_mode",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def read_noise(self, format=None, **args):
        """
        Return the read_noise value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the estimated readout noise (in electrons) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_read_noise"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "read_noise"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'read_noise'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.read_noise(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "read_noise",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "read_noise",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def read_speed_setting(self, format=None, **args):
        """
        Return the read_speed_setting value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the read speed setting (either 'fast' or 'slow') of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_read_speed_setting"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "read_speed_setting"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'read_speed_setting'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.read_speed_setting(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "read_speed_setting",
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
                                       name = "read_speed_setting",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def requested_bg(self, format=None, **args):
        """
        Return the requested_bg value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested background (as an integer percentile value) of 
                 the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_requested_bg"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "requested_bg"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'requested_bg'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.requested_bg(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "requested_bg",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "requested_bg",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def requested_cc(self, format=None, **args):
        """
        Return the requested_cc value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested cloud cover (as an integer percentile value) of
                 the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_requested_cc"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "requested_cc"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'requested_cc'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.requested_cc(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "requested_cc",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "requested_cc",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def requested_iq(self, format=None, **args):
        """
        Return the requested_iq value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested image quality (as an integer percentile value)
                 of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_requested_iq"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "requested_iq"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'requested_iq'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.requested_iq(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "requested_iq",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "requested_iq",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def requested_wv(self, format=None, **args):
        """
        Return the requested_wv value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :return: the requested water vapour (as an integer percentile value) of
                 the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_requested_wv"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "requested_wv"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'requested_wv'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.requested_wv(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "requested_wv",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "requested_wv",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def saturation_level(self, format=None, **args):
        """
        Return the saturation_level value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: integer as default (i.e., format=None)
        :rtype: dictionary containing one or more integer(s) (format=as_dict)
        :return: the saturation level (in ADU) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_saturation_level"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "saturation_level"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'saturation_level'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.saturation_level(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "saturation_level",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = int )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "saturation_level",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def target_dec(self, format=None, **args):
        """
        Return the target_dec value

        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the target_dec value
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_target_dec"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "target_dec"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'target_dec'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.target_dec(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "target_dec",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "target_dec",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def target_ra(self, format=None, **args):
        """
        Return the target_ra value

        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the target_ra value
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_target_ra"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "target_ra"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'target_ra'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.target_ra(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "target_ra",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "target_ra",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def slit(self, format=None, **args):
        """
        Return the slit value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned slit value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       slit value
        :type pretty: Python boolean
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the name of the slit used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_slit"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "slit"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'slit'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.slit(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "slit",
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
                                       name = "slit",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def ut_datetime(self, format=None, **args):
        """
        Return the ut_datetime value
        
        This descriptor attempts to figure out the datetime even when the
        headers are malformed or not present. It tries just about every header
        combination that could allow it to determine an appropriate datetime
        for the file in question. This makes it somewhat specific to Gemini
        data, in that the headers it looks at, and the assumptions it makes in
        trying to parse their values, are those known to occur in Gemini data.
        Note that some of the early gemini data, and that taken from lower
        level engineering interfaces, lack standard headers. Also the format
        and occurrence of various headers has changed over time, even on the
        same instrument. If strict is set to True, the date or time are
        determined from valid FITS keywords. If it cannot be determined, None
        is returned. If dateonly or timeonly are set to True, then a
        datetime.date object or datetime.time object, respectively, is
        returned, containing only the date or time, respectively. These two
        interplay with strict in the sense that if strict is set to True and a
        date can be determined but not a time, then this function will return
        None unless the dateonly flag is set, in which case it will return the
        valid date. The dateonly and timeonly flags are intended for use by
        the ut_date and ut_time descriptors.
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param strict: set to True to not try to guess the date or time
        :type strict: Python boolean
        :param dateonly: set to True to return a datetime.date
        :type dateonly: Python boolean
        :param timeonly: set to True to return a datetime.time
        :param timeonly: Python boolean
        :param format: the return format
        :type format: string
        :rtype: datetime.datetime (dateonly=False and timeonly=False)
        :rtype: datetime.time (timeonly=True)
        :rtype: datetime.date (dateonly=True)
        :return: the UT date and time at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_ut_datetime"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "ut_datetime"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'ut_datetime'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.ut_datetime(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            from datetime import datetime
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "ut_datetime",
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
                                       name = "ut_datetime",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def ut_time(self, format=None, **args):
        """
        Return the ut_time value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: datetime as default (i.e., format=None)
        :return: the UT time at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_ut_time"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "ut_time"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'ut_time'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.ut_time(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            from datetime import datetime
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "ut_time",
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
                                       name = "ut_time",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def wavefront_sensor(self, format=None, **args):
        """
        Return the wavefront_sensor value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the wavefront sensor (either 'AOWFS', 'OIWFS', 'PWFS1', 
                 'PWFS2', some combination in alphabetic order separated with 
                 an ampersand or None) used for the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_wavefront_sensor"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "wavefront_sensor"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'wavefront_sensor'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.wavefront_sensor(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "wavefront_sensor",
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
                                       name = "wavefront_sensor",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def wavelength_band(self, format=None, **args):
        """
        Return the wavelength_band value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the wavelength band name (e.g., J, V, R, N) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_wavelength_band"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "wavelength_band"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'wavelength_band'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.wavelength_band(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "wavelength_band",
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
                                       name = "wavelength_band",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def wavelength_reference_pixel(self, format=None, **args):
        """
        Return the wavelength_reference_pixel value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
                       set to as_dict to return a dictionary, where the number 
                       of dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type format: string
        :rtype: float as default (i.e., format=None)
        :rtype: dictionary containing one or more float(s) (format=as_dict)
        :return: the 1-based reference pixel of the central wavelength of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_wavelength_reference_pixel"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "wavelength_reference_pixel"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'wavelength_reference_pixel'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.wavelength_reference_pixel(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "wavelength_reference_pixel",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "wavelength_reference_pixel",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def wcs_ra(self, format=None, **args):
        """
        Return the wcs_ra value

        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the wcs_ra value
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_wcs_ra"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "wcs_ra"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'wcs_ra'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.wcs_ra(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "wcs_ra",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "wcs_ra",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def wcs_dec(self, format=None, **args):
        """
        Return the wcs_dec value

        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the wcs_dec value
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_wcs_dec"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "wcs_dec"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'wcs_dec'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.wcs_dec(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "wcs_dec",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "wcs_dec",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def well_depth_setting(self, format=None, **args):
        """
        Return the well_depth_setting value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the well depth setting (either 'Shallow', 'Deep' or 
                 'Invalid') of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_well_depth_setting"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "well_depth_setting"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'well_depth_setting'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.well_depth_setting(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "well_depth_setting",
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
                                       name = "well_depth_setting",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def x_offset(self, format=None, **args):
        """
        Return the x_offset value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the telescope offset in x (in arcsec) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_x_offset"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "x_offset"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'x_offset'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.x_offset(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "x_offset",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "x_offset",
                                       keyword = keyword,
                                       ad = self,
                                       pytype = None )
                return ret
        except:
            raise

    def y_offset(self, format=None, **args):
        """
        Return the y_offset value
        
        :param dataset: the dataset
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: float as default (i.e., format=None)
        :return: the telescope offset in y (in arcsec) of the observation
        """
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_y_offset"
            keyword = None
            if key in keydict:
                keyword = keydict[key]

            if not hasattr(self.descriptor_calculator, "y_offset"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise Errors.DescriptorError(self.exception_info)
                else:
                    msg = ("Unable to find an appropriate descriptor "
                           "function or a default keyword for 'y_offset'")
                    raise Errors.DescriptorInfrastructureError(msg)
            else:
                try:
                    retval = self.descriptor_calculator.y_offset(self, **args)
                except Exception as e:
                    raise Errors.DescriptorError(e)

            
            ret = DescriptorValue( retval,
                                   format = format,
                                   name = "y_offset",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = float )
            return ret

        except Errors.DescriptorError:
            if self.descriptor_calculator.throwExceptions == True:
                raise
            else:
                if not hasattr(self, "exception_info"):
                    setattr(self, "exception_info", sys.exc_info()[1])
                ret = DescriptorValue( None,
                                       format = format,
                                       name = "y_offset",
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

