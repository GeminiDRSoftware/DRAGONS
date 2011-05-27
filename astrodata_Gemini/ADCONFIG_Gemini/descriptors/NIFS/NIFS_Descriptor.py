import math

from astrodata import Descriptors
from astrodata import Errors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy import string

from StandardNIFSKeyDict import stdkeyDictNIFS
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class NIFS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = stdkeyDictNIFS
    
    nifsArrayDict = None
    nifsConfigDict = None
    
    def __init__(self):
        self.nifsArrayDict = \
            Lookups.get_lookup_table('Gemini/NIFS/NIFSArrayDict',
                                   'nifsArrayDict')
        self.nifsConfigDict = \
            Lookups.get_lookup_table('Gemini/NIFS/NIFSConfigDict',
                                   'nifsConfigDict')
        GEMINI_DescriptorCalc.__init__(self)
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            stripID = True
        # Get the disperser value from the header of the PHU. The disperser
        # keyword is defined in the local key dictionary (stdkeyDictNIFS) but
        # is read from the updated global key dictionary (self._specifickey_dict)
        disperser = dataset.phu_get_key_value(self._specifickey_dict['key_disperser'])
        if disperser is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if stripID:
            # Return the stripped disperser string
            ret_disperser = string.removeComponentID(disperser)
        else:
            # Return the disperser string
            ret_disperser = str(disperser)
        
        return ret_disperser
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Get the filter name value from the header of the PHU. The filter name
        # keywords are defined in the local key dictionary (stdkeyDictNIFS) but
        # are read from the updated global key dictionary (self._specifickey_dict)
        filter_name = \
            dataset.phu_get_key_value(self._specifickey_dict['key_filter_name'])
        if filter_name is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if pretty:
            stripID = True
        if stripID:
            # Return the stripped filter name string
            ret_filter_name = string.removeComponentID(filter_name)
        else:
            # Return the filter name string
            ret_filter_name = str(filter_name)
        if filter_name == 'Blocked':
            ret_filter_name = 'blank'
        
        return ret_filter_name
    
    def gain(self, dataset, **args):
        # Get the bias value (biasvolt) from the header of the PHU. The bias
        # keyword is defined in the local key dictionary (stdkeyDictNIFS) but
        # is read from the updated global key dictionary (self._specifickey_dict)
        biasvolt = dataset.phu_get_key_value(self._specifickey_dict['key_bias'])
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        bias_values = self.nifsArrayDict.keys()
        count = 0
        for bias in bias_values:
            if abs(float(bias) - abs(biasvolt)) < 0.1:
                count += 1
                if float(self.nifsArrayDict[bias][1]):
                    # Return the gain float
                    ret_gain = float(self.nifsArrayDict[bias][1])
                else:
                    Errors.TableValueError()
        if count == 0:
            Errors.TableKeyError()
        
        return ret_gain
    
    nifsArrayDict = None
    
    def non_linear_level(self, dataset, **args):
        # Get the bias value (biasvolt) from the header of the PHU. The bias
        # keyword is defined in the local key dictionary (stdkeyDictNIFS) but
        # is read from the updated global key dictionary (self._specifickey_dict)
        biasvolt = dataset.phu_get_key_value(self._specifickey_dict['key_bias'])
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Get the saturation level using the appropriate descriptor
        saturation_level = dataset.saturation_level()
        if saturation_level is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Determine whether the dataset has been / will be corrected for
        # non-linearity
        if dataset.phu_get_key_value('NONLINCR'):
            corrected = True
        else:
            corrected = False
        # The array is non-linear at some fraction of the saturation level.
        # Get this fraction from the lookup table
        bias_values = self.nifsArrayDict.keys()
        count = 0
        for bias in bias_values:
            if abs(float(bias) - abs(biasvolt)) < 0.1:
                count += 1
                row = self.nifsArrayDict[bias]
                if corrected:
                    # Use row[3] if correcting for non-linearity
                    if float(row[3]):
                        linearlimit = float(row[3])
                    else:
                        Errors.TableValueError()
                else:
                    # Use row[7] if not correcting for non-linearity
                    if float(row[7]):
                        linearlimit = float(row[7])
                    else:
                        Errors.TableValueError()
        if count == 0:
            Errors.TableKeyError()
        # Return the saturation level integer
        ret_non_linear_level = int(saturation_level * linearlimit)
        
        return ret_non_linear_level
    
    nifsArrayDict = None
    
    def pixel_scale(self, dataset, **args):
        # Get the focal plane mask, disperser and filter values using the
        # appropriate descriptors. Use as_pytype() to return the values as the
        # default python type, rather than an object
        focal_plane_mask = dataset.focal_plane_mask().as_pytype()
        disperser = dataset.disperser().as_pytype()
        filter_name = dataset.filter_name().as_pytype()
        if focal_plane_mask is None or disperser is None or \
            filter_name is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        pixel_scale_key = (focal_plane_mask, disperser, filter_name)
        if pixel_scale_key in getattr(self, 'nifsConfigDict'):
            row = self.nifsConfigDict[pixel_scale_key]
        else:
            raise Errors.TableKeyError()
        if float(row[2]):
            # Return the pixel scale float
            ret_pixel_scale = float(row[2])
        else:
            raise Errors.TableValueError()
        
        return ret_pixel_scale
    
    nifsConfigDict = None
    
    def read_mode(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) and the the bias
        # value (biasvolt) from the header of the PHU. The lnrs and biasvolt
        # keywords are defined in the local key dictionary (stdkeyDictNIFS) but
        # are read from the updated global key dictionary (self._specifickey_dict)
        lnrs = dataset.phu_get_key_value(self._specifickey_dict['key_lnrs'])
        biasvolt = dataset.phu_get_key_value(self._specifickey_dict['key_bias'])
        if lnrs is None or biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        if lnrs == 1:
            read_mode = 'Bright Object'
        elif lnrs == 4:
            read_mode = 'Medium Object'
        elif lnrs == 16:
            read_mode = 'Faint Object'
        else:
            read_mode = 'Invalid'
        # Return the read mode string
        ret_read_mode = str(read_mode)
        
        return ret_read_mode
    
    def read_noise(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) and the the bias
        # value (biasvolt) from the header of the PHU. The lnrs and biasvolt
        # keywords are defined in the local key dictionary (stdkeyDictNIFS) but
        # are read from the updated global key dictionary (self._specifickey_dict)
        lnrs = dataset.phu_get_key_value(self._specifickey_dict['key_lnrs'])
        biasvolt = dataset.phu_get_key_value(self._specifickey_dict['key_bias'])
        if lnrs is None or biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        bias_values = self.nifsArrayDict.keys()
        count = 0
        for bias in bias_values:
            if abs(float(bias) - abs(biasvolt)) < 0.1:
                count += 1
                if float(self.nifsArrayDict[bias][0]):
                    read_noise = float(self.nifsArrayDict[bias][0])
                else:
                    Errors.TableValueError()
        if count == 0:
            Errors.TableKeyError()
        # Return the read noise float
        ret_read_noise = float((read_noise * math.sqrt(coadds)) \
            / math.sqrt(lnrs))
        
        return ret_read_noise
    
    nifsArrayDict = None
    
    def saturation_level(self, dataset, **args):
        # Get the bias value (biasvolt) from the header of the PHU. The bias
        # keyword is defined in the local key dictionary (stdkeyDictNIFS) but
        # is read from the updated global key dictionary (self._specifickey_dict)
        biasvolt = dataset.phu_get_key_value(self._specifickey_dict['key_bias'])
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Re-raise the exception. It
            # will be dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, 'exception_info'):
                raise dataset.exception_info
        bias_values = self.nifsArrayDict.keys()
        count = 0
        for bias in bias_values:
            if abs(float(bias) - abs(biasvolt)) < 0.1:
                count += 1
                if float(self.nifsArrayDict[bias][2]):
                    well = float(self.nifsArrayDict[bias][2])
                else:
                    Errors.TableValueError()
        if count == 0:
            Errors.TableKeyError()
        # Return the saturation level integer
        ret_saturation_level = int(well * coadds)
        
        return ret_saturation_level
    
    nifsArrayDict = None
