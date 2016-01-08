import math

from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.interface.Descriptors import DescriptorValue

from gempy.gemini import gemini_metadata_utils as gmu

from NIFS_Keywords import NIFS_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

class NIFS_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = NIFS_KeyDict
    
    nifsArrayDict = None
    nifsConfigDict = None
    
    def __init__(self):
        self.nifsArrayDict = Lookups.get_lookup_table(
            "Gemini/NIFS/NIFSArrayDict", "nifsArrayDict")
        self.nifsConfigDict = Lookups.get_lookup_table(
            "Gemini/NIFS/NIFSConfigDict", "nifsConfigDict")
        GEMINI_DescriptorCalc.__init__(self)
    
    def disperser(self, dataset, stripID=False, pretty=False, **args):
        if pretty:
            stripID = True
        
        # Determine the disperser keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_disperser")
        
        # Get the value of the disperser keyword from the header of the PHU
        disperser = dataset.phu_get_key_value(keyword)
        
        if disperser is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if stripID:
            # Return the stripped disperser string
            ret_disperser = gmu.removeComponentID(disperser)
        else:
            # Return the disperser string
            ret_disperser = str(disperser)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_disperser, name="disperser", ad=dataset)
        
        return ret_dv
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Determine the filter name keyword from the global keyword dictionary
        keyword = self.get_descriptor_key("key_filter_name")
        
        # Get the value of the filter name keyword from the header of the PHU
        filter_name = dataset.phu_get_key_value(keyword)
        
        if filter_name is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if pretty:
            stripID = True
        if stripID:
            # Return the stripped filter name string
            ret_filter_name = gmu.removeComponentID(filter_name)
        else:
            # Return the filter name string
            ret_filter_name = str(filter_name)
        if filter_name == "Blocked":
            ret_filter_name = "blank"
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_filter_name, name="filter_name",
                                 ad=dataset)
        return ret_dv
    
    def gain(self, dataset, **args):
        # Determine the bias value keyword (biasvolt) from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_bias")
        
        # Get the value of the bias value keyword from the header of the PHU
        biasvolt = dataset.phu_get_key_value(keyword)
        
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
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
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_gain, name="gain", ad=dataset)
        
        return ret_dv
    
    nifsArrayDict = None
    
    def non_linear_level(self, dataset, **args):
        # Determine the bias value keyword (biasvolt) from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_bias")
        
        # Get the value of the bias value keyword from the header of the PHU
        biasvolt = dataset.phu_get_key_value(keyword)
        
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the saturation level using the appropriate descriptor
        saturation_level = dataset.saturation_level()
        
        if saturation_level is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Determine whether the dataset has been / will be corrected for
        # non-linearity
        if dataset.phu_get_key_value("NONLINCR"):
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
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_non_linear_level, name="non_linear_level",
                                 ad=dataset)
        return ret_dv
    
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
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        pixel_scale_key = (focal_plane_mask, disperser, filter_name)
        
        if pixel_scale_key in getattr(self, "nifsConfigDict"):
            row = self.nifsConfigDict[pixel_scale_key]
        else:
            raise Errors.TableKeyError()
        
        if float(row[2]):
            # Return the pixel scale float
            ret_pixel_scale = float(row[2])
        else:
            raise Errors.TableValueError()
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_pixel_scale, name="pixel_scale",
                                 ad=dataset)
        return ret_dv
    
    nifsConfigDict = None
    
    def read_mode(self, dataset, **args):
        # Determine the number of non-destructive read pairs (lnrs) and the
        # bias value (biasvolt) keywords from the global keyword dictionary
        keyword1 = self.get_descriptor_key("key_lnrs")
        keyword2 = self.get_descriptor_key("key_bias")
        
        # Get the value of the number of non-destructive read pairs and the
        # bias value keywords from the header of the PHU
        lnrs = dataset.phu_get_key_value(keyword1)
        biasvolt = dataset.phu_get_key_value(keyword2)
        
        if lnrs is None or biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if lnrs == 1:
            read_mode = "Bright Object"
        elif lnrs == 4:
            read_mode = "Medium Object"
        elif lnrs == 16:
            read_mode = "Faint Object"
        else:
            read_mode = "Invalid"
        
        # Return the read mode string
        ret_read_mode = str(read_mode)
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_mode, name="read_mode", ad=dataset)
        
        return ret_dv
    
    def read_noise(self, dataset, **args):
        # Determine the number of non-destructive read pairs (lnrs) and the
        # bias value (biasvolt) keywords from the global keyword dictionary
        keyword1 = self.get_descriptor_key("key_lnrs")
        keyword2 = self.get_descriptor_key("key_bias")
        
        # Get the value of the number of non-destructive read pairs and the
        # bias value keywords from the header of the PHU
        lnrs = dataset.phu_get_key_value(keyword1)
        biasvolt = dataset.phu_get_key_value(keyword2)
        
        if lnrs is None or biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
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
        ret_read_noise = float((read_noise * math.sqrt(coadds)) /
                               math.sqrt(lnrs))
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_read_noise, name="read_noise", ad=dataset)
        
        return ret_dv
    
    nifsArrayDict = None
    
    def saturation_level(self, dataset, **args):
        # Determine the bias value keyword (biasvolt) from the global keyword
        # dictionary
        keyword = self.get_descriptor_key("key_bias")
        
        # Get the value of the bias value keyword from the header of the PHU
        biasvolt = dataset.phu_get_key_value(keyword)
        
        if biasvolt is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        # Get the number of coadds using the appropriate descriptor
        coadds = dataset.coadds()
        
        if coadds is None:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception. It will be
            # dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
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
        
        # Instantiate the return DescriptorValue (DV) object
        ret_dv = DescriptorValue(ret_saturation_level, name="saturation_level",
                                 ad=dataset)
        return ret_dv
    
    nifsArrayDict = None
