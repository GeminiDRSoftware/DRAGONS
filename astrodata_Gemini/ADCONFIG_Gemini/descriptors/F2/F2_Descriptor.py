from datetime import datetime
from time import strptime

from astrodata import Descriptors
from astrodata import Errors
from astrodata import Lookups
from astrodata.Calculator import Calculator
from gempy import gemini_metadata_utils as gmu

from StandardF2KeyDict import stdkeyDictF2
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class F2_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = stdkeyDictF2
    
    f2ArrayDict = None
    f2ConfigDict = None
    
    def __init__(self):
        self.f2ArrayDict = Lookups.get_lookup_table(
            "Gemini/F2/F2ArrayDict", "f2ArrayDict")
        self.nifsConfigDict = Lookups.get_lookup_table(
            "Gemini/F2/F2ConfigDict", "f2ConfigDict")
        GEMINI_DescriptorCalc.__init__(self)
    
    def data_section(self, dataset, pretty=False, extname="SCI", **args):
        ret_data_section = {}
        data_section = "[1:2048,1:2048]"
        
        for ext in dataset[extname]:
            if pretty:
                # Return the data section string that uses 1-based 
                # indexing as the value in the form [x1:x2,y1:y2]
                ret_data_section.update(
                    {(ext.extname(), ext.extver()): data_section})
            else:
                # Return the data section list that used 0-based, non-inclusive
                # indexing as the value in the form [x1, x2, y1, y2]
                ret_data_section.update(
                    {(ext.extname(), ext.extver()): gmu.sectionStrToIntList(
                                                                data_section)})
        return ret_data_section
        
    array_section = data_section
    detector_section = data_section

    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        # Get the UT date using the appropriate descriptor
        ut_date = str(dataset.ut_date())
        if ut_date is None:
            # The descriptor functions return None if a value cannot be
            # found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        obs_ut_date = datetime(*strptime(ut_date, "%Y-%m-%d")[0:6])
        # Old commissioning data was taken before March 1, 2010
        old_ut_date = datetime(2010, 3, 1, 0, 0)

        if obs_ut_date > old_ut_date:
            
            # Get the two filter name values from the header of the PHU. The
            # two filter name keywords may be defined in a local key dictionary
            # (stdkey_dict<INSTRUMENT>) but are read from the updated global
            # key dictionary (self.get_descriptor_key())
            filter1 = dataset.phu_get_key_value(
                self.get_descriptor_key("key_filter1"))
            filter2 = dataset.phu_get_key_value(
                self.get_descriptor_key("key_filter2"))
            if filter1 is None or filter2 is None:
                # The phu_get_key_value() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
        else:
            # Make sure the filter_name descriptor is backwards compatible with
            # old engineering data
            # Get the two filter name values from the header of the PHU. The
            # two filter name keywords may be defined in a local key dictionary
            # (stdkey_dict<INSTRUMENT>) but are read from the updated global
            # key dictionary (self.get_descriptor_key())
            filter1 = dataset.phu_get_key_value(
                self.get_descriptor_key("key_old_filter1"))
            filter2 = dataset.phu_get_key_value(
                self.get_descriptor_key("key_old_filter2"))
            if filter1 is None or filter2 is None:
                # The phu_get_key_value() function returns None if a value
                # cannot be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(dataset, "exception_info"):
                    raise dataset.exception_info
            
        if stripID or pretty:
            # Strip the component ID from the two filter name values
            filter1 = gmu.removeComponentID(filter1)
            filter2 = gmu.removeComponentID(filter2)
        filter = []
        if pretty:
            # Remove any filters that have the value "open" or "Open"
            if "open" not in filter1 and "Open" not in filter1:
                filter.append(str(filter1))
            if "open" not in filter2 and "Open" not in filter2:
                filter.append(str(filter2))
            if len(filter) == 0:
                filter.append("open")
            if "Block" in filter1 or "Block" in filter2:
                filter.append("blank")
            if "Dark" in filter1 or "Dark" in filter2:
                filter.append("blank")
            if "DK" in filter1 or "DK" in filter2:
                filter.append("dark")
        else:
            filter = [filter1, filter2]
        if len(filter) > 1:
            # Concatenate the filter names with "&"
            filter_name = "%s&%s" % (filter[0], filter[1])
        else:
            filter_name = str(filter[0])
        # Return a dictionary where the key of the dictionary is an (EXTNAME,
        # EXTVER) tuple and the value is the filter name string
        ret_filter_name = {}
        # Loop over the science extensions of the dataset
        for ext in dataset["SCI"]:
            ret_filter_name.update(
                {(ext.extname(), ext.extver()):filter_name})
        if ret_filter_name == {}:
            # If the dictionary is still empty, the AstroData object was not
            # automatically assigned a "SCI" extension and so the above for loop
            # was not entered
            raise Errors.CorruptDataError()
        
        return ret_filter_name

    def gain(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) from the header
        # of the PHU. The lnrs keyword is defined in the local key dictionary
        # (stdkeyDictF2) but are read from the updated global key dictionary
        # (self.get_descriptor_key)
        lnrs = dataset.phu_get_key_value(self.get_descriptor_key("key_lnrs"))
        
        if lnrs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if lnrs in getattr(self, "f2ArrayDict"):
            gain = self.f2ArrayDict[lnrs][1]
        else:
            raise Errors.TableKeyError()
        
        # Return the gain float
        ret_gain = float(gain)
        
        return ret_gain
    
    f2ArrayDict = None
    
    def instrument(self, dataset, **args):
        ret_instrument = "F2"
        
        return ret_instrument
    
    def non_linear_level(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) from the header
        # of the PHU. The lnrs keyword is defined in the local key dictionary
        # (stdkeyDictF2) but are read from the updated global key dictionary
        # (self.get_descriptor_key)
        lnrs = dataset.phu_get_key_value(self.get_descriptor_key("key_lnrs"))
        
        if lnrs is None:
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
        
        if lnrs in getattr(self, "f2ArrayDict"):
            non_linear_fraction = self.f2ArrayDict[lnrs][3]
        else:
            raise Errors.TableKeyError()
        
        # Return the read noise float
        ret_non_linear_level = int(saturation_level * non_linear_fraction)
        
        return ret_non_linear_level
    
    f2ArrayDict = None
    
    def read_noise(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) from the header
        # of the PHU. The lnrs keyword is defined in the local key dictionary
        # (stdkeyDictF2) but are read from the updated global key dictionary
        # (self.get_descriptor_key)
        lnrs = dataset.phu_get_key_value(self.get_descriptor_key("key_lnrs"))
        
        if lnrs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if lnrs in getattr(self, "f2ArrayDict"):
            read_noise = self.f2ArrayDict[lnrs][0]
        else:
            raise Errors.TableKeyError()
        
        # Return the read noise float
        ret_read_noise = float(read_noise)
        
        return ret_read_noise
    
    f2ArrayDict = None
    
    def saturation_level(self, dataset, **args):
        # Get the number of non-destructive read pairs (lnrs) from the header
        # of the PHU. The lnrs keyword is defined in the local key dictionary
        # (stdkeyDictF2) but are read from the updated global key dictionary
        # (self.get_descriptor_key)
        lnrs = dataset.phu_get_key_value(self.get_descriptor_key("key_lnrs"))
        
        if lnrs is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        if lnrs in getattr(self, "f2ArrayDict"):
            saturation_level = self.f2ArrayDict[lnrs][2]
        else:
            raise Errors.TableKeyError()
        
        # Return the read noise float
        ret_saturation_level = int(saturation_level)
        
        return ret_saturation_level
    
    f2ArrayDict = None
    
    def wavelength_band(self, dataset, **args):
        if "IMAGE" in dataset.types:
            # If imaging, associate the filter name with a central wavelength
            filter_table = Lookups.get_lookup_table(
                "Gemini/F2/F2FilterWavelength", "filter_wavelength")
            filter = str(dataset.filter_name(pretty=True))
            if filter in filter_table:
                ctrl_wave = filter_table[filter]
            else:
                raise Errors.TableKeyError()
        else:
            ctrl_wave = float(dataset.central_wavelength(asMicrometers=True))

        min_diff = None
        band = None
        for (std_band,std_wave) in self.std_wavelength_band.items():
            diff = abs(std_wave-ctrl_wave)
            if min_diff is None or diff<min_diff:
                min_diff = diff
                band = std_band
        if band is None:
            raise Errors.CalcError()

        return band

    def x_offset(self, dataset, **args):
        # Get the y offset from the header of the PHU.
        y_offset = dataset.phu_get_key_value(
            self.get_descriptor_key("key_y_offset"))
        
        if y_offset is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        ret_x_offset = -y_offset
        
        return ret_x_offset
    
    def y_offset(self, dataset, **args):
        # Get the x offset from the header of the PHU.
        x_offset = dataset.phu_get_key_value(
            self.get_descriptor_key("key_x_offset"))
        
        if x_offset is None:
            # The phu_get_key_value() function returns None if a value cannot
            # be found and stores the exception info. Re-raise the exception.
            # It will be dealt with by the CalculatorInterface.
            if hasattr(dataset, "exception_info"):
                raise dataset.exception_info
        
        ret_y_offset = -x_offset
        
        return ret_y_offset
