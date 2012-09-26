from astrodata import Errors
from gempy.gemini import gemini_metadata_utils as gmu

from GSAOI_Keywords import GSAOI_KeyDict
from GEMINI_Descriptors import GEMINI_DescriptorCalc

class GSAOI_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dictionary with the local key dictionary
    # associated with this descriptor class
    _update_stdkey_dict = GSAOI_KeyDict
    
    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)

    def array_section(self, dataset, pretty=False, extname="SCI", **args):
        # Since this descriptor function accesses keywords in the headers of
        # the pixel data extensions, always return a dictionary where the key
        # of the dictionary is an (EXTNAME, EXTVER) tuple.
        ret_array_section = {}

        # Loop over the science extensions in the dataset
        for ext in dataset[extname]:
            # Get the data section from the header of each pixel data extension
            raw_array_section = ext.get_key_value(
                self.get_descriptor_key("key_array_section"))

            if raw_array_section is None:
                # The get_key_value() function returns None if a value cannot
                # be found and stores the exception info. Re-raise the
                # exception. It will be dealt with by the CalculatorInterface.
                if hasattr(ext, "exception_info"):
                    raise ext.exception_info

            if pretty:
                # Return a dictionary with the array section string that uses
                # 1-based indexing as the value in the form [x1:x2,y1:y2]
                ret_array_section.update({
                    (ext.extname(), ext.extver()):str(raw_array_section)})
            else:
                # Return a dictionary with the array section list that uses
                # 0-based, non-inclusive indexing as the value in the form
                # [x1, x2, y1, y2]
                array_section = gmu.sectionStrToIntList(raw_array_section)
                ret_array_section.update({
                    (ext.extname(), ext.extver()):array_section})

        if ret_array_section == {}:
            # If the dictionary is still empty, the AstroData object was not
            # automatically assigned an "extname" extension and so the above
            # for loop was not entered
            raise Errors.CorruptDataError()
        
        return ret_array_section
