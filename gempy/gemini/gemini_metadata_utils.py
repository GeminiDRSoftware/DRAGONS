# The string module contains utilities functions that manipulate strings

import re
import sys

from astrodata.structuredslice import pixel_exts, bintable_exts

def removeComponentID(instr):
    """
    Remove a component ID from a filter name
    :param instr: the filter name
    :type instr: string
    :rtype: string
    :return: the filter name with the component ID removed
    """
    m = re.match (r"(?P<filt>.*?)_G(.*?)", instr)
    if not m:
        # There was no "_G" in the input string. Return the input string
        ret_str = str(instr)
    else:
        ret_str = str(m.group("filt"))
    
    return ret_str

def sectionStrToIntList(section):
    """
    Convert the input section in the form '[x1:x2,y1:y2]' to a list in the
    form [x1 - 1, x2, y1 - 1, y2], where x1, x2, y1 and y2 are
    integers. The values in the output list are converted to use 0-based and 
    non-inclusive indexing, making it compatible with numpy.
    
    :param section: the section (in the form [x1:x2,y1:y2]) to be
                    converted to a list
    :type section: string
    
    :rtype: list
    :return: the converted section as a list that uses 0-based and
             non-inclusive in the form [x1 - 1, x2, y1 - 1, y2]
    """
    # Strip the square brackets from the input section and then create a
    # list in the form ['x1:x2','y1:y2']
    xylist = section.strip('[]').split(',')
    
    # Create variables containing the single x1, x2, y1 and y2 values
    x1 = int(xylist[0].split(':')[0]) - 1
    x2 = int(xylist[0].split(':')[1]) 
    y1 = int(xylist[1].split(':')[0]) - 1
    y2 = int(xylist[1].split(':')[1]) 
    
    # Return the list in the form [x1 - 1, x2, y1 - 1, y2]
    return [x1, x2, y1, y2]

def gemini_date():

    import datetime
    
    # Define transit as 14:00
    transit = datetime.time(hour=14)
    
    # Define a one day timedelta
    one_day = datetime.timedelta(days=1)
    
    # Get current time in local and UT
    lt_now = datetime.datetime.now()
    ut_now = datetime.datetime.utcnow()
    
    # Format UT and LT dates into strings
    lt_date = lt_now.date().strftime("%Y%m%d")
    ut_date = ut_now.date().strftime("%Y%m%d")
    
    # If before transit, use the earlier date
    fake_date = None
    if lt_now.time()<transit:
        if lt_date==ut_date:
            fake_date = ut_date
        else:
            # UT date changed before transit, use the
            # local date
            fake_date = lt_date
    
    # If before transit, use the later date
    else:
        if lt_date!=ut_date:
            fake_date = ut_date
        else:
            # UT date hasn't changed and it's after transit,
            # so use UT date + 1
            fake_date = (ut_now.date() + one_day).strftime("%Y%m%d")
    
    return fake_date

def parse_percentile(string):
    # Given the type of string that ought to be present in the site condition
    # headers, this function returns the integer percentile number
    #
    # Is it 'Any' - ie 100th percentile?
    if(string == "Any"):
        return 100
    
    # Is it a xx-percentile string?
    m = re.match("^(\d\d)-percentile$", string)
    if(m):
        return int(m.group(1))
    
    # We didn't recognise it
    return None

def filternameFrom(filters):
        
    # reject "open" "grism" and "pupil"
    filters2 = []
    for filt in filters:
        filtlow = filt.lower()
        if "open" in filtlow or "grism" in filtlow or "pupil" in filtlow:
            pass
        else:
            filters2.append(filt)
    
    filters = filters2
    
    # blank means an opaque mask was in place, which of course
    # blocks any other in place filters
    
    if "blank" in filters:
        filtername = "blank"
    elif len(filters) == 0:
        filtername = "open"
    else:
        filters.sort()
        filtername = str("&".join(filters))
    
    return filtername

def get_key_value_dict(dataset, keyword):
    # Since this helper function accesses keywords in the headers of the pixel
    # data extensions, always return a dictionary where the key of the
    # dictionary is an (EXTNAME, EXTVER) tuple
    ret_keyword_value = {}
    
    return_dictionary = False
        
    # Loop over the pixel data extensions in the dataset
    for ext in dataset[pixel_exts]:
        # Get the value of the keyword from the header of each pixel data
        # extension
        value = ext.get_key_value(keyword)
        
        if value is None:
            # The get_key_value() function returns None if a value cannot be
            # found and stores the exception info. Store the last occurance of
            # the exception to the dataset.
            if hasattr(ext, "exception_info"):
                setattr(dataset, "exception_info", ext.exception_info)
            
        # Update the dictionary with the value
        ret_keyword_value.update({(ext.extname(), ext.extver()):value})
    
    try:
        if ret_keyword_value == {}:
            # If the dictionary is still empty, the AstroData object has no
            # pixel data extensions
            raise Errors.CorruptDataError()
        
        unique_values = set(ret_keyword_value.values())
        if len(unique_values) == 1 and None in unique_values:
            # The value of the keyword was not found for any of the pixel data
            # extensions (all the values in the dictionary are equal to None)
            raise dataset.exception_info
        
        return ret_keyword_value
    
    except:
        if not hasattr(dataset, "exception_info"):
            setattr(dataset, "exception_info", sys.exc_info()[1])
        
        return None
