"""
Module provides utility functions to manipulate Gemini specific metadata
strings. Eg., filter names, time strings, etc.
"""
import re
import datetime

# ------------------------------------------------------------------------------
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
    """
    Return a date string of the form 'YYYYMMDD' for the current
    operational date defined by a transit time at 14.00h (LT).

    parameters: <void>
    return:     <str>     date string, eg., '20130904'
    """
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

