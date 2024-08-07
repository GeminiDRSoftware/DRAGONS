# gnirs/LS_MDF_table.py
#
# This file contains a look-up table for GNIRS long-slit (LS) data,
# with information on the locations and widths of the various slits visible
# on the detector. It is added to files in prepare(), from
# GNIRSLongSlit.addMDF().

# primitives_gnirs_longslit imports this dictionary to find the slit
# definitions based on a key generated from the 'telescope', '_grating', and
# 'camera' attributes of a file.

# columns are x-position of center of slit, width in arcsec, width in pixels
slit_info = {
    'South_111/mm_LongBlue':  (474, 49, 980),
    'South_111/mm_LongRed':   (474, 49, 980), # no data in archive
    'South_111/mm_ShortBlue': (474, 99, 660),
    'South_111/mm_ShortRed':  (485, 99, 660),
    'South_32/mm_LongBlue':   (411, 49, 980), # no data in archive
    'South_32/mm_LongRed':    (411, 49, 980),
    'South_32/mm_ShortBlue':  (459, 99, 660),
    'South_32/mm_ShortRed':   (504, 99, 660),
    'North_111/mm_LongBlue':  (525, 49.75, 980),
    'North_111/mm_LongRed':   (471.5, 46.65, 935),
    'North_111/mm_ShortBlue': (474, 99, 660),
    'North_111/mm_ShortRed':  (480, 99, 660),
    'North_32/mm_LongBlue':   (520.5, 49.65, 993),
    'North_32/mm_LongRed':    (492, 49.3, 986),
    'North_32/mm_ShortBlue':  (496, 99, 660),
    'North_32/mm_ShortRed':   (504, 99, 660),
    'North_10/mm_LongBlue':   (541, 49, 980),
    'North_10/mm_LongRed':    (465, 49, 980)
    }

# For the LongRed camera on Gemini-North, at least for the 10/mm grating but
# possibly also for the other gratings, the slit is sometimes offset so that
# the full 49" length isn't visible, causing the single on-detector edge to
# vary in position. It seems to be a mostly bimodal distribution, with the slit
# length appearing to be either ~45" or ~49, so the values in the table here
# are something of a compromise which should work for both cases.

# The LongBlue camera can have the illuminated region shifted from side to side;
# e.g., N20120419S0097 (right edge visible) vs. N20121213S0312 (left edge
# visible). The current center is a compromise that allows both to be found.
