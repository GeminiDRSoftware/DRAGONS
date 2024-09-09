# f2/lookups/MDF_LS.py
#
# This file contains a look-up table for F2 long-slit (LS) data,
# with information on the locations and widths of the various slits visible
# on the detector. It is added to files in prepare(), from
# F2LongSlit.addMDF().

# primitives_f2_longslit imports this dictionary to find the slit
# definitions based on a key generated from the focal plane and f-ratio
# attributes of a file.

# This dictionary contains the following information for each slit configuration:
# x-coordinate of the pixel in the center of the slit, slit length in arcsec,
# slit length in pixels. The F2 pixel scale is 0.18 arsec/pixel, according to
# the instrument webpage.
slit_info = {
    '1pix': (763.5, 1509),
    '2pix': (769.5, 1473),
    '3pix': (770.0, 1510),
    '4pix': (772.5, 1503),
    '6pix': (777.0, 1510),
    '8pix': (771.0, 1510)
    }
