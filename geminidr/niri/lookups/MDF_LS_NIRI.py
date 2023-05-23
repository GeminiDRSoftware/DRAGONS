# niri/lookups/MDF_LS.py
#
# This file contains a look-up table for NIRI long-slit (LS) data,
# with information on the locations and widths of the various slits visible
# on the detector. It is added to files in prepare(), from
# NIRILongSlit.addMDF().

# primitives_niri_longslit imports this dictionary to find the slit
# definitions based on a key generated from the focal plane and f-ratio
# attributes of a file.

# The NIRI webpage lists "f/32 4-pixel" and "f/32 7-pixel" options, but the
# archive only has options for "f32-6pix" and "f32-9pix". The component numbers
# make it clear that these are the same as the 4 and 7 pixels configurations
# listed.
# Columns are the y-value of slit center on the sensor, then the slitlength in
# arcsec and pixels.
slit_info = {
    'f6-2pix':   (501, 51.4, 439.),
    'f6-2pixBl': (499, 51.6, 442),
    'f6-4pix':   (498, 112.5, 962),
    'f6-4pixBl': (494, 51.5, 440),
    'f6-6pix':   (505, 110, 965.81),
    'f6-6pixBl': (496.5, 51.5, 439),
    # The nominal slit length for the f/32 camera is 22", but
    # the illuminated region goes off the edge and is longer
    # than the nominal 1,000 pixels in each case, so 22.2" is a
    # minimum guess, assuming the pixel scale is correct.
    'f32-6pix':  (506, 22.2, 1015),
    'f32-9pix':  (508, 22.2, 1015),
    'f32-10pix': (507, 22.2, 1015)
    }
