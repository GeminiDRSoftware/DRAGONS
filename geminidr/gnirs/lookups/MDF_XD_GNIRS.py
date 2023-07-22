# gnirs/lookups/MDF_XD_GNIRS.py
#
# This file contains a look-up table for GNIRS cross-dispersed (XD) data,
# with information on the locations and widths of the various slits visible
# on the detector. It is added to files in prepare(), from
# GNIRSCrossDispersed.addMDF().

# primitives_gnirs_crossdispersed imports this dictionary to find the slit
# definitions based on a key generated from the 'telescope', '_prism', 'decker',
# '_grating', and 'camera' attributes of a file.
slit_info = {}

# Gemini South, Short camera, 32/mm grating
slit_info['Gemini-South_SXD_G5509_SC_XD_32/mm_G5506_ShortBlue_G5521'] = (
    (1, 2, 3, 4, 5, 6),                 # slit ID
    (325, 435, 512, 584, 659, 742.5),   # x_ccd
    (512, 512, 512, 512, 512, 512),     # y_ccd
    (3, 4, 5, 6, 7, 8),                 # specorder
    (7.0, 7.0, 7.0, 7.0, 7.0, 7.0),     # width_arcsec
    (43.8, 42.8, 42.8, 43, 43, 43)      # width_pixels
    )
