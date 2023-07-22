# gnirs/lookups/orders_XD_GNIRS.py
#
# This file contains a look-up table for GNRIS cross-dispersed (XD) data, with
# information on the changes to the WCS necessary for cutting out the individual
# orders in the data.
#

# primitives_gnirs_crossdispersed imports this dictionary to find the order
# information based on a key generated from the 'telescope', '_prism', 'decker',
# '_grating', and 'camera' attributes of a file.
order_info = {}

# Gemini South, Short camera, 32/mm grating
order_info['Gemini-South_XD_G5509_SC_XD_32/mm_G5506_ShortBlue_G5521'] = (
    (2210, 1660, 1330, 1110, 950, 830),                     # central wavelength, nm
    (-0.645, -0.479, -0.381, -0.322, -0.273, -0.244),  # dispersion, nm/pixel
    (-88.5, -73.5, -69, -77, -89.5, -106.5)                 # crpix adjustment
    )
