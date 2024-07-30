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

# Gemini South, Short camera, 32 l/mm grating
order_info['Gemini-South_SXD_G5509_SC_XD_32/mm_G5506_ShortBlue_G5521'] = (
    (2210, 1660, 1330, 1110, 950, 830),                # central wavelength, nm
    (-0.645, -0.479, -0.381, -0.322, -0.273, -0.244),  # dispersion, nm/pixel
)

# Gemini North, Long camera, 32 l/mm grating
order_info['Gemini-North_SXD_G5536_SCXD_G5531_32/mm_G5533_ShortBlue_G5540'] = (
    (2210, 1660, 1330, 1110, 950, 830),
    (-0.645, -0.479, -0.381, -0.322, -0.273, -0.244),
)

# Gemini North, Long camera, 10 l/mm grating
order_info['Gemini-North_LXD_G5535_LCXD_G5531_10/mm_G5532_LongBlue_G5542'] = (
    (2210, 1660, 1330, 1110, 950, 830),
    (-0.64504, -0.4847, -0.3895, -0.322, -0.273, -0.244),
)


# Gemini North, Short camera, 111 l/mm grating
order_info['Gemini-North_SXD_G5536_SCXD_G5531_111/mm_G5534_ShortBlue_G5540'] = (
    (2210, 1660, 1330, 1110, 950, 830),
    (-0.1853, -0.13882, -0.11117, -0.09242, -0.08, -0.0694),
)
