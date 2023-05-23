# gnirs/xd_MDF_table.py
#
# This file contains a look-up table for GNIRS cross-dispersed (XD) data,
# with information on the locations and widths of the various slits visible
# on the detector. It is added to files in prepare(), from
# GNIRSCrossDispered.addMDF().

from astropy.table import Table

# primitives_gnirs_crossdispersed imports this dictionary to find the slit
# definitions based on a key generated from the 'telescope', '_prism', 'decker',
# '_grating', and 'camera' attributes of a file.
mdf_table = {}

# Gemini South, Short camera, 32/mm grating
columns = ('slit_id', 'x_ccd', 'y_ccd', 'specorder', 'slitlength_asec',
           'slitlength_mm', 'slitlength_pixels')

slit_id = (1, 2, 3, 4, 5, 6)
x_ccd = (325, 435, 512, 584, 659, 742.5)
y_ccd = (512, 512, 512, 512, 512, 512)
specorder = (3, 4, 5, 6, 7, 8)
width_asec = (0, 0, 0, 0, 0, 0)
width_mm = (0, 0, 0, 0, 0, 0)
width_pixels = (43.8, 42.8, 42.8, 43, 43, 43)

mdf_table['Gemini-South_XD_G5509_SC_XD_32/mm_G5506_ShortBlue_G5521'] = Table(
    [slit_id, x_ccd, y_ccd, specorder, width_asec, width_mm, width_pixels],
    names=columns)
