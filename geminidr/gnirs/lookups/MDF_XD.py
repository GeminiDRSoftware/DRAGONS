# gnirs/lookups/MDF_XD_GNIRS.py
#
# This file contains a look-up table for GNIRS cross-dispersed (XD) data,
# with information on the locations and widths of the various slits visible
# on the detector. It is added to files in prepare(), from
# GNIRSCrossDispersed.addMDF(). x_ccd refers to the middle of the slit at the
# row y_ccd, which is where tracing of the edges begins in determineSlitEdges()
# (this may need to be other than the center of the array for some configurations)

# primitives_gnirs_crossdispersed imports this dictionary to find the slit
# definitions based on a key generated from the 'telescope', '_prism', 'decker',
# '_grating', and 'camera' attributes of a file.

def get_slit_info(key, central_wavelength=None):
    """
    Returns the slit information for GNIRS cross-dispersed data.

    Returns
    -------
    tuple
        A tuple containing slit information: x_ccd, y_ccd, and width_pixels.
    """
    if callable(slit_info[key]):
        info = slit_info[key](central_wavelength)
    else:
        info = slit_info[key]

    return info

def _gem_north_lxd_lcxd_111_longblue(central_wavelength):
    """
    Returns slit information for the Gemini North Long camera, 111 l/mm grating,
    LXD configuration.

    Parameters
    ----------
    central_wavelength : float
        The central wavelength in nm to determine the slit positions.

    Returns
    -------
    tuple
        A tuple containing x_ccd, y_ccd, and width_pixels.
    """
    if central_wavelength is None:
        raise ValueError("central_wavelength must be provided for this configuration.")
    central_wavelength *= 1.e6  # Convert from meters to um (descriptor default)

    # x position calculation based on central wavelength
    solutions = {
        # order: (slope, constant)  1 degree polynomial
        'order3': (-387.148, 1021.51),
        'order4': (-294.465, 1032.95),
        'order5': (-309.060, 1215.08),
        'order6': (-370.932, 1489.49),
        'order7': (-438.000, 1776.67),  # falls off the detector below 2.06 um
        # order 8 is hardly visible in the flats. Also can fall off the
        # detector at some wavelength settings.
    }
    x_ccd = []
    for solution in solutions:
        slope, constant = solutions[solution]
        x_ccd.append(slope * central_wavelength + constant)

    x_ccd = tuple(x_ccd)
    y_ccd = 512
    width_pixels = 100

    return (x_ccd, y_ccd, width_pixels)

slit_info = {

# Not all configurations have data present in the archive - some notes:
#  * Long camera can be used with the SXD prism, but not the reverse.
#  * There's no reason to use the 10 l/mm grating with the Short camera.
#  * Using the Long camera is best with AO, so it was not used while GNIRS was
#    at Gemini South.

# ------------------------------- Gemini North --------------------------------
# -------------------------------  Short camera
# North, Short, 10 l/mm, SXD
# North, Short, 10 l/mm, LXD
# North, Short, 32 l/mm, SXD
'Gemini-North_SXD_G5536_SCXD_G5531_32/mm_G5533_ShortBlue_G5540': (
    (290, 400, 477, 550, 624, 707),     # x_ccd
    512,     # y_ccd
    47            # width_pixels
    ),
# North, Short, 32 l/mm, LXD
# North, Short, 111 l/mm, SXD
'Gemini-North_SXD_G5536_SCXD_G5531_111/mm_G5534_ShortBlue_G5540': (
    (275, 389, 466, 534, 607, 685),     # x_ccd
    175,     # y_ccd
    47            # width_pixels
    ),
# North, Short, 111 l/mm, LXD
# --------------------------------- Long camera -------------------------------
# North, Long, 10 l/mm, SXD
'Gemini-North_SXD_G5536_SCXD_G5531_10/mm_G5532_LongBlue_G5542': (
    (103, 455, 683, 884),               # x_ccd
    300,               # y_ccd
    140                # width_pixels
    ),
# North, Long, 10 l/mm, LXD
'Gemini-North_LXD_G5535_LCXD_G5531_10/mm_G5532_LongBlue_G5542': (
    (145, 381, 531, 657, 778, 916),     # x_ccd
    175,     # y_ccd
    105      # width_pixels
    ),
# North, Long, 32 l/mm, SXD
'Gemini-North_SXD_G5536_SCXD_G5531_32/mm_G5533_LongBlue_G5542': (
    # There are only 3 slits in the IRAF MDF; the 4th is mostly off-detector
    (208, 563, 792, 995),               # x_ccd
    50,                   # y_ccd
    140                # width_pixels
    ),
# North, Long, 32 l/mm, LXD
'Gemini-North_LXD_G5535_LCXD_G5531_32/mm_G5533_LongBlue_G5542': (
    (220, 430, 580, 714, 857),               # x_ccd
    512,                   # y_ccd
    100                # width_pixels
    ),
# North, Long, 111 l/mm, SXD
# North, Long, 111 l/mm, LXD
'Gemini-North_LXD_G5535_LCXD_G5531_111/mm_G5534_LongBlue_G5542':
    _gem_north_lxd_lcxd_111_longblue,

# ------------------------------- Gemini South --------------------------------
# -------------------------------  Short camera
# South, Short, 10 l/mm, SXD
# South, Short, 10 l/mm, LXD
# South, Short, 32 l/mm, SXD
'Gemini-South_SXD_G5509_SC_XD_32/mm_G5506_ShortBlue_G5521': (
    (325, 435, 512, 584, 659, 742.5),   # x_ccd
    512,     # y_ccd
    43      # width_pixels
    ),
# South, Short, 32 l/mm, LXD
# South, Short, 111 l/mm, SXD
'Gemini-South_SXD_G5509_SC_XD_111/mm_G5505_ShortBlue_G5513': (
    (326, 436, 512, 586, 667, 756),          # x_ccd
    512,          # y_ccd
    48                # width_pixels
    ),
# South, Short, 111 l/mm, LXD
# --------------------------------- Long camera -------------------------------
# South, Long, 10 l/mm, SXD
# South, Long, 10 l/mm, LXD
# South, Long, 32 l/mm, SXD
# South, Long, 32 l/mm, LXD
# South, Long, 111 l/mm, SXD
# South, Long, 111 l/mm, LXD
}

# In some cases changes of components mean the generated keys will be different,
# but the configuration isn't meaningfully affected. Define such cases here.
slit_info['Gemini-South_SXD_G5509_SC_XD_111/mm_G5505_ShortBlue_G5521']  =\
    slit_info['Gemini-South_SXD_G5509_SC_XD_111/mm_G5505_ShortBlue_G5513']


