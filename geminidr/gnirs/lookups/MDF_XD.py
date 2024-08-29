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
    (1, 2, 3, 4, 5, 6),                 # slit ID
    (290, 400, 477, 550, 624, 707),     # x_ccd
    (512, 512, 512, 512, 512, 512),     # y_ccd
    (3, 4, 5, 6, 7, 8),                 # specorder
    (7.0, 7.0, 7.0, 7.0, 7.0, 7.0),     # width_arcsec
    (46, 47, 47, 47, 47, 47)            # width_pixels
    ),
# North, Short, 32 l/mm, LXD
# North, Short, 111 l/mm, SXD
'Gemini-North_SXD_G5536_SCXD_G5531_111/mm_G5534_ShortBlue_G5540': (
    (1, 2, 3, 4, 5, 6),                 # slit ID
    (275, 389, 466, 534, 607, 685),     # x_ccd
    (175, 175, 175, 175, 175, 175),     # y_ccd
    (3, 4, 5, 6, 7, 8),                 # specorder
    (5.0, 5.0, 5.0, 5.0, 5.0, 5.0),     # width_arcsec
    (46, 47, 47, 47, 47, 47)            # width_pixels
    ),
# North, Short, 111 l/mm, LXD
# --------------------------------- Long camera -------------------------------
# North, Long, 10 l/mm, SXD
'Gemini-North_SXD_G5536_SCXD_G5531_10/mm_G5532_LongBlue_G5542': (
    (1, 2, 3, 4),                       # slit ID
    (103, 455, 683, 884),               # x_ccd
    (300, 300, 300, 300),               # y_ccd
    (3, 4, 5, 6),                       # specorder
    (7.0, 7.0, 7.0, 7.0),               # width_arcsec
    (140, 140, 140, 140)                # width_pixels
    ),
# North, Long, 10 l/mm, LXD
'Gemini-North_LXD_G5535_LCXD_G5531_10/mm_G5532_LongBlue_G5542': (
    (1, 2, 3, 4, 5, 6),                 # slit ID
    (145, 381, 531, 657, 778, 916),     # x_ccd
    (175, 175, 175, 175, 175, 175),     # y_ccd
    (3, 4, 5, 6, 7, 8),                 # specorder
    (5.0, 5.0, 5.0, 5.0, 5.0, 5.0),     # width_arcsec
    (104, 106, 104, 105, 105, 105)      # width_pixels
    ),
# North, Long, 32 l/mm, SXD
'Gemini-North_SXD_G5536_SCXD_G5531_32/mm_G5533_LongBlue_G5542': (
    (1, 2, 3, 4),                       # slit ID
    (208, 563, 792, 995),               # x_ccd
    (50, 50, 50, 50),                   # y_ccd
    (3, 4, 5, 6),                       # specorder
    (7.0, 7.0, 7.0, 7.0),               # width_arcsec
    (140, 140, 140, 140)                # width_pixels
    ),
# North, Long, 32 l/mm, LXD
'Gemini-North_LXD_G5535_LCXD_G5531_32/mm_G5533_LongBlue_G5542': (
    (1, 2, 3, 4),                       # slit ID
    (265, 468, 621, 764),               # x_ccd
    (50, 50, 50, 50),                   # y_ccd
    (3, 4, 5, 6),                       # specorder
    (5.0, 5.0, 5.0, 5.0),               # width_arcsec
    (100, 100, 100, 100)                # width_pixels
    ),
# North, Long, 111 l/mm, SXD
# North, Long, 111 l/mm, LXD
'Gemini-North_LXD_G5535_LCXD_G5531_111/mm_G5534_LongBlue_G5542': (
    (1, 2, 3, 4),                       # slit ID
    (267, 462, 620, 775),               # x_ccd
    (256, 256, 256, 256),               # y_ccd
    (3, 4, 5, 6),                       # specorder
    (5.0, 5.0, 5.0, 5.0),               # width_arcsec
    (100, 100, 100, 100)                # width_pixels
    ),

# ------------------------------- Gemini South --------------------------------
# -------------------------------  Short camera
# South, Short, 10 l/mm, SXD
# South, Short, 10 l/mm, LXD
# South, Short, 32 l/mm, SXD
'Gemini-South_SXD_G5509_SC_XD_32/mm_G5506_ShortBlue_G5521': (
    (1, 2, 3, 4, 5, 6),                 # slit ID
    (325, 435, 512, 584, 659, 742.5),   # x_ccd
    (512, 512, 512, 512, 512, 512),     # y_ccd
    (3, 4, 5, 6, 7, 8),                 # specorder
    (7.0, 7.0, 7.0, 7.0, 7.0, 7.0),     # width_arcsec
    (43.8, 42.8, 42.8, 43, 43, 43)      # width_pixels
    ),
# South, Short, 32 l/mm, LXD
# South, Short, 111 l/mm, SXD
'Gemini-South_SXD_G5509_SC_XD_111/mm_G5505_ShortBlue_G5513': (
    (1, 2, 3, 4, 5),                    # slit ID
    (326, 436, 512, 586, 667),          # x_ccd
    (512, 512, 512, 512, 512),          # y_ccd
    (3, 4, 5, 6, 7),                    # specorder
    (7.8, 7.7, 7.5, 7.0, 7.0),          # width_arcsec
    (52, 51, 49, 46, 46)                # width_pixels
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
