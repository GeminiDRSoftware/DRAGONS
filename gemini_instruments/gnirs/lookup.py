from collections import namedtuple

# Data structures used by this module

DetectorConfig = namedtuple("Config", "readnoise gain well linearlimit nonlinearlimit")
Config = namedtuple("Config", "mdf offsetsection pixscale mode")

detector_properties = {
    # Taken from http://www.gemini.edu/sciops/instruments/gnirs/imaging/detector-properties-and-read-modes
    # Dictionary key is the read mode and well depth setting
    # Dictionary values are in the following order:
    # readnoise, gain, well, linearlimit, nonlinearlimit
    # readnoise and well are in units of electrons
    ('Very Bright Objects', 'Shallow'): DetectorConfig(155., 13.5, 90000., 0.714286, 1.0),
    ('Bright Objects', 'Shallow'): DetectorConfig(30., 13.5, 90000., 0.714286, 1.0),
    ('Faint Objects', 'Shallow'): DetectorConfig(10., 13.5, 90000., 0.714286, 1.0),
    ('Very Faint Objects', 'Shallow'): DetectorConfig(7., 13.5, 90000., 0.714286, 1.0),
    ('Very Bright Objects', 'Deep'): DetectorConfig(155., 13.5, 180000., 0.714286, 1.0),
    ('Bright Objects', 'Deep'): DetectorConfig(30., 13.5, 180000., 0.714286, 1.0),
    ('Faint Objects', 'Deep'): DetectorConfig(10., 13.5, 180000., 0.714286, 1.0),
    ('Very Faint Objects', 'Deep'): DetectorConfig(7., 13.5, 180000., 0.714286, 1.0),
    }

nominal_zeropoints = {
    # There are none defined...
}

# pixel scales for GNIRS Short and Long cameras
pixel_scale_shrt = 0.15
pixel_scale_long = 0.05

config_dict = {
    # Dictionary keys are in the following order:
    # prism, decker, grating, camera
    # Used every combination of prism, grating and camera available in
    #   gnirs$data/nsappwave.fits r1.43, EH, February 1, 2013
    # Dictionary values are in the following order:
    # mdf, offsetsection, pixscale, mode

    # ShortBlue_G5513 [GS, decommissioned June 2005], 32/mm
    ( "MIR_G5511" , "SC_Long" , "32/mm_G5506" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-ls-short-32-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "32/mm_G5506" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "32/mm_G5506" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "32/mm_G5506" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "** ENG 49450 **" , "32/mm_G5506" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    ( "SXD_G5509" , "SC_XD/IFU" , "32/mm_G5506" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-xd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "SC_XD" , "32/mm_G5506" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-xd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "IFU" , "32/mm_G5506" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-xd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # ShortBlue_G5513 [GS, decommissioned June 2005], 111/mm
    ( "MIR_G5511" , "SC_Long" , "111/mm_G5505" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-ls-short-111-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "111/mm_G5505" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "111/mm_G5505" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "111/mm_G5505" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    ( "SXD_G5509" , "SC_XD/IFU" , "111/mm_G5505" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-xd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "SC_XD" , "111/mm_G5505" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-xd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "IFU" , "111/mm_G5505" , "ShortBlue_G5513" ): Config("gnirs$data/gnirs-xd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # ShortBlue_G5521 [GS, installed June 2005], 32/mm
    ( "MIR_G5511" , "SC_Long" , "32/mm_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ls-short-32-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "32/mm_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "32/mm_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "32/mm_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "** ENG 49450 **" , "32/mm_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    ( "SXD_G5509" , "SC_XD/IFU" , "32/mm_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "SC_XD" , "32/mm_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "IFU" , "32/mm_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # ShortBlue_G5521 [GS, installed June 2005], 32/mmSB
    ( "MIR_G5511" , "SC_Long" , "32/mmSB_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ls-short-32-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "32/mmSB_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "32/mmSB_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "32/mmSB_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    ( "SXD_G5509" , "SC_XD/IFU" , "32/mmSB_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "SC_XD" , "32/mmSB_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "IFU" , "32/mmSB_G5506" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # ShortBlue_G5521 [GS, installed June 2005], 111/mm
    ( "MIR_G5511" , "SC_Long" , "111/mm_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ls-short-111-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "111/mm_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "111/mm_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "111/mm_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    ( "SXD_G5509" , "SC_XD/IFU" , "111/mm_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "SC_XD" , "111/mm_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "IFU" , "111/mm_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # ShortBlue_G5521 [GS, installed June 2005], 111/mmSB
    ( "MIR_G5511" , "SC_Long" , "111/mmSB_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ls-short-111-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "111/mmSB_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "111/mmSB_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "111/mmSB_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    ( "SXD_G5509" , "SC_XD/IFU" , "111/mmSB_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "SC_XD" , "111/mmSB_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),
    ( "SXD_G5509" , "IFU" , "111/mmSB_G5505" , "ShortBlue_G5521" ): Config("gnirs$data/gnirs-xd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # ShortBlue_G5538 [GN, currently in storage, replaced with G5540], 32/mmSB
    ( "MIR_G5537" , "SC_Long" , "32/mmSB_G5533" , "ShortBlue_G5538" ): Config("gnirs$data/gnirsn-ls-short-32-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),

    ( "SB+SXD_G5536" , "SCXD_G5531" , "32/mmSB_G5533" , "ShortBlue_G5538" ): Config("gnirs$data/gnirsn-sxd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # ShortBlue_G5538 [GN, currently in storage, replaced with G5540], 111/mmSB
    ( "MIR_G5537" , "SC_Long" , "111/mmSB_G5534" , "ShortBlue_G5538" ): Config("gnirs$data/gnirsn-ls-short-111-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),

    ( "SB+SXD_G5536" , "SCXD_G5531" , "111/mmSB_G5534" , "ShortBlue_G5538" ): Config("gnirs$data/gnirsn-sxd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # ShortBlue_G5540 [GN, installed October 2012], 32/mmSB
    ( "MIR_G5537" , "SC_Long" , "32/mmSB_G5533" , "ShortBlue_G5540" ): Config("gnirs$data/gnirsn-ls-short-32-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),

    ( "SB+SXD_G5536" , "SCXD_G5531" , "32/mmSB_G5533" , "ShortBlue_G5540" ): Config("gnirs$data/gnirsn-sxd-short-32-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # ShortBlue_G5540 [GN, installed October 2012], 111/mmSB
    ( "MIR_G5537" , "SC_Long" , "111/mmSB_G5534" , "ShortBlue_G5540" ): Config("gnirs$data/gnirsn-ls-short-111-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),

    ( "SB+SXD_G5536" , "SCXD_G5531" , "111/mmSB_G5534" , "ShortBlue_G5540" ): Config("gnirs$data/gnirsn-sxd-short-111-mdf.fits", "[1:190,*]", pixel_scale_shrt, "XD"),

    # LongBlue_G5515, 10/mmLB
    ( "MIR_G5511" , "LC_Long" , "10/mmLB_G5507" , "LongBlue_G5515" ): Config("gnirs$data/gnirsn-ls-long-10-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    ( "LXD_G5508" , "LC_XD" , "10/mmLB_G5507" , "LongBlue_G5515" ): Config("gnirs$data/gnirsn-lxd-long-10-mdf.fits", "[1:190,*]", pixel_scale_long, "XD"),

    ( "SXD_G5509" , "LC_XD" , "10/mmLB_G5507" , "LongBlue_G5515" ): Config("gnirs$data/gnirsn-sxd-long-10-mdf.fits", "[1:190,*]", pixel_scale_long, "XD"),

    # LongBlue_G5515, 32/mmLB
    ( "MIR_G5511" , "LC_Long" , "32/mmLB_G5506" , "LongBlue_G5515" ): Config("gnirs$data/gnirsn-ls-long-32-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    ( "LXD_G5508" , "LC_XD" , "32/mmLB_G5506" , "LongBlue_G5515" ): Config("gnirs$data/gnirsn-lxd-long-32-mdf.fits", "[1:190,*]", pixel_scale_long, "XD"),

    ( "SXD_G5509" , "LC_XD" , "32/mmLB_G5506" , "LongBlue_G5515" ): Config("gnirs$data/gnirsn-sxd-long-32-mdf.fits", "[1:190,*]", pixel_scale_long, "XD"),

    # LongBlue_G5515, 111/mmLB
    ( "MIR_G5511" , "LC_Long" , "111/mmLB_G5505" , "LongBlue_G5515" ): Config("gnirs$data/gnirsn-ls-long-111-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    # LongBlue_G5542 [GN, installed October 2009], 10/mmLB
    ( "MIR_G5537" , "LC_Long" , "10/mmLB_G5532" , "LongBlue_G5542" ): Config("gnirs$data/gnirsn-ls-long-10-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    ( "LB+LXD_G5535" , "LCXD_G5531" , "10/mmLBLX_G5532" , "LongBlue_G5542" ): Config("gnirs$data/gnirsn-lxd-long-10-mdf.fits", "[1:190,*]", pixel_scale_long, "XD"),

    ( "LB+SXD_G5536" , "LCXD_G5531" , "10/mmLBSX_G5532" , "LongBlue_G5542" ): Config("gnirs$data/gnirsn-sxd-long-10-mdf.fits", "[1:190,*]", pixel_scale_long, "XD"),

    # LongBlue_G5542 [GN, installed October 2009], 32/mmLB
    ( "MIR_G5537" , "LC_Long" , "32/mmLB_G5533" , "LongBlue_G5542" ): Config("gnirs$data/gnirsn-ls-long-32-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    ( "LB+LXD_G5535" , "LCXD_G5531" , "32/mmLB_G5533" , "LongBlue_G5542" ): Config("gnirs$data/gnirsn-lxd-long-32-mdf.fits", "[1:190,*]", pixel_scale_long, "XD"),

    ( "LB+SXD_G5536" , "LCXD_G5531" , "32/mmLB_G5533" , "LongBlue_G5542" ): Config("gnirs$data/gnirsn-sxd-long-32-mdf.fits", "[1:190,*]", pixel_scale_long, "XD"),

    # LongBlue_G5542 [GN, installed October 2009], 111/mmLB
    ( "MIR_G5537" , "LC_Long" , "111/mmLB_G5534" , "LongBlue_G5542" ): Config("gnirs$data/gnirsn-ls-long-111-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    # ShortRed_G5514 [GS, decommissioned June 2005], 111/mm
    ( "MIR_G5511" , "SC_Long" , "111/mm_G5505" , "ShortRed_G5514" ): Config("gnirs$data/gnirs-ls-short-111-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "111/mm_G5505" , "ShortRed_G5514" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "111/mm_G5505" , "ShortRed_G5514" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "111/mm_G5505" , "ShortRed_G5514" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    # ShortRed_G5522 [GS, installed June 2005], 32/mmSR
    ( "MIR_G5511" , "SC_Long" , "32/mmSR_G5506" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ls-short-32-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "32/mmSR_G5506" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "32/mmSR_G5506" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "32/mmSR_G5506" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ifu-short-32-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    # ShortRed_G5522 [GS, installed June 2005]. 111/mm
    ( "MIR_G5511" , "SC_Long" , "111/mm_G5505" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ls-short-111-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "111/mm_G5505" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "111/mm_G5505" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "111/mm_G5505" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    # ShortRed_G5522 [GS, installed June 2005]. 111/mmSR
    ( "MIR_G5511" , "SC_Long" , "111/mmSR_G5505" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ls-short-111-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),
    ( "MIR_G5511" , "SC_XD/IFU" , "111/mmSR_G5505" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "SC_XD" , "111/mmSR_G5505" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),
    ( "MIR_G5511" , "IFU" , "111/mmSR_G5505" , "ShortRed_G5522" ): Config("gnirs$data/gnirs-ifu-short-111-mdf2.fits", "[900:1024,*]", pixel_scale_shrt, "IFU"),

    # ShortRed_G5539 [GN, currently in storage], 32/mmSR
    ( "MIR_G5537" , "SC_Long", "32/mmSR_G5533", "ShortRed_G5539"): Config("gnirs$data/gnirsn-ls-short-32-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),

    # ShortRed_G5539 [GN, currently in storage], 111/mmSR
    ( "MIR_G5537" , "SC_Long", "111/mmSR_G5534", "ShortRed_G5539"): Config("gnirs$data/gnirsn-ls-short-111-mdf.fits", "[850:1024,*]", pixel_scale_shrt, "LS"),

    # LongRed_G5516, 10/mmLR
    ( "MIR_G5511" , "LC_Long" , "10/mmLR_G5507" , "LongRed_G5516"): Config("gnirs$data/gnirsn-ls-long-10-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    # LongRed_G5516, 32/mmLR
    ( "MIR_G5511" , "LC_Long" , "32/mmLR_G5506" , "LongRed_G5516" ): Config("gnirs$data/gnirs-ls-long-32-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    # LongRed_G5516. 111/mmLR
    ( "MIR_G5511" , "LC_Long" , "111/mmLR_G5505" , "LongRed_G5516" ): Config("gnirs$data/gnirs-ls-long-111-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    # LongRed_G5543 [GN, installed October 2009], 10/mmLR
    ( "MIR_G5537" , "LC_Long" , "10/mmLR_G5532" , "LongRed_G5543"): Config("gnirs$data/gnirsn-ls-long-10-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    # LongRed_G5543 [GN, installed October 2009], 32/mmLR
    ( "MIR_G5537" , "LC_Long" , "32/mmLR_G5533" , "LongRed_G5543" ): Config("gnirs$data/gnirsn-ls-long-32-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),

    # LongRed_G5543 [GN, installed October 2009], 111/mmLR
    ( "MIR_G5537" , "LC_Long" , "111/mmLR_G5534" , "LongRed_G5543" ): Config("gnirs$data/gnirsn-ls-long-111-mdf.fits", "[1:30,*]", pixel_scale_long, "LS"),
}

# Key is (LNRS, NDAVGS)
read_modes = {
    (32, 16): "Very Faint Objects",
    (16, 16): "Faint Objects",
    ( 1, 16): "Bright Objects",
    ( 1,  1): "Very Bright Objects"
}

filter_wavelengths = {
    'YPHOT'       : 1.0300,
    'X'           : 1.1000,
    'X_(order_6)' : 1.1000,
    'JPHOT'       : 1.2500,
    'J_(order_5)' : 1.2700,
    'H'           : 1.6300,
    'H_(order_4)' : 1.6300,
    'H2'          : 2.1250,
    'K_(order_3)' : 2.1950,
    'KPHOT'       : 2.2200,
    'PAH'         : 3.2950,
    'L'           : 3.5000,
    'L_(order_2)' : 3.5000,
    'M'           : 5.1000,
    'M_(order_1)' : 5.1000,
}

dispersion_by_config = {
    # Dictionary keys are in the following order:
    # "grating, camera".
    # Dictionary values are in the following order:
    # "Filter": dispersion
    # Dispersion values are in A/pix.
    # The dispersion values are based on wvl. coverages for each filter/mode listed in GNIRS instrument pages.
    ("10/mm, Short")  : {"M": -19.39},
    ("10/mm, Long")   : {"X": -3.24,  "J": -3.89,   "H": -4.85, "K": -6.47, "L": -9.72, "M": -19.43},
    ("32/mm, Short")  : {"X": -3.23,  "J": -3.88,   "H": -4.84, "K": -6.46, "L": -9.69, "M": -19.34},
    ("32/mm, Long")   : {"X": -1.07,  "J": -1.29,   "H": -1.62, "K": -2.16, "L": -3.24, "M": -6.45},
    ("111/mm, Short") : {"X": -0.92,  "J": -1.10,   "H": -1.39, "K": -1.85, "L": -2.73, "M": -5.62},
    ("111/mm, Long")  : {"X": -0.309, "J": -0.371,  "H": -0.464,"K": -0.618,"L": -0.922,"M": -1.875}
}
