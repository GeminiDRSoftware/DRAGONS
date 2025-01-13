from collections import namedtuple

# Data structures used by this module

NonLinCoeffs = namedtuple("NonLinCoeffs", "max_counts time_delta gamma eta")
CenwaveAndRes = namedtuple("CenwaveAndRes","cenwave cenpixwave resolution")

filter_wavelengths = {
    'Jcon(1065)'   : 1.0650,
    'HeI'          : 1.0830,
    'Pa(gamma)'    : 1.0940,
    'Jcon(112)'    : 1.1220,
    'Jcon(121)'    : 1.2070,
    'Pa(beta)'     : 1.2820,
    'H-con(157)'   : 1.5700,
    'CH4(short)'   : 1.5800,
    'H'            : 1.6500,
    'CH4(long)'    : 1.6900,
    'H2Oice(2045)' : 2.0450,
    'HeI(2p2s)'    : 2.0590,
    'Kcon(209)'    : 2.0975,
    'K(prime)'     : 2.1200,
    'H2 1-0 S1'    : 2.1239,
    'K(short)'     : 2.1500,
    'H2 2-1 S1'    : 2.2465,
    'Br(gamma)'    : 2.1686,
    'Kcon(227)'    : 2.2718,
    'CH4ice(2275)' : 2.2750,
    'CO 2-0 (bh)'  : 2.2890,
    'H2Oice'       : 3.0500,
    'hydrocarb'    : 3.2950,
    'L(prime)'     : 3.7800,
    'Br(alpha)Con' : 3.9900,
    'Br(alpha)'    : 4.0520,
    'M(prime)'     : 4.6800,
    }

array_properties = {
    # Database for nprepare.cl
    # Date: 2004 July 6
    # Author: Joe Jensen, Gemini Observatory
    # The long 6-pix and 4-pix centered slits are currently installed
    #
    # Array characteristics
    "readnoise"  : 70,          # electrons (1 read pair, 1 digital av.)
    "medreadnoise" : 35.,       # electrons (1 read pair, 16 dig av.)
    "lowreadnoise" : 12.3,      # electrons (16 read pairs, 16 dig av.)
    "gain"         : 12.3,      # electrons/ADU
    "shallowwell"  : 200000.,   # electrons full-well
    "deepwell"     : 280000.,   # electrons full-well
    "shallowbias"  : -0.6,      # detector bias (V)
    "deepbias"     : -0.87,     # detector bias (V)
    "linearlimit"  : 0.7,       # non-linear regime (fraction of saturation)
}

nonlin_coeffs = {
    # In the following form for NIRI data:
    #("read_mode", naxis2, "well_depth_setting"):
    #    (maximum counts, exposure time correction, gamma, eta)
    #      for data in *electrons*, not ADU (which is what matters, anyway!)
    #      max_counts was used in nirlin.py for flagging pixels, so is not
    #        used in DRAGONS
    # These lines show the original ADU-based numbers, for provenance
    #("Low Background", 1024, "Shallow"):
    #    NonLinCoeffs(12000*12.3, 1.2662732, 7.3877618e-06/12.3, 1.940645271e-10/12.3**2),
    #("Medium Background", 1024, "Shallow"):
    #    NonLinCoeffs(12000*12.3, 0.09442515154, 3.428783846e-06/12.3, 4.808353308e-10/12.3**2),
    #"Medium Background", 256, "Shallow"):
    #    NonLinCoeffs(12000*12.3, 0.01029262589, 6.815415667e-06/12.3, 2.125210479e-10/12.3**2),
    #("High Background", 1024, "Shallow"):
    #    NonLinCoeffs(12000*12.3, 0.009697324059, 3.040036696e-06/12.3, 4.640788333e-10/12.3**2),
    #("High Background", 1024, "Deep"):
    #    NonLinCoeffs(21000*12.3, 0.007680816203, 3.581914163e-06/12.3, 1.820403678e-10/12.3**2),
    ("Low Background", 1024, "Shallow"):
        NonLinCoeffs(12000*12.3, 1.2662732, 6.006310407e-07, 1.282732019e-12),
    ("Medium Background", 1024, "Shallow"):
        NonLinCoeffs(12000*12.3, 0.09442515154, 2.787629143e-07, 3.178236042e-12),
    ("Medium Background", 256, "Shallow"):
        NonLinCoeffs(12000*12.3, 0.01029262589, 5.540988347e-07, 1.404726339e-12),
    ("High Background", 1024, "Shallow"):
        NonLinCoeffs(12000*12.3, 0.009697324059, 2.471574550e-07, 3.067478573e-12),
    ("High Background", 1024, "Deep"):
        NonLinCoeffs(21000*12.3, 0.007680816203, 2.912125336e-07, 1.203254464e-12),
    }

spec_sections = {
    #
    # Camera+FPmask        SPECSEC1           SPECSEC2          SPECSEC3
    #
    "f6f6-2pix_G5211"   :   (  "[1:1024,276:700]" ,  "none", "none" ),
    "f6f6-4pix_G5212"   :   (  "[1:1024,1:1024]"  ,  "none", "none" ),
    "f6f6-6pix_G5213"   :   (  "[1:1024,1:1024]"  ,  "none", "none" ),
    "f6f6-2pixBl_G5214" :   (  "[1:1024,276:700]" ,  "none", "none" ),
    "f6f6-4pixBl_G5215" :   (  "[1:1024,276:700]" ,  "none", "none" ),
    "f6f6-6pixBl_G5216" :   (  "[1:1024,276:700]" ,  "none", "none" ),
    "f6f6-4pix_G5222"   :   (  "[1:1024,276:700]" ,  "none", "none" ),
    "f6f6-6pix_G5223"   :   (  "[1:1024,276:700]" ,  "none", "none" )
}

# Refactored by CJS 2016-10-14
# Note that the components of the key have to be alphabetized
filter_name_mapping = {
    ('PK50', 'Y') :             'Y',
    ('J') :                     'J',
    ('H') :                     'H',
    ('Kprime') :                'K(prime)',
    ('K') :                     'K',
    ('Lprime') :                'L(prime)',
    ('Mprime') :                'M(prime)',
    ('Jcon(121)', 'PK50') :     'Jcon(121)',
    ('Hcon(157)', 'PK50') :     'H-con(157)',
    ('CH4short', 'PK50') :      'CH4(short)',
    ('CH4long', 'PK50') :       'CH4(long)',
    ('FeII', 'PK50') :          'FeII',
    ('HeI(2p2s)', 'Kprime') :   'HeI(2p2s)',
    ('H2Oice204') :             'H2Oice(2045)',
    ('kcon(209)') :             'Kcon(209)',
    ('kcon(227)') :             'Kcon(227)',
    ('H2v=1-0S1', 'K') :        'H2 1-0 S1',
    ('Brgamma') :               'Br(gamma)',
    ('CH4ice227') :             'CH4ice(2275)',
    ('Jsort') :                 'J order sort',
    ('Hsort', 'PK50') :         'H order sort',
    ('Ksort') :                 'K order sort',
    ('Lsort') :                 'L order sort',
    ('Msort') :                 'M order sort',
    ('Jcon1065', 'PK50') :      'Jcon(1065)',
    ('Jcon(112)', 'PK50') :     'Jcon(112)',
    ('Kshort') :                'K(short)',
    ('H2v=2-1S1') :             'H2 2-1 S1',
    ('PK50', 'Pabeta') :        'Pa(beta)',
    ('PaGamma') :               'Pa(gamma)',
    ('HeI') :                   'HeI',
    ('Bra') :                   'Br(alpha)',
    ('H2Oice') :                'H2O ice',
    ('hydrocarb') :             'hydrocarb',
    ('Bracont') :               'Br(alpha)Con',
    ('COv=2-0bh') :             'CO 2-0 (bh)',
}

nominal_zeropoints = {
    # Table of NIRI Nominal Zeropoint magnitudes
    # By Camera and Filter ID.
    # From Photometric zero points in "sciops.instruments
    #   .PerformanceMonitoring.DataProducts.NIRI"

    # NOTE NOTE NOTE
    # The numbers on that web page are for 1 ADU/s not 1 electron/s, so we need to
    # add 2.5*log_10(gain) = 2.5*log_10(12.3) = 2.72 to each of them.

    # BAND  CAMERA: Zeropoint (average)
    ('Y',  'f6'):  22.99+2.72,
    ('J',  'f6'):  23.97+2.72,
    ('J',  'f32'): 23.33+2.72,
    ('H',  'f6'):  24.05+2.72,
    ('H',  'f32'): 23.62+2.72,
    ('K',  'f6'):  23.43+2.72,
    ('K',  'f32'): 22.99+2.72,
    ('K(short)',  'f6'):  23.40+2.72,
    ('K(short)',  'f32'): 22.95+2.72,
    ('K(prime)',  'f6'):  23.68+2.72,
    ('K(prime)',  'f32'): 23.60+2.72,
}

# Adapted from old AD_Config/IR/appwave.py. Since NIRI is the only instrument
# to use this, removed all extraneous columns
# Dictionary keys:
# (camera, focal plane mask, grism)
# Dictionary values:
# central wavelength (nm) (apparently this is the center of "useful wvl range"; it is used as cenwave in the archive),
# wavelength at central pixel (nm) (as measured by OS),
# resolution (from NIRI webpage)
spec_wavelengths = {
    ('f6', 'f6-2pix',   'Jgrism') : CenwaveAndRes(1233.0, 1211.0, 770),
    ('f6', 'f6-4pix',   'Jgrism') : CenwaveAndRes(1233.0, 1211.0, 610),
    ('f6', 'f6-6pix',   'Jgrism') : CenwaveAndRes(1233.0, 1211.0, 460),
    ('f6', 'f6-2pixBl', 'Jgrism') : CenwaveAndRes(1184.0, 1162.0, 770),
    ('f6', 'f6-4pixBl', 'Jgrism') : CenwaveAndRes(1184.0, 1162.0, 650),
    ('f6', 'f6-6pixBl', 'Jgrism') : CenwaveAndRes(1184.0, 1162.0, 480),
    ('f6', 'f6-2pix',   'Hgrism') : CenwaveAndRes(1695.0, 1665.0, 1650),
    ('f6', 'f6-4pix',   'Hgrism') : CenwaveAndRes(1695.0, 1665.0, 825),
    ('f6', 'f6-6pix',   'Hgrism') : CenwaveAndRes(1695.0, 1665.0, 520),
    ('f6', 'f6-2pixBl', 'Hgrism') : CenwaveAndRes(1625.0, 1595.0, 1650),
    ('f6', 'f6-4pixBl', 'Hgrism') : CenwaveAndRes(1625.0, 1595.0, 940),
    ('f6', 'f6-6pixBl', 'Hgrism') : CenwaveAndRes(1625.0, 1595.0, 550),
    ('f6', 'f6-2pix',   'Kgrism') : CenwaveAndRes(2263.0, 2223.0, 1300),
    ('f6', 'f6-4pix',   'Kgrism') : CenwaveAndRes(2263.0, 2223.0, 780),
    ('f6', 'f6-6pix',   'Kgrism') : CenwaveAndRes(2263.0, 2223.0, 520),
    ('f6', 'f6-2pixBl', 'Kgrism') : CenwaveAndRes(2167.0, 2127.0, 1300),
    ('f6', 'f6-4pixBl', 'Kgrism') : CenwaveAndRes(2167.0, 2127.0, 780),
    ('f6', 'f6-6pixBl', 'Kgrism') : CenwaveAndRes(2167.0, 2127.0, 520),
    ('f6', 'f6-2pix',   'Lgrism') : CenwaveAndRes(3574.0, 3521.0, 1100),
    ('f6', 'f6-4pix',   'Lgrism') : CenwaveAndRes(3574.0, 3521.0, 690),
    ('f6', 'f6-6pix',   'Lgrism') : CenwaveAndRes(3574.0, 3521.0, 460),
    ('f6', 'f6-2pixBl', 'Lgrism') : CenwaveAndRes(3435.0, 3365.0, 1100),
    ('f6', 'f6-4pixBl', 'Lgrism') : CenwaveAndRes(3435.0, 3365.0, 770),
    ('f6', 'f6-6pixBl', 'Lgrism') : CenwaveAndRes(3435.0, 3365.0, 490),
    ('f6', 'f6-2pix',   'Mgrism') : CenwaveAndRes(5140.0, 5080.0, 1100),
    ('f6', 'f6-4pix',   'Mgrism') : CenwaveAndRes(5140.0, 5080.0, 770),
    ('f6', 'f6-6pix',   'Mgrism') : CenwaveAndRes(5140.0, 5080.0, 460),
    ('f6', 'f6-2pixBl', 'Mgrism') : CenwaveAndRes(4940.0, 4940.0, 1100), # copy from non-Bl
    ('f6', 'f6-4pixBl', 'Mgrism') : CenwaveAndRes(4940.0, 4940.0, 770), # copy from non-Bl
    ('f6', 'f6-6pixBl', 'Mgrism') : CenwaveAndRes(4940.0, 4940.0, 460), # copy from non-Bl
    ('f32', 'f32-6pix',  'Jgrism') : CenwaveAndRes(1203.0, 1203.0, 1000), # same as f32 4-pix
    ('f32', 'f32-9pix',  'Jgrism') : CenwaveAndRes(1203.0, 1203.0, 620), # same as f32 7-pix
    ('f32', 'f6-2pix',  'Jgrism') : CenwaveAndRes(1203.0, 1203.0, 450), # same as f32 10-pix
    ('f32', 'f32-6pix',  'Hgrism') : CenwaveAndRes(1641.2, 1641.2, 880),
    ('f32', 'f32-9pix',  'Hgrism') : CenwaveAndRes(1641.2, 1641.2, 630),
    ('f32', 'f6-2pix',  'Hgrism') : CenwaveAndRes(1641.2, 1641.2, 500),
    ('f32', 'f32-6pix',  'Kgrism') : CenwaveAndRes(2184.0, 2184.0, 1280),
    ('f32', 'f32-9pix',  'Kgrism') : CenwaveAndRes(2184.0, 2184.0, 775),
    ('f32', 'f6-2pix',  'Kgrism') : CenwaveAndRes(2184.0, 2184.0, 570)
}


dispersion_by_config = {
    # Dictionary keys are in the following order:
    # 'camera, disperser'
    # Dispersion is in nm/pix.
    # This is a concise version of gnirs$data/nsappwave.fits table.
    ("f6", "Jgrism")  : -0.3604,
    ("f6", "Hgrism")  : -0.5157,
    ("f6", "Kgrism")  : -0.7086,
    ("f6", "Lgrism")  : -1.139,
    ("f6", "Mgrism")  : -1.651,
    ("f32", "Jgrism") : 0.3041,
    ("f32", "Hgrism") : 0.2964,
    ("f32", "Kgrism") : 0.4396,
}
