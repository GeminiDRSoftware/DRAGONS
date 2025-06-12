from collections import namedtuple

# Data structures used by this module

DetectorConfig = namedtuple("Config", "readnoise gain well linearlimit nonlinearlimit")
Config = namedtuple("Config", "mdf offsetsection pixscale mode")

detector_properties = {
    'UNKNOWN': { # GNIRS South
        # From an archived web page from 2006.  Found by Andy Stephens.
        # https://web.archive.org/web/20060901071906/http://www.gemini.edu/sciops/instruments/nirs/nirsIndex.html
        #
        # Dictionary key is the read mode and well depth setting
        # Dictionary values are in the following order:
        # readnoise, gain, well, linearlimit, nonlinearlimit
        # readnoise and well are in units of electrons
        ('Very Bright Objects', 'Shallow'): DetectorConfig(145., 13.5, 90000., 0.714286, 1.0),
        ('Bright Objects', 'Shallow'): DetectorConfig(38., 13.5, 90000., 0.714286, 1.0),
        ('Faint Objects', 'Shallow'): DetectorConfig(12., 13.5, 90000., 0.714286, 1.0),
        ('Very Faint Objects', 'Shallow'): DetectorConfig(7., 13.5, 90000., 0.714286, 1.0),
        ('Very Bright Objects', 'Deep'): DetectorConfig(145., 13.5, 180000., 0.714286, 1.0),
        ('Bright Objects', 'Deep'): DetectorConfig(38., 13.5, 180000., 0.714286, 1.0),
        ('Faint Objects', 'Deep'): DetectorConfig(12., 13.5, 180000., 0.714286, 1.0),
        ('Very Faint Objects', 'Deep'): DetectorConfig(7., 13.5, 180000., 0.714286, 1.0),

    },

    'SN7638228.1': { # GNIRS North - early days
        # Copied from SN7638228.1.2.  Values in Gemini IRAF's array.fits are
        # incomplete, but similar enough to assume that this is better than
        # using the incomplete set.  Can be adjusted if the GNIRS team provides
        # better values.
        #
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
    },
    'SN7638228.1.2': {   # GNIRS North - Orginal Detector Controller
        # Taken from https://www.gemini.edu/instrumentation/gnirs/components [June 2024]
        #
        # Dictionary key is the read mode and well depth setting
        # Dictionary values are in the following order:
        # readnoise, gain, well, linearlimit, nonlinearlimit
        # readnoise and well are in units of electrons
        ('Very Bright Objects', 'Shallow'): DetectorConfig(155., 13.5, 90000., 0.72, 1.0),
        ('Bright Objects', 'Shallow'): DetectorConfig(30., 13.5, 90000., 0.72, 1.0),
        ('Faint Objects', 'Shallow'): DetectorConfig(10., 13.5, 90000., 0.72, 1.0),
        ('Very Faint Objects', 'Shallow'): DetectorConfig(7., 13.5, 90000., 0.72, 1.0),
        ('Very Bright Objects', 'Deep'): DetectorConfig(155., 13.5, 180000., 0.72, 1.0),
        ('Bright Objects', 'Deep'): DetectorConfig(30., 13.5, 180000., 0.72, 1.0),
        ('Faint Objects', 'Deep'): DetectorConfig(10., 13.5, 180000., 0.72, 1.0),
        ('Very Faint Objects', 'Deep'): DetectorConfig(7., 13.5, 180000., 0.72, 1.0),
    },
    'SN7638228.1.2+ARC-III': {  # GNIRS North - New Detector Controller [July 2024]
        # Temporary values (copied from SN7638228.1.2)
        #
        # Dictionary key is the read mode and well depth setting
        # Dictionary values are in the following order:
        # readnoise, gain, well, linearlimit, nonlinearlimit
        # readnoise and well are in units of electrons
        ('Very Bright Objects', 'Shallow'): DetectorConfig(155., 13.5, 90000., 0.72, 1.0),
        ('Bright Objects', 'Shallow'): DetectorConfig(30., 13.5, 90000., 0.72, 1.0),
        ('Faint Objects', 'Shallow'): DetectorConfig(10., 13.5, 90000., 0.72, 1.0),
        ('Very Faint Objects', 'Shallow'): DetectorConfig(7., 13.5, 90000., 0.72, 1.0),
        ('Very Bright Objects', 'Deep'): DetectorConfig(155., 13.5, 180000., 0.72, 1.0),
        ('Bright Objects', 'Deep'): DetectorConfig(30., 13.5, 180000., 0.72, 1.0),
        ('Faint Objects', 'Deep'): DetectorConfig(10., 13.5, 180000., 0.72, 1.0),
        ('Very Faint Objects', 'Deep'): DetectorConfig(7., 13.5, 180000., 0.72, 1.0),
    },
}

nominal_zeropoints = {
    # There are none defined...
}

# pixel scales for GNIRS Short and Long cameras
pixel_scale = {
    'ShortBlue': 0.15170,  # +/- 0.00012 "/pix
    'ShortRed': 0.15,
    'LongBlue': 0.05071,  # +/- 0.0001 "/pix
    'LongRed': 0.05095,   # +/- 0.0002 "/pix
}

# JUST FOR THE config_dict.
pixel_scale_shrt = 0.15
pixel_scale_long = 0.05

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
    # Dispersion values are in nm/pix (updated by CJS 20250109)
    # The dispersion values are based on wvl. coverages for each filter/mode listed in GNIRS instrument pages.
    ("10/mm", "Short")  : {"M": -1.939},
    ("10/mm", "Long")   : {"X": -0.324,  "J": -0.389,   "H": -0.485, "K": -0.647, "L": -0.972, "M": -1.943},
    ("32/mm", "Short")  : {"X": -0.323,  "J": -0.388,   "H": -0.484, "K": -0.646, "L": -0.969, "M": -1.934},
    ("32/mm", "Long")   : {"X": -0.107,  "J": -0.129,   "H": -0.162, "K": -0.216, "L": -0.324, "M": -0.645},
    ("111/mm", "Short") : {"X": -0.092,  "J": -0.110,   "H": -0.139, "K": -0.185, "L": -0.273, "M": -0.562},
    ("111/mm", "Long")  : {"X": -0.0309, "J": -0.0371,  "H": -0.0464,"K": -0.0618,"L": -0.0922,"M": -0.1875}
}
