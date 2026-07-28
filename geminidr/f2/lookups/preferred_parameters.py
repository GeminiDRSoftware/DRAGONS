# F2 specific preferred parameters

preferred_parameters = {
    # Keyed on (disperser, filter_name)
    # Secondary key is the primitive name

    ("JH", "JH_G0809"): {
        'normalizeFlat': {'regions': "405:1755"},
        'fitTelluric':   {'order': 30},
    },
    ("JH", "JH_G0816"): {
        'normalizeFlat': {'regions': "400:1800"},
        'fitTelluric':   {'order': 29},
    },
    ("HK", "HK_G0806"): {
        'normalizeFlat': {'regions': "185:1845"},
        'fitTelluric':   {'order': 30},
    },
    ("HK", "HK_G0817"): {
        'normalizeFlat': {'regions': "270:1825"},
        'fitTelluric':   {'order': 22},
    },
    ("HK", "JH_G0809"): {
        'normalizeFlat': {'regions': "1150:2045"},
        'fitTelluric':   {'order': 10},
    },
    ("HK", "JH_G0816"): {
        'normalizeFlat': {'regions': "1150:2045"},
        'fitTelluric':   {'order': 10},
    },
    ("R3K", "J_G0802"): {
        'normalizeFlat': {'regions': "1090:1890"},
        'fitTelluric':   {'order': 30}
    },
    ("R3K", "H_G0803"): {
        'normalizeFlat': {'regions': "625:1655"},
        'fitTelluric':   {'order': 15},
    },
    ("R3K", "Ks_G0804"): {
        'normalizeFlat': {'regions': "715:1590"},
        'fitTelluric':   {'order': 20},
    },
    ("R3K", "K-long_G0812"): {
        'normalizeFlat': {'regions': "225:1815"},
        'fitTelluric':   {'order': 25},
    },

}