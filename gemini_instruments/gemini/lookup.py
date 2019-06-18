# Gives the effective wavelength in microns for the standard wavelength regimes
wavelength_band = {
    "None": 0.000,
    "u": 0.350,
    "g": 0.475,
    "r": 0.630,
    "i": 0.780,
    "Z": 0.900,
    "Y": 1.020,
    "X": 1.100,
    "J": 1.200,
    "H": 1.650,
    "K": 2.200,
    "L": 3.400,
    "M": 4.800,
    "N": 11.70,
    "Q": 18.30,
}

nominal_extinction = {
    # These are the nominal MK and CP extinction values
    # ie the k values where the magnitude of the star should be modified by 
    # -= k(airmass-1.0)
    #
    # Columns are given as:
    # (telescope, filter) : k
    #
    ('Gemini-North', 'u'): 0.42,
    ('Gemini-North', 'g'): 0.14,
    ('Gemini-North', 'r'): 0.11,
    ('Gemini-North', 'i'): 0.10,
    ('Gemini-North', 'z'): 0.05,
    #
    ('Gemini-South', 'u'): 0.38,
    ('Gemini-South', 'g'): 0.18,
    ('Gemini-South', 'r'): 0.10,
    ('Gemini-South', 'i'): 0.08,
    ('Gemini-South', 'z'): 0.05
}

# TODO: This should be moved to the instruments. Figure out a way...
# CJS says: I don't think it should. These are standard filter
# wavelengths that can live in astrodata. Specific instruments
# override them, and *those* can live in the instrument lookups.

# Instrument, filter effective wavlengths
# '*' can be overriden by a specific instrument
filter_wavelengths = {
    ('*', 'u'): 0.3500,
    ('*', 'g'): 0.4750,
    ('*', 'r'): 0.6300,
    ('*', 'i'): 0.7800,
    ('*', 'z'): 0.9250,
    ('*', 'Y'): 1.0200,
    ('*', 'J'): 1.2500,
    ('*', 'H'): 1.6350,
    ('*', 'K'): 2.2000,
    ('*', 'FeII'): 1.6440,
}
