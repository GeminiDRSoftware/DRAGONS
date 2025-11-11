from collections import namedtuple
ArrayProperties = namedtuple("ArrayProperties",
                             "readnoise gain welldepth linlimit nonlinlimit coeffs")
DispersionOffsetMask = namedtuple("DispersionOffsetMask", "dispersion cenwaveoffset cutonwvl cutoffwvl")


filter_wavelengths = {
    'Jlow'  : 1.1220,
    'JH'    : 1.3900,
    'H'     : 1.6310,
    'HK'    : 1.8710,
    'K-blue': 2.0600,
    'Ks'    : 2.1570,
    'K-red' : 2.3100,
    'Klong' : 2.2000,
}

array_properties = {
    # Taken from f2$data/f2array.fits
    # Dictionary key is the number of non-destructive read pairs
    # Dictionary values are in the following order:
    # readnoise, gain, well, linearlimit, nonlinearlimit,
    # coeff1, coeff2, coeff3,
    #    1:(7.215, 5.20, 250000, 0.80, 1.0, 0.0, 0.0, 0.80),
    # Linear limit: F2 is linear within 0.5% over 4-25 kADU,
    # 25 kADU = 97680e-
    # Non-linear limit: Saturation starts at ~35 kADU,
    # 35 kADU = 155400e-
    # readnoise and well are in units of electrons
    '1': ArrayProperties(11.7, 4.44, 155400, 0.72, 1.0, (0.9999107, -4.2854848e-8, 1.0096585e-12)),
    '4': ArrayProperties(6.0, 4.44, 155400, 0.72, 1.0, (0.9999107, -4.2854848e-8, 1.0096585e-12)),
    '6': ArrayProperties(5.3, 4.44, 155400, 0.72, 1.0, (0.9999107, -4.2854848e-8, 1.0096585e-12)),
    '8': ArrayProperties(5.0, 4.44, 155400, 0.72, 1.0, (0.9999107, -4.2854848e-8, 1.0096585e-12))
    }

nominal_zeropoints = {
    # Table of F2 nominal zeropoint magnitudes by filter.
    # Updated 07.27.2015

    # From Ruben Diaz:
    # The zero points for the photometric MKO system were obtained using 88
    # independent data points from standard stars observations made during
    # the three commissioning runs of May through July 2013. The software
    # package THELI (Schirmer 2013) was used.
    # Y = 25.12 - 0.01*k + 0.50*(Y-J) +/- 0.03
    # J = 25.21 - 0.02*k + 0.87*(J-H) +/- 0.05
    # H = 25.42 - 0.01*k + 0.73*(J-H) +/- 0.05
    # Ks= 24.64 - 0.05*k - 0.27*(H-Ks) +/- 0.06
    # (where k is the extinction coefficient.)
    # Updated values from 2014 standards will be available soon.
    # Note: Mischa confirmed k is the airmass
    # Values used assume airmass of 1, and (Y-J), (J-H), (J-H), (H-Ks) = 0
    # EJD
    # Leaving a camera field in the lookup key, in the future when the use of
    # F2 with MCAO (GeMS) is commissioned, the nominal zero points from the
    # f/16 mode and the f/33 mode are expected to differ.

    # NOTE NOTE NOTE
    # The numbers above are for 1 ADU/s not 1 electron/s, so we need to
    # add 2.5*log_10(gain) = 2.5*log_10(4.44) = 1.62 to each of them.

    # (BAND, CAMERA): Nominal zeropoint for airmass=1
    ('Y', 'f/16'): 25.11+1.62,
    # ('J', 'f/16'): 25.19+1.62,
    # ('H', 'f/16'): 25.41+1.62,
    # ('Ks', 'f/16'): 24.59+1.62,

    # New values from Joan (Feb 2024) already in 1 e-/s
    ('J', 'f/16'): 26.51,  # +/- 0.08
    ('H', 'f/16'): 26.75,  # +/- 0.08
    ('Ks', 'f/16'): 26.12, # +/- 0.08

}
 # Instruction for making this LUT:
 # https://docs.google.com/document/d/1LVTUFWkXJygkRUvqjsFm_7VZnqy7I4fy/edit?usp=sharing&ouid=106387637049533476653&rtpof=true&sd=true
dispersion_offset_mask = {
    # Dictionary keys are in the following order:
    # "grism, filter".
    # Dictionary values are in the following order:
    # (dispersion (nm/pix), central_wavelength offset along the dispersion direction (pix),
    # illum_mask cut-on wvl (nm), illum_mask cut-off wvl (nm) )
    # All values were refined using the archive F2 ARC images.

    ("JH", "JH_G0809"):     DispersionOffsetMask(-0.651, -35, 888, 1774),   # old filter. (cut-on, cut-off) wvl at filter T=(1%, 50%) - to avoid order overlap
    ("JH", "JH_G0816"):     DispersionOffsetMask(-0.651, 44, 857, 1782),    # new filter. T=(1%, 50%)
    ("HK", "HK_G0806"):     DispersionOffsetMask(-0.757, 6, 1245, 2540),    # old filter. T=(1%, 1%)
    ("HK", "HK_G0817"):     DispersionOffsetMask(-0.757, -30, 1273, 2534),  # new filter. T=(1%, 1%)
    ("HK", "JH_G0809"):     DispersionOffsetMask(-0.760, 635, 888, 2700),   # old filter. T=(1%, :) - to keep both orders as per IS request
    ("HK", "JH_G0816"):     DispersionOffsetMask(-0.760, 710, 857, 2700),   # new filter. T=(1%, :)
    ("R3K", "J-lo_G0801"):  DispersionOffsetMask(-0.168, -173, 1027, 1204), # T=(1%, 1%)
    ("R3K", "J_G0802"):     DispersionOffsetMask(-0.202, 418, 1159, 1349),  # T=(1%, 1%)
    ("R3K", "H_G0803"):     DispersionOffsetMask(-0.260, 22, 1467, 1804),   # T=(1%, 1%)
    ("R3K", "Ks_G0804"):    DispersionOffsetMask(-0.349, 128, 1966, 2350),  # T=(1%, 1%)
    ("R3K", "K-long_G0812"):DispersionOffsetMask(-0.351, -14, 1865, 2520)   # T=(1%, 1%)
    # Consider adding the following modes:
    #"HK, Ks": # SV data only
    #"HK, K-long": # CAL data only
    #"HK, J": # SV data only
    #"HK, H": # SV data only
    #"R3K, Y": # ENG data only
}

resolving_power = {
    # Average (within 70% of wvl ranges) F2 resolutions for various grism/slit combinations.
    # The values from the F2 instrument web pages.
    # Note that average resolutions are significantly lower than peak resolutions.
    # Dictionary keys:
    # "slit width in pixels"
    # Dictionary values:
    # "Grism": average resolution
    "1": {"JH": 1300, "HK": 1300, "R3K": 3600},
    "2": {"JH": 900, "HK": 900, "R3K": 2800},
    "3": {"JH": 600, "HK": 600, "R3K": 1600},
    "4": {"JH": 350, "HK": 350, "R3K": 1300},
    "6": {"JH": 130, "HK": 130, "R3K": 1000},
    "8": {"JH": 100, "HK": 100, "R3K": 750}
}
