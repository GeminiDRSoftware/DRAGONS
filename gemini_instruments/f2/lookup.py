from collections import namedtuple
ArrayProperties = namedtuple("ArrayProperties",
                             "readnoise gain welldepth linlimit nonlinlimit coeffs")

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
    # readnoise, gain, well, linearlimit, coeff1, coeff2, coeff3,
    # nonlinearlimit
    #    1:(7.215, 5.20, 250000, 0.80, 1.0, 0.0, 0.0, 0.80),
    # Linear limit: F2 is linear within 0.5% over 4-22 kADU,
    # 22 kADU = 97680e-
    # Non-linear limit: Saturation starts at ~35 kADU,
    # 35 kADU = 155400e-
    # readnoise and well are in units of electrons
    '1': ArrayProperties(11.7, 4.44, 155400, 0.6286, 1.0, (1.0, 0.0, 0.0)),
    '4': ArrayProperties(6.0, 4.44, 155400, 0.6286, 1.0, (1.0, 0.0, 0.0)),
    '6': ArrayProperties(5.3, 4.44, 155400, 0.6286, 1.0, (1.0, 0.0, 0.0)),
    '8': ArrayProperties(5.0, 4.44, 155400, 0.6286, 1.0, (1.0, 0.0, 0.0))
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
    ('J', 'f/16'): 25.19+1.62,
    ('H', 'f/16'): 25.41+1.62,
    ('Ks', 'f/16'): 24.59+1.62,

}
    # Dictionary keys are in the following order:
    # "grism, filter".
    # Dictionary values are in the following order:
    # (dispersion (A/pix), central_wavelength (A))
    # This is a concise version of gnirs$data/nsappwave.fits table
dispersion_and_wavelength = {
    ("JH, JH"):     (-6.667, 13900),
    ("HK, HK"):     (-7.826, 18710),
    #("HK, JH"):     (-4.000, 13900), # clearly a mistake in GIRAF LUT (at least dispersion)
    ("HK, JH"):     (-7.500, 18750),
    #("HK, Ks"): ?? only SV data
    #("HK, K-long"): ?? only CAL data
    #("HK, J"): ?? only SV data
    #("HK, H"): ?? only SV data
    ("R3K, Jlow"):  (-1.667, 11220),
    ("R3K, Y"):     (-1.642, 10200),
   # ("R3K, J"):     (-2.022, 12550), # as in IRAF and headers
    ("R3K, J"):     (-2.022, 13350), # as measured from the images
    ("R3K, H"):     (-2.609, 16310),
    #("R3K, Ks"):    (-3.462, 21570),
    ("R3K, Ks"):    (-3.462, 21950),
    ("R3K, K-long"):(-3.462, 22000)
}
