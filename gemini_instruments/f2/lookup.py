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
    # ('J', 'f/16'): 25.19+1.62,
    # ('H', 'f/16'): 25.41+1.62,
    # ('Ks', 'f/16'): 24.59+1.62,

    # New values from Joan (Feb 2024) already in 1 e-/s
    ('J', 'f/16'): 26.51,  # +/- 0.08
    ('H', 'f/16'): 26.75,  # +/- 0.08
    ('Ks', 'f/16'): 26.12, # +/- 0.08

}
