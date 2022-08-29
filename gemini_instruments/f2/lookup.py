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

dispersion_offset_mask = {
    # Dictionary keys are in the following order:
    # "grism, filter".
    # Dictionary values are in the following order:
    # (dispersion (nm/pix), central_wavelength offset along the dispersion direction (pix),
    # illum_mask cut-on wvl (nm), illum_mask cut-off wvl (nm) )
    # All values were refined using the archive F2 ARC images.

    # The offset values for JH and HK filters are the same for old and new filters, since WAVELENG was not updated.
    # If WAVELENG gets eventually updated, offsets for the setups with new filters need to be updated too.

    ("JH", "JH_G0809"): DispersionOffsetMask(-0.651, -35, 888, 1774),   # old filter. (cut-on, cut-off) wvl at filter T=(1%, 50%) - to avoid order overlap
    ("JH", "JH_G0816"): DispersionOffsetMask(-0.651, -35, 857, 1782),   # new filter. If WAVELENG gets updated to 1.3385 um, new offset = 44 px
    ("HK", "HK_G0806"): DispersionOffsetMask(-0.757, 6, 1245, 2540),    # old filter. T=(1%, 1%)
    ("HK", "HK_G0817"): DispersionOffsetMask(-0.757, 6, 1273, 2534),    # new filter. If WAVELENG gets updated to 1.900 um, new offset = -32 px
    ("HK", "JH_G0809"): DispersionOffsetMask(-0.760, 644, 888, 2700),   # old filter. T=(1%, :) - to keep both orders
    ("HK", "JH_G0816"): DispersionOffsetMask(-0.760, 644, 857, 2700),   # new filter. If WAVELENG gets updated to 1.3385 um, new offset = ?
    ("R3K", "J-lo"):  DispersionOffsetMask(-0.168, -173, 1027, 1204),   # T=(1%, 1%)
    ("R3K", "J"):     DispersionOffsetMask(-0.202, 418, 1159, 1349),    # T=(1%, 1%)
    ("R3K", "H"):     DispersionOffsetMask(-0.260, 22, 1467, 1804),     # T=(1%, 1%)
    ("R3K", "Ks"):    DispersionOffsetMask(-0.349, 128, 1966, 2350),    # T=(1%, 1%)
    ("R3K", "K-long"):DispersionOffsetMask(-0.351, -14, 1865, 2520)     # T=(1%, X%) - cut-off value selected at inter-order min
    # Consider adding the following modes:
    #"HK, Ks": # SV data only
    #"HK, K-long": # CAL data only
    #"HK, J": # SV data only
    #"HK, H": # SV data only
    #"R3K, Y": # ENG data only
}
