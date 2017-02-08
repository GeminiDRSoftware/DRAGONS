from collections import namedtuple

ArrayProperties = namedtuple("ArrayProperties", "readnoise gain welldepth "
                             "linlimit coeffs nonlinlimit")

filter_wavelengths = {
    'Z'        : 1.0150,
    'HeI1083'  : 1.0830,
    'PaG'      : 1.0940,
    'Jcont'    : 1.2070,
    'PaB'      : 1.2820,
    'Hcont'    : 1.5700,
    'CH4short' : 1.5800,
    'CH4long'  : 1.6900,
    'H2O'      : 2.0000,
    'HeI-2p2s' : 2.0580,
    'Kcntshrt' : 2.0930,
    'Kprime'   : 2.1200,
    'H2(1-0)'  : 2.1220,
    'Kshort'   : 2.1500,
    'H2(2-1)'  : 2.2480,
    'BrG'      : 2.1660,
    'Kcntlong' : 2.2700,
    'CO2360'   : 2.3600,
}

array_properties = {
    # Taken from http://www.gemini.edu/sciops/instruments/gsaoi/instrument-description/detector-characteristics
    # Dictionary key is the read mode and array section
    # Dictionary values are in the following order:
    # readnoise, gain, well, linearlimit, coeff1, coeff2, coeff3, nonlinearlimit
    # read noise is in units of electrons, well is in units of ADU

    # Well depth from email from Rodrigo Carrasco, 20150824:
    # Detector well depth. GSAOI detector is formed by 4 2kx2k arrays.
    # Each array have different saturation levels and well depth. The
    # lowest saturation level is for array 2. The following table give
    # to you values:
    #
    # Array      Saturation  Gain   Well Depth (e-)      Well Depth (e-)
    #              (ADU)           (no non-Lin. corr)   (after non lin. corr. - 2%)
    # --------------------------------------------------------------------------------------------
    # 1 (-074)     52400        2.434     100426                123826
    # 2 (-064)     50250        2.010     74266                 98060
    # 3 (-071)     53760        2.411     104528                127073
    # 4 (-061)     52300        2.644     110624                132962
    #
    # As you can see, the lowest well depth is in array. I suggest to use
    # this value (after non-linearity correction is applied (98000 e-) for
    # all arrays.
    #
    # You have to correct for non-linearity the detectors before you can
    # measure any thing, specially any photometry. When we reduce GSAOI
    # images, we apply the correction for non-linearity first. This is done
    # with the task "gaprepare" inside the GSAOI package. This task also
    # remove the first and last 4 pixels (columns and rows) from each array.
    # These pixels are not illuminated. Then we sky correct and flat field
    # the data. Finally, each array is multiply by the own gain to have all
    # 4 arrays at same level in electrons.
    #
    # What I suggest is to assume one value for the 4 arrays after gain
    # multiplication of 100000 electrons for the well depth (conservative,
    # but should be ok). Note that this well depth is based on Array 2 and
    # is at the 2% from 96% before saturation, after correction for
    # non-linearity.

    ('Bright Objects', 'H2RG-032-074'): ArrayProperties(26.53, 2.434, 52400, 0.73, (0.9947237, 7.400241E-7, 2.539787E-11), 0.97),
    ('Bright Objects', 'H2RG-032-064'): ArrayProperties(19.10, 2.010, 50250, 0.64, (0.9944842, 6.748257E-7, 3.679960E-11), 0.97),
    ('Bright Objects', 'H2RG-032-071'): ArrayProperties(27.24, 2.411, 53700, 0.76, (0.9947278, 7.067051E-7, 2.177223E-11), 0.98),
    ('Bright Objects', 'H2RG-032-061'): ArrayProperties(32.26, 2.644, 52300, 0.75, (0.9958842, 5.670857E-7, 2.718377E-11), 0.96),
    ('Faint Objects', 'H2RG-032-074'): ArrayProperties(13.63, 2.434, 52400, 0.73, (0.9947237, 7.400241E-7, 2.539787E-11), 0.97),
    ('Faint Objects', 'H2RG-032-064'): ArrayProperties(9.85, 2.010, 50250, 0.64, (0.9944842, 6.748257E-7, 3.679960E-11), 0.97),
    ('Faint Objects', 'H2RG-032-071'): ArrayProperties(14.22, 2.411, 53700, 0.76, (0.9947278, 7.067051E-7, 2.177223E-11), 0.98),
    ('Faint Objects', 'H2RG-032-061'): ArrayProperties(16.78, 2.644, 52300, 0.75, (0.9958842, 5.670857E-7, 2.718377E-11), 0.96),
    ('Very Faint Objects', 'H2RG-032-074'): ArrayProperties(10.22, 2.434, 52400, 0.73, (0.9947237, 7.400241E-7, 2.539787E-11), 0.97),
    ('Very Faint Objects', 'H2RG-032-064'): ArrayProperties(7.44, 2.010, 50250, 0.64, (0.9944842, 6.748257E-7, 3.679960E-11), 0.97),
    ('Very Faint Objects', 'H2RG-032-071'): ArrayProperties(10.61, 2.411, 53700, 0.76, (0.9947278, 7.067051E-7, 2.177223E-11), 0.98),
    ('Very Faint Objects', 'H2RG-032-061'): ArrayProperties(12.79, 2.644, 52300, 0.75, (0.9958842, 5.670857E-7, 2.718377E-11), 0.96),
    }

nominal_zeropoints = {
    # Table of GSAOI nominal zeropoint magnitude by camera and filter.
    # Updated 2015.08.24 with info from Rodrigo Carrasco
    # The zero points are:
    # J:   25.48+K(1-airmass) - used Flux in ADU - K=0.092
    # H:  25.77-J(1-airmass) - used Flux in ADU - K=0.031
    # Ks: 25.17-K(1-airmass) - used Flux in ADU - K=0.068
    # The extinction coefficients K are from CTIO.
    # For QAP, the airmass is assumed to be 1.

    # NOTE NOTE NOTE
    # The numbers are for 1 ADU/s not 1 electron/s, so we need to
    # add 2.5*log_10(gain) to each of them.
    # The gain is in the GSAOIArrayDict lookup table, should this
    # lookup directly reference the other rather than having the
    # same info twice?
    # Sector 1: gain = 2.434, 2.5*log_10(gain) = 0.966
    # Sector 2: gain = 2.010, 2.5*log_10(gain) = 0.758
    # Sector 3: gain = 2.411, 2.5*log_10(gain) = 0.955
    # Sector 4: gain = 2.644, 2.5*log_10(gain) = 1.056

    # Note: Z-band numbers come from S Leggett's paper,
    # http://iopscience.iop.org/article/10.1088/0004-637X/799/1/37/meta
    # "We derived identical zeropoints, within the uncertainties, for both
    # bright and faint read modes and for each of the four detectors. The
    # zeropoints were measured to be 26.71 at GSAOI-Z (equivalent to MKO-Y)
    # and 26.40 at J, where zeropoint is defined as the magnitude of an
    # object that produces one count (or data number) per second."


    # (FILTER, Array sector) : Nominal zeropoint for airmass=1
    #    ('Z', 'H2RG-032-074'): 27.68,   # 26.71 + 2.5*log10(2.434)
    #    ('Z', 'H2RG-032-064'): 27.47,   # 26.71 + 2.5*log10(2.010)
    #    ('Z', 'H2RG-032-071'): 27.67,   # 26.71 + 2.5*log10(2.411)
    #    ('Z', 'H2RG-032-061'): 27.77,   # 26.71 + 2.5*log10(2.644)
    #    ('J', 'H2RG-032-074'): 25.58,   # 24.61 + 2.5*log10(2.434)
    #    ('J', 'H2RG-032-064'): 20.07,   # 19.31 + 2.5*log10(2.010)
    #    ('J', 'H2RG-032-071'): 25.31,   # 24.35 + 2.5*log10(2.411)
    #    ('J', 'H2RG-032-061'): 27.96,   # 26.90 + 2.5*log10(2.644)
    #    ('H', 'H2RG-032-074'): 25.86,   # 24.89 + 2.5*log10(2.434)
    #    ('H', 'H2RG-032-064'): 20.29,   # 19.53 + 2.5*log10(2.010)
    #    ('H', 'H2RG-032-071'): 25.58,   # 24.62 + 2.5*log10(2.411)
    #    ('H', 'H2RG-032-061'): 28.26,   # 27.20 + 2.5*log10(2.644)
    #    ('Ks', 'H2RG-032-074'): 25.28,  # 24.31 + 2.5*log10(2.434)
    #    ('Ks', 'H2RG-032-064'): 19.84,  # 19.08 + 2.5*log10(2.010)
    #    ('Ks', 'H2RG-032-071'): 25.01,  # 24.05 + 2.5*log10(2.411)
    #    ('Ks', 'H2RG-032-061'): 27.63,  # 26.57 + 2.5*log10(2.644)

    # J, H, Ks values in electrons, provided from Rodrigo 2016.02.05.
    # Here are the zero points derived from data take 2013 Feb 19UT.
    # The errors of the zero points are quite large (in average 0.2
    # mag). Only two stars at different airmasses were observed that
    # night. You can use these zero points as starting point for the QAP.
    # I didn't fit the colour term because these stars are not suitable
    # to do that. Need a secondary start which we never observed. The
    # Extinction coefficient was fixed to the average values from 2MASS.
    # The difference between these values and the values given in the
    # GSAOI web page is the way the instrumental magnitude was calculated.
    # m_lambda  = - 2.5 log F_lambda where the flux is in electrons.
    #
    # Array 1:
    #ZP(J) = 26.857(0.188) - 0.092 * Xj
    #ZP(H) = 26.796(0.230) - 0.031 * Xh
    #ZP(Ks) = 26.201(0.251) - 0.065 * Xk
    #
    # Array 2:
    #ZP(J) = 26.891(0.160) - 0.092 * Xj
    #ZP(H) = 26.931(0.159) - 0.031 * Xh
    #ZP(Ks) = 26.287(0.198) - 0.065 * Xk
    #
    # Array 3:
    #ZP(J) = 26.774(0.211) - 0.092 * Xj
    #ZP(H) = 26.821(0.226) - 0.031 * Xh
    #ZP(Ks) = 26.154(0.218) - 0.065 * Xk
    #
    # Array 4:
    #ZP(J) = 26.727(0.235) - 0.092 * Xj
    #ZP(H) = 26.796(0.230) - 0.031 * Xh
    #ZP(Ks) = 26.192(0.268) - 0.065 * Xk
    #
    # The Z-band values are taken from S Leggett's paper
    # http://iopscience.iop.org/article/10.1088/0004-637X/799/1/37/meta
    # "We derived identical zeropoints, within the uncertainties, for both
    # bright and faint read modes and for each of the four detectors. The
    # zeropoints were measured to be 26.71 at GSAOI-Z (equivalent to MKO-Y)
    # and 26.40 at J, where zeropoint is defined as the magnitude of an
    # object that produces one count (or data number) per second."
    # By comparison of J-band with Rodrigo's values, we believe Z-band to
    # be in units of electrons as well.

    # The K' and K transforms from Ks are taken from the NIRI color
    # transforms at
    # http://www.gemini.edu/sciops/instruments/near-ir-resources/nir-photometric-standard-stars/niri-filter-color-transformation
    # Kshort = K + 0.002 + 0.026(J-K)
    # Kprime = K + 0.22(H-K)

    ('Z', 'H2RG-032-074'): 26.71,
    ('Z', 'H2RG-032-064'): 26.71,
    ('Z', 'H2RG-032-071'): 26.71,
    ('Z', 'H2RG-032-061'): 26.71,

    ('J', 'H2RG-032-074'): 26.857,
    ('J', 'H2RG-032-064'): 26.891,
    ('J', 'H2RG-032-071'): 26.774,
    ('J', 'H2RG-032-061'): 26.727,

    ('H', 'H2RG-032-074'): 26.796,
    ('H', 'H2RG-032-064'): 26.931,
    ('H', 'H2RG-032-071'): 26.821,
    ('H', 'H2RG-032-061'): 26.796,

    ('Kshort', 'H2RG-032-074'): 26.201,
    ('Kshort', 'H2RG-032-064'): 26.287,
    ('Kshort', 'H2RG-032-071'): 26.154,
    ('Kshort', 'H2RG-032-061'): 26.192,

    ('K', 'H2RG-032-074'): 26.199,
    ('K', 'H2RG-032-064'): 26.285,
    ('K', 'H2RG-032-071'): 26.152,
    ('K', 'H2RG-032-061'): 26.190,

    ('Kprime', 'H2RG-032-074'): 26.419,
    ('Kprime', 'H2RG-032-064'): 26.505,
    ('Kprime', 'H2RG-032-071'): 26.372,
    ('Kprime', 'H2RG-032-061'): 26.410,
}

read_modes = { 2 : 'Bright Objects',
               8 : 'Faint Objects',
              16 : 'Very Faint Objects' }
