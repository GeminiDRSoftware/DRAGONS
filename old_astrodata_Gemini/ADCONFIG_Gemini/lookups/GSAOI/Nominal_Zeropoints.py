
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
