
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
    
    # (FILTER, Array sector) : Nominal zeropoint for airmass=1
    ('J', 1): 24.61,
    ('J', 2): 19.31,
    ('J', 3): 24.35,
    ('J', 4): 26.90,
    ('H', 1): 24.89,
    ('H', 2): 19.53,
    ('H', 3): 24.62,
    ('H', 4): 27.20,
    ('Ks', 1): 24.31,  
    ('Ks', 2): 19.08,  
    ('Ks', 3): 24.05,  
    ('Ks', 4): 26.57,  

}
