f2ArrayDict = {
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
    1:(11.7, 4.44, 155400, 0.6286, 1.0, 0.0, 0.0, 1.0),
    4:(6.0, 4.44, 155400, 0.6286, 1.0, 0.0, 0.0, 1.0),
    8:(5.0, 4.44, 155400, 0.6286, 1.0, 0.0, 0.0, 1.0)    
    }
