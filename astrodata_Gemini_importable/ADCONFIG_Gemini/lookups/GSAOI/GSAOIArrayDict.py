GSAOIArrayDict = {
    # Taken from http://www.gemini.edu/sciops/instruments/gsaoi/instrument-description/detector-characteristics
    # Dictionary key is the read mode and array section
    # Dictionary values are in the following order:
    # readnoise, gain, well, linearlimit, coeff1, coeff2, coeff3, nonlinearlimit
    
    # Well depth from email from Rodrigo Carrasco, 20150824:
    # Detector well depth. GSAOI detector is formed by 4 2kx2k arrays. 
    # Each array have different saturation levels and well depth. The 
    # lowest saturation level is for array 2. The following table give 
    # to you values:
    #
    # Array      Saturation  Gain   Well Depth (e-)      Well Depth (e-)
    # (ADU)                       (no non-Lin. corr)   (after non lin. corr. - 2%)
    # --------------------------------------------------------------------------------------------
    # 1            52400        2.434     100426                123826
    # 2            50250        2.010     74266                 98060 
    # 3            53760        2.411     104528                127073
    # 4            52300        2.644     110624                132962
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


    ('Bright Objects', 1): (26.53, 2.434, 123826, 0.98, 0.9947, 7.4E-7, 2.5E-11, 1.0),
    ('Bright Objects', 2): (19.10, 2.010, 98060, 0.98, 0.994, 6.E-7, 3.7E-11, 1.0),
    ('Bright Objects', 3): (27.24, 2.411, 127073, 0.98, 0.9947, 7.1E-7, 2.2E-11, 1.0),
    ('Bright Objects', 4): (32.26, 2.644, 132962, 0.98, 0.9959, 5.7E-7, 2.7E-11, 1.0),
    ('Faint Objects', 1): (13.63, 2.434, 123826, 0.98, 0.9947, 7.4E-7, 2.5E-11, 1.0),
    ('Faint Objects', 2): (9.85, 2.010, 98060, 0.98, 0.994, 6.E-7, 3.7E-11, 1.0),
    ('Faint Objects', 3): (14.22, 2.411, 127073, 0.98, 0.9947, 7.1E-7, 2.2E-11, 1.0),
    ('Faint Objects', 4): (16.78, 2.644, 132962, 0.98, 0.9959, 5.7E-7, 2.7E-11, 1.0),
    ('Very Faint Objects', 1): (10.22, 2.434, 123826, 0.98, 0.9947, 7.4E-7, 2.5E-11, 1.0),
    ('Very Faint Objects', 2): (7.44, 2.010, 98060, 0.98, 0.994, 6.E-7, 3.7E-11, 1.0),
    ('Very Faint Objects', 3): (10.61, 2.411, 127073, 0.98, 0.9947, 7.1E-7, 2.2E-11, 1.0),
    ('Very Faint Objects', 4): (12.79, 2.644, 132962, 0.98, 0.9959, 5.7E-7, 2.7E-11, 1.0),
    }
