gnirsArrayDict = {
    # Taken from http://www.gemini.edu/sciops/instruments/gnirs/imaging/detector-properties-and-read-modes
    # Dictionary key is the read mode and well depth setting
    # Dictionary values are in the following order:
    # readnoise, gain, well, linearlimit, nonlinearlimit
    # readnoise and well are in units of electrons
    ('Very Bright Objects', 'Shallow'): (155., 13.5, 90000., 0.714286, 1.0),
    ('Bright Objects', 'Shallow'): (30., 13.5, 90000., 0.714286, 1.0),
    ('Faint Objects', 'Shallow'): (10., 13.5, 90000., 0.714286, 1.0),
    ('Very Faint Objects', 'Shallow'): (7., 13.5, 90000., 0.714286, 1.0),
    ('Very Bright Objects', 'Deep'): (155., 13.5, 180000., 0.714286, 1.0),
    ('Bright Objects', 'Deep'): (30., 13.5, 180000., 0.714286, 1.0),
    ('Faint Objects', 'Deep'): (10., 13.5, 180000., 0.714286, 1.0),
    ('Very Faint Objects', 'Deep'): (7., 13.5, 180000., 0.714286, 1.0),
    }

#gnirsArrayDict = {
        # Taken from gnirs$data/array.fits
        # Dictionary key is the bias
        # Dictionary values are in the following order:
        # label, readnoise, gain, well, linearlimit, coeff1, coeff2, coeff3,
        # nonlinearlimit
#        "0.3000" : ( "1" , "126.0000" , "13.5000" , "7950" , "0.9000" , "1.017475" , "0.244937" , "1.019483" , "0.500000" ),
#        "0.6000" : ( "2" , "150.0000" , "13.5000" , "16400" , "0.9000" , "1.021168" , "0.134277" , "0.417923" , "0.600000" )
#    }
