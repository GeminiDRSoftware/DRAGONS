nominal_zeropoints = {
    # Table of GMOS Nominal Zeropoint magnitudes
    # By filter and detector ID.
    # For new CCDs, just add them with the new detector ID
    # If we don't have per-chip values, just add the same value
    #   for each detector in the focal plane.
    #
    # The "composite" detector name is the DETID header and should here give the mean value
    # - This is used for mosaiced images that have the three detectors stiched together.
    #
    # At the moment, we don't account for the colour terms of the zeropoints
    # they are mostly < 0.1 mag,  especially longer than u, though it would be good to add
    # support for these in the future
    #
    # Nominal extinction values for CP and MK are given in a separate table 
    # at the Gemini level, and provided for by a descriptor
    #
    # Columns below are given as:
    # ("detector ID", "filter") : zeropoint
    #
    # GMOS-N original EEV detectors
    # Values from http://www.gemini.edu/sciops/instruments/gmos/calibration?q=node/10445  20111021
    ('EEV 9273-20-04', "u") : 25.47,
    ('EEV 9273-16-03', "u") : 25.47,
    ('EEV 9273-20-03', "u") : 25.47,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "u") : 25.47,
    ('EEV 9273-20-04', "g") : 27.95,
    ('EEV 9273-16-03', "g") : 27.95,
    ('EEV 9273-20-03', "g") : 27.95,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "g") : 27.95,
    ('EEV 9273-20-04', "r") : 28.20,
    ('EEV 9273-16-03', "r") : 28.20,
    ('EEV 9273-20-03', "r") : 28.20,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "r") : 28.20,
    ('EEV 9273-20-04', "i") : 27.94,
    ('EEV 9273-16-03', "i") : 27.94,
    ('EEV 9273-20-03', "i") : 27.94,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "i") : 27.94,
    ('EEV 9273-20-04', "z") : 26.78,
    ('EEV 9273-16-03', "z") : 26.78,
    ('EEV 9273-20-03', "z") : 26.78,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "z") : 26.78,
    #
    # GMOS-N New (Nov 2011) deep depletion E2V CCDs.
    # Bogus values made up by PH for u-band
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "u") : 10.00,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "u") : 10.00,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "u") : 10.00,
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "u") : 10.00,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "g") : 28.27,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "g") : 28.27,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "g") : 28.27,
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "g") : 28.27,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "r") : 28.37,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "r") : 28.37,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "r") : 28.37,
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "r") : 28.37,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "i") : 28.43,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "i") : 28.43,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "i") : 28.43,
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "i") : 28.43,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "z") : 27.73,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "z") : 27.73,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "z") : 27.73,
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "z") : 27.73,

    #
    # GMOS-S
    # Updated values using mean of 2014-01 values from website above.
    # CCD1 and CCD3 values not recorded - changed to have the same relative
    # value to CCD2 as previous values. Until individual chip measurements
    # are available there will be small systematic offsets due to the
    # measured stars not all being on CCD2. - JT (2014-01-20)
    ('EEV 2037-06-03', 'u'): 24.662,
    ('EEV 8194-19-04', 'u'): 24.639,
    ('EEV 8261-07-04', 'u'): 24.667,
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04', 'u'): 24.639,
    ('EEV 2037-06-03', 'g'): 28.194,
    ('EEV 8194-19-04', 'g'): 28.203,
    ('EEV 8261-07-04', 'g'): 28.234,
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04', 'g'): 28.203,
    ('EEV 2037-06-03', 'r'): 28.296,
    ('EEV 8194-19-04', 'r'): 28.286,
    ('EEV 8261-07-04', 'r'): 28.321,
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04', 'r'): 28.286,
    ('EEV 2037-06-03', 'i'): 27.844,
    ('EEV 8194-19-04', 'i'): 27.849,
    ('EEV 8261-07-04', 'i'): 27.883,
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04', 'i'): 27.849,
    ('EEV 2037-06-03', 'z'): 26.709,
    ('EEV 8194-19-04', 'z'): 26.718,
    ('EEV 8261-07-04', 'z'): 26.748,
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04', 'z'): 26.718,

    # GMOS-S Hamamatsu CCDs
    # Updating values supplied by Pascale Hibbon
    # 2014-08-27: Individual array values have not yet been determined using
    #             the mosaicked values
    # Error values were included in the original communication, they are the
    # numbers in-line that are commented out
    ('BI5-36-4k-2', 'u'): 24.551, # 1.074,
    ('BI11-33-4k-1', 'u'): 24.551, # 1.074,
    ('BI12-34-4k-1', 'u'): 24.551, # 1.074,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'u'): 24.551, # 1.074,
    ('BI5-36-4k-2', 'g'): 28.002, # 0.312,
    ('BI11-33-4k-1', 'g'): 28.002, # 0.312,
    ('BI12-34-4k-1', 'g'): 28.002, # 0.312,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'g'): 28.002, # 0.312,
    ('BI5-36-4k-2', 'r'): 28.089, # 0.323,
    ('BI11-33-4k-1', 'r'): 28.089, # 0.323,
    ('BI12-34-4k-1', 'r'): 28.089, # 0.323,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'r'): 28.089, # 0.323,
    ('BI5-36-4k-2', 'i'): 28.159, # 0.379,
    ('BI11-33-4k-1', 'i'): 28.159, # 0.379,
    ('BI12-34-4k-1', 'i'): 28.159, # 0.379,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'i'): 28.159, # 0.379,
    ('BI5-36-4k-2', 'z'): 27.786, # 0.432,
    ('BI11-33-4k-1', 'z'): 27.786, # 0.432,
    ('BI12-34-4k-1', 'z'): 27.786, # 0.432,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'z'): 27.786, # 0.432,
}
