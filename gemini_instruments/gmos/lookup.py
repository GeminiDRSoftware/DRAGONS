filter_wavelengths = {
    'HeII'  : 0.4680,
    'HeIIC' : 0.4780,
    'OIII'  : 0.4990,
    'OIIIC' : 0.5140,
    'Ha'    : 0.6560,
    'HaC'   : 0.6620,
    'SII'   : 0.6720,
    'OVIC'  : 0.6790,
    'OVI'   : 0.6838,
    'ri'    : 0.7054,  # custom user filter
    'CaT'   : 0.8600,
    'Z'     : 0.8760,
    'DS920' : 0.9200,
    'Y'     : 1.0100,
}

# Database of GMOS CCD amplifier GAIN and READNOISE properties
gmosampsGain = {
    # Database of GMOS CCD amplifier GAIN properties
    # after 2017-02-24
    # Columns below are given as:
    # READOUT, GAINSTATE, AMPNAME : GAIN
    # GMOS-N: Hamamatsu CCDs
    # Latest values from K.Chiboucas on 2017-05-05. Uncertainty on gain: 0.04
    # AMPS 1-4 (CCD1)
    ("slow", "low", "BI13-20-4k-1, 1"): 1.568,
    ("slow", "high", "BI13-20-4k-1, 1"): 1.000,
    ("fast", "high", "BI13-20-4k-1, 1"): 5.04,
    ("fast", "low", "BI13-20-4k-1, 1"): 1.91,
    ("slow", "low", "BI13-20-4k-1, 2"): 1.620,
    ("slow", "high", "BI13-20-4k-1, 2"): 1.000,
    ("fast", "high", "BI13-20-4k-1, 2"): 5.05,
    ("fast", "low", "BI13-20-4k-1, 2"): 1.97,
    ("slow", "low", "BI13-20-4k-1, 3"): 1.618,
    ("slow", "high", "BI13-20-4k-1, 3"): 1.000,
    ("fast", "high", "BI13-20-4k-1, 3"): 5.18,
    ("fast", "low", "BI13-20-4k-1, 3"): 1.95,
    ("slow", "low", "BI13-20-4k-1, 4"): 1.675,
    ("slow", "high", "BI13-20-4k-1, 4"): 1.000,
    ("fast", "high", "BI13-20-4k-1, 4"): 5.21,
    ("fast", "low", "BI13-20-4k-1, 4"): 2.01,
    # AMPS 5-8 (CCD2)
    ("slow", "low", "BI12-09-4k-2, 1"): 1.664,
    ("slow", "high", "BI12-09-4k-2, 1"): 1.000,
    ("fast", "high", "BI12-09-4k-2, 1"): 5.29,
    ("fast", "low", "BI12-09-4k-2, 1"): 2.01,
    ("slow", "low", "BI12-09-4k-2, 2"): 1.633,
    ("slow", "high", "BI12-09-4k-2, 2"): 1.000,
    ("fast", "high", "BI12-09-4k-2, 2"): 5.14,
    ("fast", "low", "BI12-09-4k-2, 2"): 1.96,
    ("slow", "low", "BI12-09-4k-2, 3"): 1.65,
    ("slow", "high", "BI12-09-4k-2, 3"): 1.000,
    ("fast", "high", "BI12-09-4k-2, 3"): 5.13,
    ("fast", "low", "BI12-09-4k-2, 3"): 1.97,
    ("slow", "low", "BI12-09-4k-2, 4"): 1.69,
    ("slow", "high", "BI12-09-4k-2, 4"): 1.000,
    ("fast", "high", "BI12-09-4k-2, 4"): 5.29,
    ("fast", "low", "BI12-09-4k-2, 4"): 2.03,
    # AMPS 9-12 (CCD3)
    ("slow", "low", "BI13-18-4k-2, 1"): 1.654,
    ("slow", "high", "BI13-18-4k-2, 1"): 1.000,
    ("fast", "high", "BI13-18-4k-2, 1"): 5.09,
    ("fast", "low", "BI13-18-4k-2, 1"): 1.95,
    ("slow", "low", "BI13-18-4k-2, 2"): 1.587,
    ("slow", "high", "BI13-18-4k-2, 2"): 1.000,
    ("fast", "high", "BI13-18-4k-2, 2"): 4.86,
    ("fast", "low", "BI13-18-4k-2, 2"): 1.89,
    ("slow", "low", "BI13-18-4k-2, 3"): 1.63,
    ("slow", "high", "BI13-18-4k-2, 3"): 1.000,
    ("fast", "high", "BI13-18-4k-2, 3"): 5.10,
    ("fast", "low", "BI13-18-4k-2, 3"): 1.95,
    ("slow", "low", "BI13-18-4k-2, 4"): 1.604,
    ("slow", "high", "BI13-18-4k-2, 4"): 1.000,
    ("fast", "high", "BI13-18-4k-2, 4"): 5.03,
    ("fast", "low", "BI13-18-4k-2, 4"): 1.95,

    # GMOS-S New Hamamatsu CCDs + swap (2023)
    # Values from German Gimeno on Jan 22, 2024
    # Valid from 20231214

    # AMPS 1-4 (CCD1)
    ("slow", "low",  "BI11-41-4k-2, 1"): 2.038,
    ("slow", "high", "BI11-41-4k-2, 1"): 1.000,
    ("fast", "high", "BI11-41-4k-2, 1"): 5.758,
    ("fast", "low",  "BI11-41-4k-2, 1"): 1.719,
    ("slow", "low",  "BI11-41-4k-2, 2"): 2.016,
    ("slow", "high", "BI11-41-4k-2, 2"): 1.000,
    ("fast", "high", "BI11-41-4k-2, 2"): 5.708,
    ("fast", "low",  "BI11-41-4k-2, 2"): 1.711,
    ("slow", "low",  "BI11-41-4k-2, 3"): 2.020,
    ("slow", "high", "BI11-41-4k-2, 3"): 1.000,
    ("fast", "high", "BI11-41-4k-2, 3"): 5.719,
    ("fast", "low",  "BI11-41-4k-2, 3"): 1.715,
    ("slow", "low",  "BI11-41-4k-2, 4"): 1.943,
    ("slow", "high", "BI11-41-4k-2, 4"): 1.000,
    ("fast", "high", "BI11-41-4k-2, 4"): 5.518,
    ("fast", "low",  "BI11-41-4k-2, 4"): 1.652,
    # AMPS 5-8 (CCD2)
    ("slow", "low",  "BI13-19-4k-3, 1"): 1.798,
    ("slow", "high", "BI13-19-4k-3, 1"): 1.000,
    ("fast", "high", "BI13-19-4k-3, 1"): 5.091,
    ("fast", "low",  "BI13-19-4k-3, 1"): 1.533,
    ("slow", "low",  "BI13-19-4k-3, 2"): 1.745,
    ("slow", "high", "BI13-19-4k-3, 2"): 1.000,
    ("fast", "high", "BI13-19-4k-3, 2"): 4.956,
    ("fast", "low",  "BI13-19-4k-3, 2"): 1.498,
    ("slow", "low",  "BI13-19-4k-3, 3"): 1.787,
    ("slow", "high", "BI13-19-4k-3, 3"): 1.000,
    ("fast", "high", "BI13-19-4k-3, 3"): 5.082,
    ("fast", "low",  "BI13-19-4k-3, 3"): 1.525,
    ("slow", "low",  "BI13-19-4k-3, 4"): 1.787,
    ("slow", "high", "BI13-19-4k-3, 4"): 1.000,
    ("fast", "high", "BI13-19-4k-3, 4"): 5.100,
    ("fast", "low",  "BI13-19-4k-3, 4"): 1.531,
    # AMPS 9-12 (CCD3)
    ("slow", "low",  "BI12-34-4k-1, 1"): 1.594,
    ("slow", "high", "BI12-34-4k-1, 1"): 1.000,
    ("fast", "high", "BI12-34-4k-1, 1"): 4.581,
    ("fast", "low",  "BI12-34-4k-1, 1"): 1.445,
    ("slow", "low",  "BI12-34-4k-1, 2"): 1.699,
    ("slow", "high", "BI12-34-4k-1, 2"): 1.000,
    ("fast", "high", "BI12-34-4k-1, 2"): 4.893,
    ("fast", "low",  "BI12-34-4k-1, 2"): 1.545,
    ("slow", "low",  "BI12-34-4k-1, 3"): 1.681,
    ("slow", "high", "BI12-34-4k-1, 3"): 1.000,
    ("fast", "high", "BI12-34-4k-1, 3"): 4.825,
    ("fast", "low",  "BI12-34-4k-1, 3"): 1.478,
    ("slow", "low",  "BI12-34-4k-1, 4"): 1.770,
    ("slow", "high", "BI12-34-4k-1, 4"): 1.000,
    ("fast", "high", "BI12-34-4k-1, 4"): 5.082,
    ("fast", "low",  "BI12-34-4k-1, 4"): 1.549
}

gmosampsGainBefore20231214 = {
    # Database of GMOS CCD amplifier GAIN properties
    # after 2017-02-24
    # Columns below are given as:
    # READOUT, GAINSTATE, AMPNAME : GAIN
    # GMOS-N: Hamamatsu CCDs
    # Latest values from K.Chiboucas on 2017-05-05. Uncertainty on gain: 0.04
    # AMPS 1-4 (CCD1)
    ("slow", "low", "BI13-20-4k-1, 1"): 1.568,
    ("slow", "high", "BI13-20-4k-1, 1"): 1.000,
    ("fast", "high", "BI13-20-4k-1, 1"): 5.04,
    ("fast", "low", "BI13-20-4k-1, 1"): 1.91,
    ("slow", "low", "BI13-20-4k-1, 2"): 1.620,
    ("slow", "high", "BI13-20-4k-1, 2"): 1.000,
    ("fast", "high", "BI13-20-4k-1, 2"): 5.05,
    ("fast", "low", "BI13-20-4k-1, 2"): 1.97,
    ("slow", "low", "BI13-20-4k-1, 3"): 1.618,
    ("slow", "high", "BI13-20-4k-1, 3"): 1.000,
    ("fast", "high", "BI13-20-4k-1, 3"): 5.18,
    ("fast", "low", "BI13-20-4k-1, 3"): 1.95,
    ("slow", "low", "BI13-20-4k-1, 4"): 1.675,
    ("slow", "high", "BI13-20-4k-1, 4"): 1.000,
    ("fast", "high", "BI13-20-4k-1, 4"): 5.21,
    ("fast", "low", "BI13-20-4k-1, 4"): 2.01,
    # AMPS 5-8 (CCD2)
    ("slow", "low", "BI12-09-4k-2, 1"): 1.664,
    ("slow", "high", "BI12-09-4k-2, 1"): 1.000,
    ("fast", "high", "BI12-09-4k-2, 1"): 5.29,
    ("fast", "low", "BI12-09-4k-2, 1"): 2.01,
    ("slow", "low", "BI12-09-4k-2, 2"): 1.633,
    ("slow", "high", "BI12-09-4k-2, 2"): 1.000,
    ("fast", "high", "BI12-09-4k-2, 2"): 5.14,
    ("fast", "low", "BI12-09-4k-2, 2"): 1.96,
    ("slow", "low", "BI12-09-4k-2, 3"): 1.65,
    ("slow", "high", "BI12-09-4k-2, 3"): 1.000,
    ("fast", "high", "BI12-09-4k-2, 3"): 5.13,
    ("fast", "low", "BI12-09-4k-2, 3"): 1.97,
    ("slow", "low", "BI12-09-4k-2, 4"): 1.69,
    ("slow", "high", "BI12-09-4k-2, 4"): 1.000,
    ("fast", "high", "BI12-09-4k-2, 4"): 5.29,
    ("fast", "low", "BI12-09-4k-2, 4"): 2.03,
    # AMPS 9-12 (CCD3)
    ("slow", "low", "BI13-18-4k-2, 1"): 1.654,
    ("slow", "high", "BI13-18-4k-2, 1"): 1.000,
    ("fast", "high", "BI13-18-4k-2, 1"): 5.09,
    ("fast", "low", "BI13-18-4k-2, 1"): 1.95,
    ("slow", "low", "BI13-18-4k-2, 2"): 1.587,
    ("slow", "high", "BI13-18-4k-2, 2"): 1.000,
    ("fast", "high", "BI13-18-4k-2, 2"): 4.86,
    ("fast", "low", "BI13-18-4k-2, 2"): 1.89,
    ("slow", "low", "BI13-18-4k-2, 3"): 1.63,
    ("slow", "high", "BI13-18-4k-2, 3"): 1.000,
    ("fast", "high", "BI13-18-4k-2, 3"): 5.10,
    ("fast", "low", "BI13-18-4k-2, 3"): 1.95,
    ("slow", "low", "BI13-18-4k-2, 4"): 1.604,
    ("slow", "high", "BI13-18-4k-2, 4"): 1.000,
    ("fast", "high", "BI13-18-4k-2, 4"): 5.03,
    ("fast", "low", "BI13-18-4k-2, 4"): 1.95,
    # GMOS-S Hamamatsu CCDs (new E video boards, 2015)
    # AMPS 1-4 (CCD1)
    ("slow", "low",  "BI5-36-4k-2, 1"): 1.852,
    ("slow", "high", "BI5-36-4k-2, 1"): 1.000,
    ("fast", "high", "BI5-36-4k-2, 1"): 5.240,
    ("fast", "low",  "BI5-36-4k-2, 1"): 1.566,
    ("slow", "low",  "BI5-36-4k-2, 2"): 1.878,
    ("slow", "high", "BI5-36-4k-2, 2"): 1.000,
    ("fast", "high", "BI5-36-4k-2, 2"): 5.330,
    ("fast", "low",  "BI5-36-4k-2, 2"): 1.592,
    ("slow", "low",  "BI5-36-4k-2, 3"): 1.874,
    ("slow", "high", "BI5-36-4k-2, 3"): 1.000,
    ("fast", "high", "BI5-36-4k-2, 3"): 5.300,
    ("fast", "low",  "BI5-36-4k-2, 3"): 1.593,
    ("slow", "low",  "BI5-36-4k-2, 4"): 1.834,
    ("slow", "high", "BI5-36-4k-2, 4"): 1.000,
    ("fast", "high", "BI5-36-4k-2, 4"): 5.183,
    ("fast", "low",  "BI5-36-4k-2, 4"): 1.556,
    # AMPS 5-8 (CCD2)
    ("slow", "low",  "BI11-33-4k-1, 1"): 1.878,
    ("slow", "high", "BI11-33-4k-1, 1"): 1.000,
    ("fast", "high", "BI11-33-4k-1, 1"): 5.334,
    ("fast", "low",  "BI11-33-4k-1, 1"): 1.613,
    ("slow", "low",  "BI11-33-4k-1, 2"): 1.840,
    ("slow", "high", "BI11-33-4k-1, 2"): 1.000,
    ("fast", "high", "BI11-33-4k-1, 2"): 5.225,
    ("fast", "low",  "BI11-33-4k-1, 2"): 1.582,
    ("slow", "low",  "BI11-33-4k-1, 3"): 1.933,
    ("slow", "high", "BI11-33-4k-1, 3"): 1.000,
    ("fast", "high", "BI11-33-4k-1, 3"): 5.459,
    ("fast", "low",  "BI11-33-4k-1, 3"): 1.648,
    ("slow", "low",  "BI11-33-4k-1, 4"): 1.908,
    ("slow", "high", "BI11-33-4k-1, 4"): 1.000,
    ("fast", "high", "BI11-33-4k-1, 4"): 5.407,
    ("fast", "low",  "BI11-33-4k-1, 4"): 1.634,
    # AMPS 9-12 (CCD3)
    ("slow", "low",  "BI12-34-4k-1, 1"): 1.652,
    ("slow", "high", "BI12-34-4k-1, 1"): 1.000,
    ("fast", "high", "BI12-34-4k-1, 1"): 4.828,
    ("fast", "low",  "BI12-34-4k-1, 1"): 1.445,
    ("slow", "low",  "BI12-34-4k-1, 2"): 1.761,
    ("slow", "high", "BI12-34-4k-1, 2"): 1.000,
    ("fast", "high", "BI12-34-4k-1, 2"): 5.129,
    ("fast", "low",  "BI12-34-4k-1, 2"): 1.545,
    ("slow", "low",  "BI12-34-4k-1, 3"): 1.724,
    ("slow", "high", "BI12-34-4k-1, 3"): 1.000,
    ("fast", "high", "BI12-34-4k-1, 3"): 4.894,
    ("fast", "low",  "BI12-34-4k-1, 3"): 1.478,
    ("slow", "low",  "BI12-34-4k-1, 4"): 1.813,
    ("slow", "high", "BI12-34-4k-1, 4"): 1.000,
    ("fast", "high", "BI12-34-4k-1, 4"): 5.120,
    ("fast", "low",  "BI12-34-4k-1, 4"): 1.549
}

gmosampsGainBefore20170224 = {
        # Database of GMOS CCD amplifier GAIN properties
        # after 2015-08-26 and before 2017-02-24
        # Columns below are given as:
        # READOUT, GAINSTATE, AMPNAME : GAIN
        # GMOS-N:
        # e2vDD CCDs best amps
        ("slow", "low",  "e2v 10031-23-05, right") : 2.310,
        ("slow", "high", "e2v 10031-23-05, right") : 4.520,
        ("fast", "high", "e2v 10031-23-05, right") : 5.340,
        ("fast", "low",  "e2v 10031-23-05, right") : 2.530,
        ("slow", "low",  "e2v 10031-01-03, right") : 2.270,
        ("slow", "high", "e2v 10031-01-03, right") : 4.390,
        ("fast", "high", "e2v 10031-01-03, right") : 5.270,
        ("fast", "low",  "e2v 10031-01-03, right") : 2.500,
        ("slow", "low",  "e2v 10031-18-04, left")  : 2.170,
        ("slow", "high", "e2v 10031-18-04, left")  : 4.820,
        ("fast", "high", "e2v 10031-18-04, left")  : 5.160,
        ("fast", "low",  "e2v 10031-18-04, left")  : 2.430,
        # e2vDD CCDs secondary amps
        ("slow", "low",  "e2v 10031-23-05, left")  : 2.310,
        ("slow", "high", "e2v 10031-23-05, left")  : 4.590,
        ("fast", "high", "e2v 10031-23-05, left")  : 5.360,
        ("fast", "low",  "e2v 10031-23-05, left")  : 2.530,
        ("slow", "low",  "e2v 10031-01-03, left")  : 2.210,
        ("slow", "high", "e2v 10031-01-03, left")  : 4.590,
        ("fast", "high", "e2v 10031-01-03, left")  : 5.100,
        ("fast", "low",  "e2v 10031-01-03, left")  : 2.410,
        ("slow", "low",  "e2v 10031-18-04, right") : 2.330,
        ("slow", "high", "e2v 10031-18-04, right") : 4.330,
        ("fast", "high", "e2v 10031-18-04, right") : 5.410,
        ("fast", "low",  "e2v 10031-18-04, right") : 2.560,
        # GMOS-S:
        # GMOS-S Hamamatsu CCDs (new E video boards, 2015)
        # AMPS 1-4 (CCD1)
        ("slow", "low",  "BI5-36-4k-2, 1"): 1.852,
        ("slow", "high", "BI5-36-4k-2, 1"): 1.000,
        ("fast", "high", "BI5-36-4k-2, 1"): 5.240,
        ("fast", "low",  "BI5-36-4k-2, 1"): 1.566,
        ("slow", "low",  "BI5-36-4k-2, 2"): 1.878,
        ("slow", "high", "BI5-36-4k-2, 2"): 1.000,
        ("fast", "high", "BI5-36-4k-2, 2"): 5.330,
        ("fast", "low",  "BI5-36-4k-2, 2"): 1.592,
        ("slow", "low",  "BI5-36-4k-2, 3"): 1.874,
        ("slow", "high", "BI5-36-4k-2, 3"): 1.000,
        ("fast", "high", "BI5-36-4k-2, 3"): 5.300,
        ("fast", "low",  "BI5-36-4k-2, 3"): 1.593,
        ("slow", "low",  "BI5-36-4k-2, 4"): 1.834,
        ("slow", "high", "BI5-36-4k-2, 4"): 1.000,
        ("fast", "high", "BI5-36-4k-2, 4"): 5.183,
        ("fast", "low",  "BI5-36-4k-2, 4"): 1.556,
        # AMPS 5-8 (CCD2)
        ("slow", "low",  "BI11-33-4k-1, 1"): 1.878,
        ("slow", "high", "BI11-33-4k-1, 1"): 1.000,
        ("fast", "high", "BI11-33-4k-1, 1"): 5.334,
        ("fast", "low",  "BI11-33-4k-1, 1"): 1.613,
        ("slow", "low",  "BI11-33-4k-1, 2"): 1.840,
        ("slow", "high", "BI11-33-4k-1, 2"): 1.000,
        ("fast", "high", "BI11-33-4k-1, 2"): 5.225,
        ("fast", "low",  "BI11-33-4k-1, 2"): 1.582,
        ("slow", "low",  "BI11-33-4k-1, 3"): 1.933,
        ("slow", "high", "BI11-33-4k-1, 3"): 1.000,
        ("fast", "high", "BI11-33-4k-1, 3"): 5.459,
        ("fast", "low",  "BI11-33-4k-1, 3"): 1.648,
        ("slow", "low",  "BI11-33-4k-1, 4"): 1.908,
        ("slow", "high", "BI11-33-4k-1, 4"): 1.000,
        ("fast", "high", "BI11-33-4k-1, 4"): 5.407,
        ("fast", "low",  "BI11-33-4k-1, 4"): 1.634,
        # AMPS 9-12 (CCD3)
        ("slow", "low",  "BI12-34-4k-1, 1"): 1.652,
        ("slow", "high", "BI12-34-4k-1, 1"): 1.000,
        ("fast", "high", "BI12-34-4k-1, 1"): 4.828,
        ("fast", "low",  "BI12-34-4k-1, 1"): 1.445,
        ("slow", "low",  "BI12-34-4k-1, 2"): 1.761,
        ("slow", "high", "BI12-34-4k-1, 2"): 1.000,
        ("fast", "high", "BI12-34-4k-1, 2"): 5.129,
        ("fast", "low",  "BI12-34-4k-1, 2"): 1.545,
        ("slow", "low",  "BI12-34-4k-1, 3"): 1.724,
        ("slow", "high", "BI12-34-4k-1, 3"): 1.000,
        ("fast", "high", "BI12-34-4k-1, 3"): 4.894,
        ("fast", "low",  "BI12-34-4k-1, 3"): 1.478,
        ("slow", "low",  "BI12-34-4k-1, 4"): 1.813,
        ("slow", "high", "BI12-34-4k-1, 4"): 1.000,
        ("fast", "high", "BI12-34-4k-1, 4"): 5.120,
        ("fast", "low",  "BI12-34-4k-1, 4"): 1.549
    }


gmosampsGainBefore20150826 = {
        # Database of GMOS CCD amplifier GAIN properties
        # after 2006-08-31
        # Columns below are given as:
        # READOUT, GAINSTATE, AMPNAME : GAIN
        # GMOS-N:
        # e2vDD CCDs best amps
        ("slow", "low",  "e2v 10031-23-05, right") : 2.310,
        ("slow", "high", "e2v 10031-23-05, right") : 4.520,
        ("fast", "high", "e2v 10031-23-05, right") : 5.340,
        ("fast", "low",  "e2v 10031-23-05, right") : 2.530,
        ("slow", "low",  "e2v 10031-01-03, right") : 2.270,
        ("slow", "high", "e2v 10031-01-03, right") : 4.390,
        ("fast", "high", "e2v 10031-01-03, right") : 5.270,
        ("fast", "low",  "e2v 10031-01-03, right") : 2.500,
        ("slow", "low",  "e2v 10031-18-04, left")  : 2.170,
        ("slow", "high", "e2v 10031-18-04, left")  : 4.820,
        ("fast", "high", "e2v 10031-18-04, left")  : 5.160,
        ("fast", "low",  "e2v 10031-18-04, left")  : 2.430,
        # e2vDD CCDs secondary amps
        ("slow", "low",  "e2v 10031-23-05, left")  : 2.310,
        ("slow", "high", "e2v 10031-23-05, left")  : 4.590,
        ("fast", "high", "e2v 10031-23-05, left")  : 5.360,
        ("fast", "low",  "e2v 10031-23-05, left")  : 2.530,
        ("slow", "low",  "e2v 10031-01-03, left")  : 2.210,
        ("slow", "high", "e2v 10031-01-03, left")  : 4.590,
        ("fast", "high", "e2v 10031-01-03, left")  : 5.100,
        ("fast", "low",  "e2v 10031-01-03, left")  : 2.410,
        ("slow", "low",  "e2v 10031-18-04, right") : 2.330,
        ("slow", "high", "e2v 10031-18-04, right") : 4.330,
        ("fast", "high", "e2v 10031-18-04, right") : 5.410,
        ("fast", "low",  "e2v 10031-18-04, right") : 2.560,
        # EEV CCDs best amps
        ("slow", "low",  "EEV 9273-16-03, right") : 2.100,
        ("slow", "high", "EEV 9273-16-03, right") : 4.405,
        ("fast", "high", "EEV 9273-16-03, right") : 4.839,
        ("fast", "low",  "EEV 9273-16-03, right") : 2.302,
        ("slow", "low",  "EEV 9273-20-04, right") : 2.337,
        ("slow", "high", "EEV 9273-20-04, right") : 4.916,
        ("fast", "high", "EEV 9273-20-04, right") : 5.432,
        ("fast", "low",  "EEV 9273-20-04, right") : 2.578,
        ("slow", "low",  "EEV 9273-20-03, left")  : 2.300,
        ("slow", "high", "EEV 9273-20-03, left")  : 4.841,
        ("fast", "high", "EEV 9273-20-03, left")  : 5.397,
        ("fast", "low",  "EEV 9273-20-03, left")  : 2.560,
        # EEV CCDs secondary amps
        ("slow", "low",  "EEV 9273-16-03, left")  : 2.012,
        ("slow", "high", "EEV 9273-16-03, left")  : 4.247,
        ("fast", "high", "EEV 9273-16-03, left")  : 4.629,
        ("fast", "low",  "EEV 9273-16-03, left")  : 2.189,
        ("slow", "low",  "EEV 9273-20-04, left")  : 2.280,
        ("slow", "high", "EEV 9273-20-04, left")  : 4.809,
        ("fast", "high", "EEV 9273-20-04, left")  : 5.275,
        ("fast", "low",  "EEV 9273-20-04, left")  : 2.498,
        ("slow", "low",  "EEV 9273-20-03, right") : 2.257,
        ("slow", "high", "EEV 9273-20-03, right") : 4.779,
        ("fast", "high", "EEV 9273-20-03, right") : 5.354,
        ("fast", "low",  "EEV 9273-20-03, right") : 2.526,
        # GMOS-S:
        # EEV CCDs best amps
        ("slow", "low",  "EEV 8056-20-03, left")  : 1.940,
        ("slow", "high", "EEV 8056-20-03, left")  : 4.100,
        ("fast", "high", "EEV 8056-20-03, left")  : 5.300,
        ("fast", "low",  "EEV 8056-20-03, left")  : 2.400,
        ("slow", "low",  "EEV 8194-19-04, left")  : 2.076,
        ("slow", "high", "EEV 8194-19-04, left")  : 4.532,
        ("fast", "high", "EEV 8194-19-04, left")  : 5.051,
        ("fast", "low",  "EEV 8194-19-04, left")  : 2.295,
        ("slow", "low",  "EEV 8261-07-04, right") : 2.097,
        ("slow", "high", "EEV 8261-07-04, right") : 4.411,
        ("fast", "high", "EEV 8261-07-04, right") : 4.833,
        ("fast", "low",  "EEV 8261-07-04, right") : 2.260,
        # EEV CCDs secondary amps
        ("slow", "low",  "EEV 8056-20-03, right") : 2.000,
        ("slow", "high", "EEV 8056-20-03, right") : 4.200,
        ("fast", "high", "EEV 8056-20-03, right") : 5.300,
        ("fast", "low",  "EEV 8056-20-03, right") : 2.400,
        ("slow", "low",  "EEV 8194-19-04, right") : 2.131,
        ("slow", "high", "EEV 8194-19-04, right") : 4.592,
        ("fast", "high", "EEV 8194-19-04, right") : 4.95,
        ("fast", "low",  "EEV 8194-19-04, right") : 2.288,
        ("slow", "low",  "EEV 8261-07-04, left")  : 2.056,
        ("slow", "high", "EEV 8261-07-04, left")  : 4.381,
        ("fast", "high", "EEV 8261-07-04, left")  : 4.868,
        ("fast", "low",  "EEV 8261-07-04, left")  : 2.264,
        # new GMOS-S EEV CCD: Best/Secondary
        ("slow", "low",  "EEV 2037-06-03, left")  : 2.372,
        ("slow", "high", "EEV 2037-06-03, left")  : 4.954,
        ("fast", "high", "EEV 2037-06-03, left")  : 5.054,
        ("fast", "low",  "EEV 2037-06-03, left")  : 2.408,
        ("slow", "low",  "EEV 2037-06-03, right") : 2.403,
        ("slow", "high", "EEV 2037-06-03, right") : 4.862,
        ("fast", "high", "EEV 2037-06-03, right") : 5.253,
        ("fast", "low",  "EEV 2037-06-03, right") : 2.551,
        # GMOS-S Hamamatsu CCDs
        # AMPS 1-4 (CCD1)
        ("slow", "low",  "BI5-36-4k-2, 1"): 1.652,
        ("slow", "high", "BI5-36-4k-2, 1"): 1.000,
        ("fast", "high", "BI5-36-4k-2, 1"): 4.857,
        ("fast", "low",  "BI5-36-4k-2, 1"): 1.440,
        ("slow", "low",  "BI5-36-4k-2, 2"): 1.720,
        ("slow", "high", "BI5-36-4k-2, 2"): 1.000,
        ("fast", "high", "BI5-36-4k-2, 2"): 5.062,
        ("fast", "low",  "BI5-36-4k-2, 2"): 1.495,
        ("slow", "low",  "BI5-36-4k-2, 3"): 1.700,
        ("slow", "high", "BI5-36-4k-2, 3"): 1.000,
        ("fast", "high", "BI5-36-4k-2, 3"): 4.849,
        ("fast", "low",  "BI5-36-4k-2, 3"): 1.510,
        ("slow", "low",  "BI5-36-4k-2, 4"): 1.626,
        ("slow", "high", "BI5-36-4k-2, 4"): 1.000,
        ("fast", "high", "BI5-36-4k-2, 4"): 4.688,
        ("fast", "low",  "BI5-36-4k-2, 4"): 1.437,
        # AMPS 5-8 (CCD2)
        ("slow", "low",  "BI11-33-4k-1, 1"): 1.664,
        ("slow", "high", "BI11-33-4k-1, 1"): 1.000,
        ("fast", "high", "BI11-33-4k-1, 1"): 4.867,
        ("fast", "low",  "BI11-33-4k-1, 1"): 1.422,
        ("slow", "low",  "BI11-33-4k-1, 2"): 1.691,
        ("slow", "high", "BI11-33-4k-1, 2"): 1.000,
        ("fast", "high", "BI11-33-4k-1, 2"): 4.706,
        ("fast", "low",  "BI11-33-4k-1, 2"): 1.409,
        ("slow", "low",  "BI11-33-4k-1, 3"): 1.673,
        ("slow", "high", "BI11-33-4k-1, 3"): 1.000,
        ("fast", "high", "BI11-33-4k-1, 3"): 4.902,
        ("fast", "low",  "BI11-33-4k-1, 3"): 1.333,
        ("slow", "low",  "BI11-33-4k-1, 4"): 1.739,
        ("slow", "high", "BI11-33-4k-1, 4"): 1.000,
        ("fast", "high", "BI11-33-4k-1, 4"): 4.938,
        ("fast", "low",  "BI11-33-4k-1, 4"): 1.561,
        # AMPS 9-12 (CCD3)
        ("slow", "low",  "BI12-34-4k-1, 1"): 1.519,
        ("slow", "high", "BI12-34-4k-1, 1"): 1.000,
        ("fast", "high", "BI12-34-4k-1, 1"): 4.343,
        ("fast", "low",  "BI12-34-4k-1, 1"): 1.304,
        ("slow", "low",  "BI12-34-4k-1, 2"): 1.510,
        ("slow", "high", "BI12-34-4k-1, 2"): 1.000,
        ("fast", "high", "BI12-34-4k-1, 2"): 4.334,
        ("fast", "low",  "BI12-34-4k-1, 2"): 1.304,
        ("slow", "low",  "BI12-34-4k-1, 3"): 1.510,
        ("slow", "high", "BI12-34-4k-1, 3"): 1.000,
        ("fast", "high", "BI12-34-4k-1, 3"): 4.384,
        ("fast", "low",  "BI12-34-4k-1, 3"): 1.336,
        ("slow", "low",  "BI12-34-4k-1, 4"): 1.613,
        ("slow", "high", "BI12-34-4k-1, 4"): 1.000,
        ("fast", "high", "BI12-34-4k-1, 4"): 4.588,
        ("fast", "low",  "BI12-34-4k-1, 4"): 1.424,
        # AMPS 9-12 (CCD3)
        #  Repeated entries for CCD3 without the last comma
        ("slow", "low",  "BI12-34-4k-1 1"): 1.519,
        ("slow", "high", "BI12-34-4k-1 1"): 1.000,
        ("fast", "high", "BI12-34-4k-1 1"): 4.343,
        ("fast", "low",  "BI12-34-4k-1 1"): 1.304,
        ("slow", "low",  "BI12-34-4k-1 2"): 1.510,
        ("slow", "high", "BI12-34-4k-1 2"): 1.000,
        ("fast", "high", "BI12-34-4k-1 2"): 4.334,
        ("fast", "low",  "BI12-34-4k-1 2"): 1.304,
        ("slow", "low",  "BI12-34-4k-1 3"): 1.510,
        ("slow", "high", "BI12-34-4k-1 3"): 1.000,
        ("fast", "high", "BI12-34-4k-1 3"): 4.384,
        ("fast", "low",  "BI12-34-4k-1 3"): 1.336,
        ("slow", "low",  "BI12-34-4k-1 4"): 1.613,
        ("slow", "high", "BI12-34-4k-1 4"): 1.000,
        ("fast", "high", "BI12-34-4k-1 4"): 4.588,
        ("fast", "low",  "BI12-34-4k-1 4"): 1.424
    }

gmosampsGainBefore20060831 = {
        # Database of GMOS CCD amplifier GAIN properties
        # before 2006-08-31
        # Columns below are given as:
        # READOUT, GAINSTATE, AMPNAME : GAIN
        # GMOS-N
        # EEV CCDs best amps
        ("slow", "low",  "EEV 9273-16-03, right") : 2.100,
        ("slow", "high", "EEV 9273-16-03, right") : 4.405,
        ("fast", "high", "EEV 9273-16-03, right") : 4.839,
        ("fast", "low",  "EEV 9273-16-03, right") : 2.302,
        ("slow", "low",  "EEV 9273-20-04, right") : 2.337,
        ("slow", "high", "EEV 9273-20-04, right") : 4.916,
        ("fast", "high", "EEV 9273-20-04, right") : 5.432,
        ("fast", "low",  "EEV 9273-20-04, right") : 2.578,
        ("slow", "low",  "EEV 9273-20-03, left")  : 2.300,
        ("slow", "high", "EEV 9273-20-03, left")  : 4.841,
        ("fast", "high", "EEV 9273-20-03, left")  : 5.397,
        ("fast", "low",  "EEV 9273-20-03, left")  : 2.560,
        # EEV CCDs secondary amps
        ("slow", "low",  "EEV 9273-16-03, left")  : 2.012,
        ("slow", "high", "EEV 9273-16-03, left")  : 4.247,
        ("fast", "high", "EEV 9273-16-03, left")  : 4.629,
        ("fast", "low",  "EEV 9273-16-03, left")  : 2.189,
        ("slow", "low",  "EEV 9273-20-04, left")  : 2.280,
        ("slow", "high", "EEV 9273-20-04, left")  : 4.809,
        ("fast", "high", "EEV 9273-20-04, left")  : 5.275,
        ("fast", "low",  "EEV 9273-20-04, left")  : 2.498,
        ("slow", "low",  "EEV 9273-20-03, right") : 2.257,
        ("slow", "high", "EEV 9273-20-03, right") : 4.779,
        ("fast", "high", "EEV 9273-20-03, right") : 5.354,
        ("fast", "low",  "EEV 9273-20-03, right") : 2.526,
        # GMOS-S: New Gain/RN/Bias values (2006oct25) for
        # EEV CCDs best amps
        ("slow", "low",  "EEV 8056-20-03, left")  : 1.940,
        ("slow", "high", "EEV 8056-20-03, left")  : 4.100,
        ("fast", "high", "EEV 8056-20-03, left")  : 5.300,
        ("fast", "low",  "EEV 8056-20-03, left")  : 2.400,
        ("slow", "low",  "EEV 8194-19-04, left")  : 2.076,
        ("slow", "high", "EEV 8194-19-04, left")  : 4.532,
        ("fast", "high", "EEV 8194-19-04, left")  : 5.051,
        ("fast", "low",  "EEV 8194-19-04, left")  : 2.295,
        ("slow", "low",  "EEV 8261-07-04, right") : 2.097,
        ("slow", "high", "EEV 8261-07-04, right") : 4.411,
        ("fast", "high", "EEV 8261-07-04, right") : 4.833,
        ("fast", "low",  "EEV 8261-07-04, right") : 2.260,
        # EEV CCDs secondary amps
        ("slow", "low",  "EEV 8056-20-03, right") : 2.000,
        ("slow", "high", "EEV 8056-20-03, right") : 4.200,
        ("fast", "high", "EEV 8056-20-03, right") : 5.300,
        ("fast", "low",  "EEV 8056-20-03, right") : 2.400,
        ("slow", "low",  "EEV 8194-19-04, right") : 2.131,
        ("slow", "high", "EEV 8194-19-04, right") : 4.592,
        ("fast", "high", "EEV 8194-19-04, right") : 4.95,
        ("fast", "low",  "EEV 8194-19-04, right") : 2.288,
        ("slow", "low",  "EEV 8261-07-04, left")  : 2.056,
        ("slow", "high", "EEV 8261-07-04, left")  : 4.381,
        ("fast", "high", "EEV 8261-07-04, left")  : 4.868,
        ("fast", "low",  "EEV 8261-07-04, left")  : 2.264,
        # new GMOS-S EEV CCD: Best/Secondary
        ("slow", "low",  "EEV 2037-06-03, left")  : 2.372,
        ("slow", "high", "EEV 2037-06-03, left")  : 4.954,
        ("fast", "high", "EEV 2037-06-03, left")  : 5.054,
        ("fast", "low",  "EEV 2037-06-03, left")  : 2.408,
        ("slow", "low",  "EEV 2037-06-03, right") : 2.403,
        ("slow", "high", "EEV 2037-06-03, right") : 4.862,
        ("fast", "high", "EEV 2037-06-03, right") : 5.253,
        ("fast", "low",  "EEV 2037-06-03, right") : 2.551
    }

gmosampsRdnoise = {
    # Database of GMOS CCD amplifier READNOISE properties
    # after 2017-02-24
    # Columns below are given as:
    # READOUT, GAINSTATE, AMPNAME : READNOISE (in electrons)
    # GMOS-N: Hamamatsu CCDs
    # Latest values from K.Chiboucas on 2017-05-05.
    ("slow", "low", "BI13-20-4k-1, 1"): 3.99,
    ("slow", "high", "BI13-20-4k-1, 1"): 0.00,
    ("fast", "high", "BI13-20-4k-1, 1"): 8.17,
    ("fast", "low", "BI13-20-4k-1, 1"): 5.88,
    ("slow", "low", "BI13-20-4k-1, 2"): 4.12,
    ("slow", "high", "BI13-20-4k-1, 2"): 0.00,
    ("fast", "high", "BI13-20-4k-1, 2"): 8.75,
    ("fast", "low", "BI13-20-4k-1, 2"): 6.01,
    ("slow", "low", "BI13-20-4k-1, 3"): 4.12,
    ("slow", "high", "BI13-20-4k-1, 3"): 0.00,
    ("fast", "high", "BI13-20-4k-1, 3"): 8.94,
    ("fast", "low", "BI13-20-4k-1, 3"): 6.88,
    ("slow", "low", "BI13-20-4k-1, 4"): 4.06,
    ("slow", "high", "BI13-20-4k-1, 4"): 0.00,
    ("fast", "high", "BI13-20-4k-1, 4"): 9.90,
    ("fast", "low", "BI13-20-4k-1, 4"): 8.02,
    # AMPS 8-5 (CCD2)
    ("slow", "low", "BI12-09-4k-2, 1"): 4.20,
    ("slow", "high", "BI12-09-4k-2, 1"): 0.00,
    ("fast", "high", "BI12-09-4k-2, 1"): 8.84,
    ("fast", "low", "BI12-09-4k-2, 1"): 5.86,
    ("slow", "low", "BI12-09-4k-2, 2"): 3.88,
    ("slow", "high", "BI12-09-4k-2, 2"): 0.00,
    ("fast", "high", "BI12-09-4k-2, 2"): 8.19,
    ("fast", "low", "BI12-09-4k-2, 2"): 5.39,
    ("slow", "low", "BI12-09-4k-2, 3"): 3.98,
    ("slow", "high", "BI12-09-4k-2, 3"): 0.00,
    ("fast", "high", "BI12-09-4k-2, 3"): 8.91,
    ("fast", "low", "BI12-09-4k-2, 3"): 6.34,
    ("slow", "low", "BI12-09-4k-2, 4"): 4.20,
    ("slow", "high", "BI12-09-4k-2, 4"): 0.00,
    ("fast", "high", "BI12-09-4k-2, 4"): 9.42,
    ("fast", "low", "BI12-09-4k-2, 4"): 6.71,
    # AMPS 12-9 (CCD3)
    ("slow", "low", "BI13-18-4k-2, 1"): 4.55,
    ("slow", "high", "BI13-18-4k-2, 1"): 0.00,
    ("fast", "high", "BI13-18-4k-2, 1"): 8.80,
    ("fast", "low", "BI13-18-4k-2, 1"): 6.79,
    ("slow", "low", "BI13-18-4k-2, 2"): 4.02,
    ("slow", "high", "BI13-18-4k-2, 2"): 0.00,
    ("fast", "high", "BI13-18-4k-2, 2"): 8.03,
    ("fast", "low", "BI13-18-4k-2, 2"): 5.98,
    ("slow", "low", "BI13-18-4k-2, 3"): 4.35,
    ("slow", "high", "BI13-18-4k-2, 3"): 0.00,
    ("fast", "high", "BI13-18-4k-2, 3"): 8.53,
    ("fast", "low", "BI13-18-4k-2, 3"): 6.13,
    ("slow", "low", "BI13-18-4k-2, 4"): 4.04,
    ("slow", "high", "BI13-18-4k-2, 4"): 0.00,
    ("fast", "high", "BI13-18-4k-2, 4"): 8.13,
    ("fast", "low", "BI13-18-4k-2, 4"): 6.10,

    # GMOS-S New Hamamatsu CCDs + swap (2023)
    # Values from German Gimeno on Jan 22, 2024
    # Valid from 20231214

    # AMPS 4-1 (CCD1)
    ("slow", "low",  "BI11-41-4k-2, 1"): 4.22,
    ("slow", "high", "BI11-41-4k-2, 1"): 0.00,
    ("fast", "high", "BI11-41-4k-2, 1"): 11.74,
    ("fast", "low",  "BI11-41-4k-2, 1"): 7.01,
    ("slow", "low",  "BI11-41-4k-2, 2"): 4.29,
    ("slow", "high", "BI11-41-4k-2, 2"): 0.00,
    ("fast", "high", "BI11-41-4k-2, 2"): 10.19,
    ("fast", "low",  "BI11-41-4k-2, 2"): 6.93,
    ("slow", "low",  "BI11-41-4k-2, 3"): 4.20,
    ("slow", "high", "BI11-41-4k-2, 3"): 0.00,
    ("fast", "high", "BI11-41-4k-2, 3"): 10.45,
    ("fast", "low",  "BI11-41-4k-2, 3"): 6.37,
    ("slow", "low",  "BI11-41-4k-2, 4"): 4.05,
    ("slow", "high", "BI11-41-4k-2, 4"): 0.00,
    ("fast", "high", "BI11-41-4k-2, 4"): 9.06,
    ("fast", "low",  "BI11-41-4k-2, 4"): 5.19,
    # AMPS 8-5 (CCD2)
    ("slow", "low",  "BI13-19-4k-3, 1"): 3.79,
    ("slow", "high", "BI13-19-4k-3, 1"): 0.00,
    ("fast", "high", "BI13-19-4k-3, 1"): 11.27,
    ("fast", "low",  "BI13-19-4k-3, 1"): 5.20,
    ("slow", "low",  "BI13-19-4k-3, 2"): 3.68,
    ("slow", "high", "BI13-19-4k-3, 2"): 0.00,
    ("fast", "high", "BI13-19-4k-3, 2"): 9.46,
    ("fast", "low",  "BI13-19-4k-3, 2"): 5.82,
    ("slow", "low",  "BI13-19-4k-3, 3"): 3.59,
    ("slow", "high", "BI13-19-4k-3, 3"): 0.00,
    ("fast", "high", "BI13-19-4k-3, 3"): 7.99,
    ("fast", "low",  "BI13-19-4k-3, 3"): 5.24,
    ("slow", "low",  "BI13-19-4k-3, 4"): 4.22,
    ("slow", "high", "BI13-19-4k-3, 4"): 0.00,
    ("fast", "high", "BI13-19-4k-3, 4"): 9.52,
    ("fast", "low",  "BI13-19-4k-3, 4"): 5.46,
    # AMPS 12-9 (CCD3)
    ("slow", "low",  "BI12-34-4k-1, 1"): 3.39,
    ("slow", "high", "BI12-34-4k-1, 1"): 0.00,
    ("fast", "high", "BI12-34-4k-1, 1"): 8.85,
    ("fast", "low",  "BI12-34-4k-1, 1"): 4.66,
    ("slow", "low",  "BI12-34-4k-1, 2"): 3.67,
    ("slow", "high", "BI12-34-4k-1, 2"): 0.00,
    ("fast", "high", "BI12-34-4k-1, 2"): 7.96,
    ("fast", "low",  "BI12-34-4k-1, 2"): 5.10,
    ("slow", "low",  "BI12-34-4k-1, 3"): 3.39,
    ("slow", "high", "BI12-34-4k-1, 3"): 0.00,
    ("fast", "high", "BI12-34-4k-1, 3"): 8.48,
    ("fast", "low",  "BI12-34-4k-1, 3"): 5.39,
    ("slow", "low",  "BI12-34-4k-1, 4"): 3.53,
    ("slow", "high", "BI12-34-4k-1, 4"): 0.00,
    ("fast", "high", "BI12-34-4k-1, 4"): 7.83,
    ("fast", "low",  "BI12-34-4k-1, 4"): 9.57
}

gmosampsRdnoiseBefore20231214 = {
    # Database of GMOS CCD amplifier READNOISE properties
    # after 2017-02-24
    # Columns below are given as:
    # READOUT, GAINSTATE, AMPNAME : READNOISE (in electrons)
    # GMOS-N: Hamamatsu CCDs
    # Latest values from K.Chiboucas on 2017-05-05.
    ("slow", "low", "BI13-20-4k-1, 1"): 3.99,
    ("slow", "high", "BI13-20-4k-1, 1"): 0.00,
    ("fast", "high", "BI13-20-4k-1, 1"): 8.17,
    ("fast", "low", "BI13-20-4k-1, 1"): 5.88,
    ("slow", "low", "BI13-20-4k-1, 2"): 4.12,
    ("slow", "high", "BI13-20-4k-1, 2"): 0.00,
    ("fast", "high", "BI13-20-4k-1, 2"): 8.75,
    ("fast", "low", "BI13-20-4k-1, 2"): 6.01,
    ("slow", "low", "BI13-20-4k-1, 3"): 4.12,
    ("slow", "high", "BI13-20-4k-1, 3"): 0.00,
    ("fast", "high", "BI13-20-4k-1, 3"): 8.94,
    ("fast", "low", "BI13-20-4k-1, 3"): 6.88,
    ("slow", "low", "BI13-20-4k-1, 4"): 4.06,
    ("slow", "high", "BI13-20-4k-1, 4"): 0.00,
    ("fast", "high", "BI13-20-4k-1, 4"): 9.90,
    ("fast", "low", "BI13-20-4k-1, 4"): 8.02,
    # AMPS 8-5 (CCD2)
    ("slow", "low", "BI12-09-4k-2, 1"): 4.20,
    ("slow", "high", "BI12-09-4k-2, 1"): 0.00,
    ("fast", "high", "BI12-09-4k-2, 1"): 8.84,
    ("fast", "low", "BI12-09-4k-2, 1"): 5.86,
    ("slow", "low", "BI12-09-4k-2, 2"): 3.88,
    ("slow", "high", "BI12-09-4k-2, 2"): 0.00,
    ("fast", "high", "BI12-09-4k-2, 2"): 8.19,
    ("fast", "low", "BI12-09-4k-2, 2"): 5.39,
    ("slow", "low", "BI12-09-4k-2, 3"): 3.98,
    ("slow", "high", "BI12-09-4k-2, 3"): 0.00,
    ("fast", "high", "BI12-09-4k-2, 3"): 8.91,
    ("fast", "low", "BI12-09-4k-2, 3"): 6.34,
    ("slow", "low", "BI12-09-4k-2, 4"): 4.20,
    ("slow", "high", "BI12-09-4k-2, 4"): 0.00,
    ("fast", "high", "BI12-09-4k-2, 4"): 9.42,
    ("fast", "low", "BI12-09-4k-2, 4"): 6.71,
    # AMPS 12-9 (CCD3)
    ("slow", "low", "BI13-18-4k-2, 1"): 4.55,
    ("slow", "high", "BI13-18-4k-2, 1"): 0.00,
    ("fast", "high", "BI13-18-4k-2, 1"): 8.80,
    ("fast", "low", "BI13-18-4k-2, 1"): 6.79,
    ("slow", "low", "BI13-18-4k-2, 2"): 4.02,
    ("slow", "high", "BI13-18-4k-2, 2"): 0.00,
    ("fast", "high", "BI13-18-4k-2, 2"): 8.03,
    ("fast", "low", "BI13-18-4k-2, 2"): 5.98,
    ("slow", "low", "BI13-18-4k-2, 3"): 4.35,
    ("slow", "high", "BI13-18-4k-2, 3"): 0.00,
    ("fast", "high", "BI13-18-4k-2, 3"): 8.53,
    ("fast", "low", "BI13-18-4k-2, 3"): 6.13,
    ("slow", "low", "BI13-18-4k-2, 4"): 4.04,
    ("slow", "high", "BI13-18-4k-2, 4"): 0.00,
    ("fast", "high", "BI13-18-4k-2, 4"): 8.13,
    ("fast", "low", "BI13-18-4k-2, 4"): 6.10,

    # GMOS-S Hamamatsu CCDs (new E video boards, 2015)
    # AMPS 4-1 (CCD1)
    ("slow", "low",  "BI5-36-4k-2, 1"): 4.21,
    ("slow", "high", "BI5-36-4k-2, 1"): 0.00,
    ("fast", "high", "BI5-36-4k-2, 1"): 8.93,
    ("fast", "low",  "BI5-36-4k-2, 1"): 6.68,
    ("slow", "low",  "BI5-36-4k-2, 2"): 4.24,
    ("slow", "high", "BI5-36-4k-2, 2"): 0.00,
    ("fast", "high", "BI5-36-4k-2, 2"): 9.13,
    ("fast", "low",  "BI5-36-4k-2, 2"): 7.10,
    ("slow", "low",  "BI5-36-4k-2, 3"): 4.28,
    ("slow", "high", "BI5-36-4k-2, 3"): 0.00,
    ("fast", "high", "BI5-36-4k-2, 3"): 8.85,
    ("fast", "low",  "BI5-36-4k-2, 3"): 6.35,
    ("slow", "low",  "BI5-36-4k-2, 4"): 4.04,
    ("slow", "high", "BI5-36-4k-2, 4"): 0.00,
    ("fast", "high", "BI5-36-4k-2, 4"): 8.07,
    ("fast", "low",  "BI5-36-4k-2, 4"): 5.49,
    # AMPS 8-5 (CCD2)
    ("slow", "low",  "BI11-33-4k-1, 1"): 4.29,
    ("slow", "high", "BI11-33-4k-1, 1"): 0.00,
    ("fast", "high", "BI11-33-4k-1, 1"): 8.35,
    ("fast", "low",  "BI11-33-4k-1, 1"): 5.19,
    ("slow", "low",  "BI11-33-4k-1, 2"): 4.07,
    ("slow", "high", "BI11-33-4k-1, 2"): 0.00,
    ("fast", "high", "BI11-33-4k-1, 2"): 8.59,
    ("fast", "low",  "BI11-33-4k-1, 2"): 6.14,
    ("slow", "low",  "BI11-33-4k-1, 3"): 4.24,
    ("slow", "high", "BI11-33-4k-1, 3"): 0.00,
    ("fast", "high", "BI11-33-4k-1, 3"): 8.62,
    ("fast", "low",  "BI11-33-4k-1, 3"): 7.38,
    ("slow", "low",  "BI11-33-4k-1, 4"): 3.90,
    ("slow", "high", "BI11-33-4k-1, 4"): 0.00,
    ("fast", "high", "BI11-33-4k-1, 4"): 8.02,
    ("fast", "low",  "BI11-33-4k-1, 4"): 5.77,
    # AMPS 12-9 (CCD3)
    ("slow", "low",  "BI12-34-4k-1, 1"): 3.53,
    ("slow", "high", "BI12-34-4k-1, 1"): 0.00,
    ("fast", "high", "BI12-34-4k-1, 1"): 7.94,
    ("fast", "low",  "BI12-34-4k-1, 1"): 6.36,
    ("slow", "low",  "BI12-34-4k-1, 2"): 3.72,
    ("slow", "high", "BI12-34-4k-1, 2"): 0.00,
    ("fast", "high", "BI12-34-4k-1, 2"): 8.30,
    ("fast", "low",  "BI12-34-4k-1, 2"): 6.80,
    ("slow", "low",  "BI12-34-4k-1, 3"): 3.61,
    ("slow", "high", "BI12-34-4k-1, 3"): 0.00,
    ("fast", "high", "BI12-34-4k-1, 3"): 7.99,
    ("fast", "low",  "BI12-34-4k-1, 3"): 7.88,
    ("slow", "low",  "BI12-34-4k-1, 4"): 4.15,
    ("slow", "high", "BI12-34-4k-1, 4"): 0.00,
    ("fast", "high", "BI12-34-4k-1, 4"): 9.34,
    ("fast", "low",  "BI12-34-4k-1, 4"): 7.72
}

gmosampsRdnoiseBefore20170224 = {
        # Database of GMOS CCD amplifier READNOISE properties
        # after 2015-08-26 and before 2017-02-24
        # Columns below are given as:
        # READOUT, GAINSTATE, AMPNAME : READNOISE (in electrons)
        # GMOS-N:
        # e2vDD CCDs best amps
        ("slow", "low",  "e2v 10031-23-05, right") : 3.17,
        ("slow", "high", "e2v 10031-23-05, right") : 3.20,
        ("fast", "high", "e2v 10031-23-05, right") : 7.80,
        ("fast", "low",  "e2v 10031-23-05, right") : 4.20,
        ("slow", "low",  "e2v 10031-01-03, right") : 3.22,
        ("slow", "high", "e2v 10031-01-03, right") : 1.50,
        ("fast", "high", "e2v 10031-01-03, right") : 5.80,
        ("fast", "low",  "e2v 10031-01-03, right") : 4.10,
        ("slow", "low",  "e2v 10031-18-04, left")  : 3.46,
        ("slow", "high", "e2v 10031-18-04, left")  : 2.80,
        ("fast", "high", "e2v 10031-18-04, left")  : 6.40,
        ("fast", "low",  "e2v 10031-18-04, left")  : 4.30,
        # e2vDD CCDs secondary amps
        ("slow", "low",  "e2v 10031-23-05, left")  : 3.41,
        ("slow", "high", "e2v 10031-23-05, left")  : 3.10,
        ("fast", "high", "e2v 10031-23-05, left")  : 7.00,
        ("fast", "low",  "e2v 10031-23-05, left")  : 4.30,
        ("slow", "low",  "e2v 10031-01-03, left")  : 3.20,
        ("slow", "high", "e2v 10031-01-03, left")  : 2.40,
        ("fast", "high", "e2v 10031-01-03, left")  : 5.70,
        ("fast", "low",  "e2v 10031-01-03, left")  : 4.10,
        ("slow", "low",  "e2v 10031-18-04, right") : 3.44,
        ("slow", "high", "e2v 10031-18-04, right") : 3.10,
        ("fast", "high", "e2v 10031-18-04, right") : 6.30,
        ("fast", "low",  "e2v 10031-18-04, right") : 4.40,
        # GMOS-S Hamamatsu CCDs (new E video boards, 2015)
        # AMPS 4-1 (CCD1)
        ("slow", "low",  "BI5-36-4k-2, 1"): 4.21,
        ("slow", "high", "BI5-36-4k-2, 1"): 0.00,
        ("fast", "high", "BI5-36-4k-2, 1"): 8.93,
        ("fast", "low",  "BI5-36-4k-2, 1"): 6.68,
        ("slow", "low",  "BI5-36-4k-2, 2"): 4.24,
        ("slow", "high", "BI5-36-4k-2, 2"): 0.00,
        ("fast", "high", "BI5-36-4k-2, 2"): 9.13,
        ("fast", "low",  "BI5-36-4k-2, 2"): 7.10,
        ("slow", "low",  "BI5-36-4k-2, 3"): 4.28,
        ("slow", "high", "BI5-36-4k-2, 3"): 0.00,
        ("fast", "high", "BI5-36-4k-2, 3"): 8.85,
        ("fast", "low",  "BI5-36-4k-2, 3"): 6.35,
        ("slow", "low",  "BI5-36-4k-2, 4"): 4.04,
        ("slow", "high", "BI5-36-4k-2, 4"): 0.00,
        ("fast", "high", "BI5-36-4k-2, 4"): 8.07,
        ("fast", "low",  "BI5-36-4k-2, 4"): 5.49,
        # AMPS 8-5 (CCD2)
        ("slow", "low",  "BI11-33-4k-1, 1"): 4.29,
        ("slow", "high", "BI11-33-4k-1, 1"): 0.00,
        ("fast", "high", "BI11-33-4k-1, 1"): 8.35,
        ("fast", "low",  "BI11-33-4k-1, 1"): 5.19,
        ("slow", "low",  "BI11-33-4k-1, 2"): 4.07,
        ("slow", "high", "BI11-33-4k-1, 2"): 0.00,
        ("fast", "high", "BI11-33-4k-1, 2"): 8.59,
        ("fast", "low",  "BI11-33-4k-1, 2"): 6.14,
        ("slow", "low",  "BI11-33-4k-1, 3"): 4.24,
        ("slow", "high", "BI11-33-4k-1, 3"): 0.00,
        ("fast", "high", "BI11-33-4k-1, 3"): 8.62,
        ("fast", "low",  "BI11-33-4k-1, 3"): 7.38,
        ("slow", "low",  "BI11-33-4k-1, 4"): 3.90,
        ("slow", "high", "BI11-33-4k-1, 4"): 0.00,
        ("fast", "high", "BI11-33-4k-1, 4"): 8.02,
        ("fast", "low",  "BI11-33-4k-1, 4"): 5.77,
        # AMPS 12-9 (CCD3)
        ("slow", "low",  "BI12-34-4k-1, 1"): 3.53,
        ("slow", "high", "BI12-34-4k-1, 1"): 0.00,
        ("fast", "high", "BI12-34-4k-1, 1"): 7.94,
        ("fast", "low",  "BI12-34-4k-1, 1"): 6.36,
        ("slow", "low",  "BI12-34-4k-1, 2"): 3.72,
        ("slow", "high", "BI12-34-4k-1, 2"): 0.00,
        ("fast", "high", "BI12-34-4k-1, 2"): 8.30,
        ("fast", "low",  "BI12-34-4k-1, 2"): 6.80,
        ("slow", "low",  "BI12-34-4k-1, 3"): 3.61,
        ("slow", "high", "BI12-34-4k-1, 3"): 0.00,
        ("fast", "high", "BI12-34-4k-1, 3"): 7.99,
        ("fast", "low",  "BI12-34-4k-1, 3"): 7.88,
        ("slow", "low",  "BI12-34-4k-1, 4"): 4.15,
        ("slow", "high", "BI12-34-4k-1, 4"): 0.00,
        ("fast", "high", "BI12-34-4k-1, 4"): 9.34,
        ("fast", "low",  "BI12-34-4k-1, 4"): 7.72
    }

gmosampsRdnoiseBefore20150826 = {
        # Database of GMOS CCD amplifier READNOISE properties
        # after 2006-08-31
        # Columns below are given as:
        # READOUT, GAINSTATE, AMPNAME : READNOISE (in electrons)
        # GMOS-N:
        # e2vDD CCDs best amps
        ("slow", "low",  "e2v 10031-23-05, right") : 3.17,
        ("slow", "high", "e2v 10031-23-05, right") : 3.20,
        ("fast", "high", "e2v 10031-23-05, right") : 7.80,
        ("fast", "low",  "e2v 10031-23-05, right") : 4.20,
        ("slow", "low",  "e2v 10031-01-03, right") : 3.22,
        ("slow", "high", "e2v 10031-01-03, right") : 1.50,
        ("fast", "high", "e2v 10031-01-03, right") : 5.80,
        ("fast", "low",  "e2v 10031-01-03, right") : 4.10,
        ("slow", "low",  "e2v 10031-18-04, left")  : 3.46,
        ("slow", "high", "e2v 10031-18-04, left")  : 2.80,
        ("fast", "high", "e2v 10031-18-04, left")  : 6.40,
        ("fast", "low",  "e2v 10031-18-04, left")  : 4.30,
        # e2vDD CCDs secondary amps
        ("slow", "low",  "e2v 10031-23-05, left")  : 3.41,
        ("slow", "high", "e2v 10031-23-05, left")  : 3.10,
        ("fast", "high", "e2v 10031-23-05, left")  : 7.00,
        ("fast", "low",  "e2v 10031-23-05, left")  : 4.30,
        ("slow", "low",  "e2v 10031-01-03, left")  : 3.20,
        ("slow", "high", "e2v 10031-01-03, left")  : 2.40,
        ("fast", "high", "e2v 10031-01-03, left")  : 5.70,
        ("fast", "low",  "e2v 10031-01-03, left")  : 4.10,
        ("slow", "low",  "e2v 10031-18-04, right") : 3.44,
        ("slow", "high", "e2v 10031-18-04, right") : 3.10,
        ("fast", "high", "e2v 10031-18-04, right") : 6.30,
        ("fast", "low",  "e2v 10031-18-04, right") : 4.40,
        # EEV best amps
        ("slow", "low",  "EEV 9273-16-03, right") : 3.5,
        ("slow", "high", "EEV 9273-16-03, right") : 5.1,
        ("fast", "high", "EEV 9273-16-03, right") : 6.6,
        ("fast", "low",  "EEV 9273-16-03, right") : 4.9,
        ("slow", "low",  "EEV 9273-20-04, right") : 3.3,
        ("slow", "high", "EEV 9273-20-04, right") : 5.1,
        ("fast", "high", "EEV 9273-20-04, right") : 7.0,
        ("fast", "low",  "EEV 9273-20-04, right") : 4.7,
        ("slow", "low",  "EEV 9273-20-03, left")  : 3.0,
        ("slow", "high", "EEV 9273-20-03, left")  : 4.8,
        ("fast", "high", "EEV 9273-20-03, left")  : 6.9,
        ("fast", "low",  "EEV 9273-20-03, left")  : 4.8,
        # EEV secondary amps
        ("slow", "low",  "EEV 9273-16-03, left")  : 3.5,
        ("slow", "high", "EEV 9273-16-03, left")  : 4.6,
        ("fast", "high", "EEV 9273-16-03, left")  : 5.9,
        ("fast", "low",  "EEV 9273-16-03, left")  : 4.5,
        ("slow", "low",  "EEV 9273-20-04, left")  : 3.9,
        ("slow", "high", "EEV 9273-20-04, left")  : 4.8,
        ("fast", "high", "EEV 9273-20-04, left")  : 11.0,
        ("fast", "low",  "EEV 9273-20-04, left")  : 4.8,
        ("slow", "low",  "EEV 9273-20-03, right") : 3.3,
        ("slow", "high", "EEV 9273-20-03, right") : 5.0,
        ("fast", "high", "EEV 9273-20-03, right") : 8.0,
        ("fast", "low",  "EEV 9273-20-03, right") : 4.7,
        # GMOS-S:
        # EEV best amps
        ("slow", "low",  "EEV 8056-20-03, left")  : 3.20,
        ("slow", "high", "EEV 8056-20-03, left")  : 4.20,
        ("fast", "high", "EEV 8056-20-03, left")  : 6.80,
        ("fast", "low",  "EEV 8056-20-03, left")  : 4.40,
        ("slow", "low",  "EEV 8194-19-04, left")  : 3.85,
        ("slow", "high", "EEV 8194-19-04, left")  : 4.81,
        ("fast", "high", "EEV 8194-19-04, left")  : 7.35,
        ("fast", "low",  "EEV 8194-19-04, left")  : 4.42,
        ("slow", "low",  "EEV 8261-07-04, right") : 3.16,
        ("slow", "high", "EEV 8261-07-04, right") : 4.34,
        ("fast", "high", "EEV 8261-07-04, right") : 7.88,
        ("fast", "low",  "EEV 8261-07-04, right") : 4.09,
        # EEV secondary amps
        ("slow", "low",  "EEV 8056-20-03, right") : 3.40,
        ("slow", "high", "EEV 8056-20-03, right") : 4.90,
        ("fast", "high", "EEV 8056-20-03, right") : 6.30,
        ("fast", "low",  "EEV 8056-20-03, right") : 4.30,
        ("slow", "low",  "EEV 8194-19-04, right") : 3.83,
        ("slow", "high", "EEV 8194-19-04, right") : 4.96,
        ("fast", "high", "EEV 8194-19-04, right") : 11.37,
        ("fast", "low",  "EEV 8194-19-04, right") : 5.03,
        ("slow", "low",  "EEV 8261-07-04, left")  : 3.27,
        ("slow", "high", "EEV 8261-07-04, left")  : 4.81,
        ("fast", "high", "EEV 8261-07-04, left")  : 8.56,
        ("fast", "low",  "EEV 8261-07-04, left")  : 4.63,
        # new GMOS-S EEV CCD: Best/Secondary
        ("slow", "low",  "EEV 2037-06-03, left")  : 3.98,
        ("slow", "high", "EEV 2037-06-03, left")  : 5.70,
        ("fast", "high", "EEV 2037-06-03, left")  : 6.84,
        ("fast", "low",  "EEV 2037-06-03, left")  : 4.49,
        ("slow", "low",  "EEV 2037-06-03, right") : 4.01,
        ("slow", "high", "EEV 2037-06-03, right") : 5.19,
        ("fast", "high", "EEV 2037-06-03, right") : 6.54,
        ("fast", "low",  "EEV 2037-06-03, right") : 5.07,
        # GMOS-S Hamamatsu CCDs
        # AMPS 4-1 (CCD1)
        ("slow", "low",  "BI5-36-4k-2, 1"): 4.240,
        ("slow", "high", "BI5-36-4k-2, 1"): 0.000,
        ("fast", "high", "BI5-36-4k-2, 1"): 10.426,
        ("fast", "low",  "BI5-36-4k-2, 1"): 7.725,
        ("slow", "low",  "BI5-36-4k-2, 2"): 4.000,
        ("slow", "high", "BI5-36-4k-2, 2"): 0.00,
        ("fast", "high", "BI5-36-4k-2, 2"): 8.109,
        ("fast", "low",  "BI5-36-4k-2, 2"): 6.891,
        ("slow", "low",  "BI5-36-4k-2, 3"): 4.250,
        ("slow", "high", "BI5-36-4k-2, 3"): 0.00,
        ("fast", "high", "BI5-36-4k-2, 3"): 8.248,
        ("fast", "low",  "BI5-36-4k-2, 3"): 7.244,
        ("slow", "low",  "BI5-36-4k-2, 4"): 4.030,
        ("slow", "high", "BI5-36-4k-2, 4"): 0.00,
        ("fast", "high", "BI5-36-4k-2, 4"): 8.283,
        ("fast", "low",  "BI5-36-4k-2, 4"): 7.114,
        # AMPS 8-5 (CCD2)
        ("slow", "low",  "BI11-33-4k-1, 1"): 4.120,
        ("slow", "high", "BI11-33-4k-1, 1"): 0.00,
        ("fast", "high", "BI11-33-4k-1, 1"): 8.223,
        ("fast", "low",  "BI11-33-4k-1, 1"): 5.986,
        ("slow", "low",  "BI11-33-4k-1, 2"): 3.830,
        ("slow", "high", "BI11-33-4k-1, 2"): 0.00,
        ("fast", "high", "BI11-33-4k-1, 2"): 7.939,
        ("fast", "low",  "BI11-33-4k-1, 2"): 6.660,
        ("slow", "low",  "BI11-33-4k-1, 3"): 3.980,
        ("slow", "high", "BI11-33-4k-1, 3"): 0.00,
        ("fast", "high", "BI11-33-4k-1, 3"): 7.650,
        ("fast", "low",  "BI11-33-4k-1, 3"): 5.658,
        ("slow", "low",  "BI11-33-4k-1, 4"): 3.800,
        ("slow", "high", "BI11-33-4k-1, 4"): 0.00,
        ("fast", "high", "BI11-33-4k-1, 4"): 8.255,
        ("fast", "low",  "BI11-33-4k-1, 4"): 7.490,
        # AMPS 12-9 (CCD3)
        ("slow", "low",  "BI12-34-4k-1, 1"): 3.460,
        ("slow", "high", "BI12-34-4k-1, 1"): 0.00,
        ("fast", "high", "BI12-34-4k-1, 1"): 7.460,
        ("fast", "low",  "BI12-34-4k-1, 1"): 6.556,
        ("slow", "low",  "BI12-34-4k-1, 2"): 3.350,
        ("slow", "high", "BI12-34-4k-1, 2"): 0.00,
        ("fast", "high", "BI12-34-4k-1, 2"): 6.522,
        ("fast", "low",  "BI12-34-4k-1, 2"): 5.401,
        ("slow", "low",  "BI12-34-4k-1, 3"): 3.250,
        ("slow", "high", "BI12-34-4k-1, 3"): 0.00,
        ("fast", "high", "BI12-34-4k-1, 3"): 7.383,
        ("fast", "low",  "BI12-34-4k-1, 3"): 5.907,
        ("slow", "low",  "BI12-34-4k-1, 4"): 3.500,
        ("slow", "high", "BI12-34-4k-1, 4"): 0.00,
        ("fast", "high", "BI12-34-4k-1, 4"): 6.574,
        ("fast", "low",  "BI12-34-4k-1, 4"): 5.446,
        # AMPS 12-9 (CCD3)
        # Repeated entries for CCD3 without the last comma
        ("slow", "low",  "BI12-34-4k-1 1"): 3.460,
        ("slow", "high", "BI12-34-4k-1 1"): 0.00,
        ("fast", "high", "BI12-34-4k-1 1"): 7.460,
        ("fast", "low",  "BI12-34-4k-1 1"): 6.556,
        ("slow", "low",  "BI12-34-4k-1 2"): 3.350,
        ("slow", "high", "BI12-34-4k-1 2"): 0.00,
        ("fast", "high", "BI12-34-4k-1 2"): 6.522,
        ("fast", "low",  "BI12-34-4k-1 2"): 5.401,
        ("slow", "low",  "BI12-34-4k-1 3"): 3.250,
        ("slow", "high", "BI12-34-4k-1 3"): 0.00,
        ("fast", "high", "BI12-34-4k-1 3"): 7.383,
        ("fast", "low",  "BI12-34-4k-1 3"): 5.907,
        ("slow", "low",  "BI12-34-4k-1 4"): 3.500,
        ("slow", "high", "BI12-34-4k-1 4"): 0.00,
        ("fast", "high", "BI12-34-4k-1 4"): 6.574,
        ("fast", "low",  "BI12-34-4k-1 4"): 5.446
    }

gmosampsRdnoiseBefore20060831 = {
        # Database of GMOS CCD amplifier READNOISE properties
        # before 2006-08-31
        # Columns below are given as:
        # READOUT, GAINSTATE, AMPNAME : READNOISE
        # GMOS-N:
        # EEV best amps
        ("slow", "low",  "EEV 9273-16-03, right") : 3.5,
        ("slow", "high", "EEV 9273-16-03, right") : 5.1,
        ("fast", "high", "EEV 9273-16-03, right") : 6.6,
        ("fast", "low",  "EEV 9273-16-03, right") : 4.9,
        ("slow", "low",  "EEV 9273-20-04, right") : 3.3,
        ("slow", "high", "EEV 9273-20-04, right") : 5.1,
        ("fast", "high", "EEV 9273-20-04, right") : 7.0,
        ("fast", "low",  "EEV 9273-20-04, right") : 4.7,
        ("slow", "low",  "EEV 9273-20-03, left")  : 3.0,
        ("slow", "high", "EEV 9273-20-03, left")  : 4.8,
        ("fast", "high", "EEV 9273-20-03, left")  : 6.9,
        ("fast", "low",  "EEV 9273-20-03, left")  : 4.8,
        # EEV secondary amps
        ("slow", "low",  "EEV 9273-16-03, left")  : 3.5,
        ("slow", "high", "EEV 9273-16-03, left")  : 4.6,
        ("fast", "high", "EEV 9273-16-03, left")  : 5.9,
        ("fast", "low",  "EEV 9273-16-03, left")  : 4.5,
        ("slow", "low",  "EEV 9273-20-04, left")  : 3.9,
        ("slow", "high", "EEV 9273-20-04, left")  : 4.8,
        ("fast", "high", "EEV 9273-20-04, left")  : 11.0,
        ("fast", "low",  "EEV 9273-20-04, left")  : 4.8,
        ("slow", "low",  "EEV 9273-20-03, right") : 3.3,
        ("slow", "high", "EEV 9273-20-03, right") : 5.0,
        ("fast", "high", "EEV 9273-20-03, right") : 8.0,
        ("fast", "low",  "EEV 9273-20-03, right") : 4.7,
        # GMOS-S:
        # EEV best amps
        ("slow", "low",  "EEV 8056-20-03, left")  : 3.2,
        ("slow", "high", "EEV 8056-20-03, left")  : 4.2,
        ("fast", "high", "EEV 8056-20-03, left")  : 6.8,
        ("fast", "low",  "EEV 8056-20-03, left")  : 4.4,
        ("slow", "low",  "EEV 8194-19-04, left")  : 3.69,
        ("slow", "high", "EEV 8194-19-04, left")  : 4.69,
        ("fast", "high", "EEV 8194-19-04, left")  : 6.14,
        ("fast", "low",  "EEV 8194-19-04, left")  : 4.73,
        ("slow", "low",  "EEV 8261-07-04, right") : 3.31,
        ("slow", "high", "EEV 8261-07-04, right") : 4.45,
        ("fast", "high", "EEV 8261-07-04, right") : 6.20,
        ("fast", "low",  "EEV 8261-07-04, right") : 4.58,
        # EEV secondary amps
        ("slow", "low",  "EEV 8056-20-03, right") : 3.4,
        ("slow", "high", "EEV 8056-20-03, right") : 4.9,
        ("fast", "high", "EEV 8056-20-03, right") : 6.3,
        ("fast", "low",  "EEV 8056-20-03, right") : 4.3,
        ("slow", "low",  "EEV 8194-19-04, right") : 3.93,
        ("slow", "high", "EEV 8194-19-04, right") : 5.46,
        ("fast", "high", "EEV 8194-19-04, right") : 8.41,
        ("fast", "low",  "EEV 8194-19-04, right") : 5.25,
        ("slow", "low",  "EEV 8261-07-04, left")  : 3.65,
        ("slow", "high", "EEV 8261-07-04, left")  : 4.85,
        ("fast", "high", "EEV 8261-07-04, left")  : 6.92,
        ("fast", "low",  "EEV 8261-07-04, left")  : 4.99,
        # new GMOS-S EEV CCD: Best/Secondary
        ("slow", "low",  "EEV 2037-06-03, left")  : 3.97,
        ("slow", "high", "EEV 2037-06-03, left")  : 5.13,
        ("fast", "high", "EEV 2037-06-03, left")  : 6.33,
        ("fast", "low",  "EEV 2037-06-03, left")  : 4.94,
        ("slow", "low",  "EEV 2037-06-03, right") : 4.28,
        ("slow", "high", "EEV 2037-06-03, right") : 5.48,
        ("fast", "high", "EEV 2037-06-03, right") : 6.28,
        ("fast", "low",  "EEV 2037-06-03, right") : 4.81
    }

gmosampsBias = {
    # Database of GMOS CCD amplifier BIAS properties
    # after 2017-02-24
    # Columns below are given as:
    # READOUT, GAINSTATE, AMPNAME : BIAS
    # GMOS-N: Hamamatsu CCDs
    # AMPS 4-1 (CCD1)
    ("slow", "low", "BI13-20-4k-1, 1"): 446.,
    ("slow", "high", "BI13-20-4k-1, 1"): 0.,
    ("fast", "high", "BI13-20-4k-1, 1"): 501.,
    ("fast", "low", "BI13-20-4k-1, 1"): 501.,
    ("slow", "low", "BI13-20-4k-1, 2"): 448.,
    ("slow", "high", "BI13-20-4k-1, 2"): 0.,
    ("fast", "high", "BI13-20-4k-1, 2"): 502.,
    ("fast", "low", "BI13-20-4k-1, 2"): 499.,
    ("slow", "low", "BI13-20-4k-1, 3"): 456.,
    ("slow", "high", "BI13-20-4k-1, 3"): 0.,
    ("fast", "high", "BI13-20-4k-1, 3"): 502.,
    ("fast", "low", "BI13-20-4k-1, 3"): 501.,
    ("slow", "low", "BI13-20-4k-1, 4"): 456.,
    ("slow", "high", "BI13-20-4k-1, 4"): 0.,
    ("fast", "high", "BI13-20-4k-1, 4"): 500.,
    ("fast", "low", "BI13-20-4k-1, 4"): 499.,
    # AMPS 8-5 (CCD2)
    ("slow", "low", "BI12-09-4k-2, 1"): 448.,
    ("slow", "high", "BI12-09-4k-2, 1"): 0.,
    ("fast", "high", "BI12-09-4k-2, 1"): 500.,
    ("fast", "low", "BI12-09-4k-2, 1"): 501.,
    ("slow", "low", "BI12-09-4k-2, 2"): 451.,
    ("slow", "high", "BI12-09-4k-2, 2"): 0.,
    ("fast", "high", "BI12-09-4k-2, 2"): 500.,
    ("fast", "low", "BI12-09-4k-2, 2"): 500.,
    ("slow", "low", "BI12-09-4k-2, 3"): 452.,
    ("slow", "high", "BI12-09-4k-2, 3"): 0.,
    ("fast", "high", "BI12-09-4k-2, 3"): 500.,
    ("fast", "low", "BI12-09-4k-2, 3"): 501.,
    ("slow", "low", "BI12-09-4k-2, 4"): 439.,
    ("slow", "high", "BI12-09-4k-2, 4"): 0.,
    ("fast", "high", "BI12-09-4k-2, 4"): 500.,
    ("fast", "low", "BI12-09-4k-2, 4"): 499.,
    # AMPS 12-9 (CCD3)
    ("slow", "low", "BI13-18-4k-2, 1"): 456.,
    ("slow", "high", "BI13-18-4k-2, 1"): 0.,
    ("fast", "high", "BI13-18-4k-2, 1"): 501.,
    ("fast", "low", "BI13-18-4k-2, 1"): 499.,
    ("slow", "low", "BI13-18-4k-2, 2"): 456.,
    ("slow", "high", "BI13-18-4k-2, 2"): 0.,
    ("fast", "high", "BI13-18-4k-2, 2"): 502.,
    ("fast", "low", "BI13-18-4k-2, 2"): 499.,
    ("slow", "low", "BI13-18-4k-2, 3"): 456.,
    ("slow", "high", "BI13-18-4k-2, 3"): 0.,
    ("fast", "high", "BI13-18-4k-2, 3"): 501.,
    ("fast", "low", "BI13-18-4k-2, 3"): 501.,
    ("slow", "low", "BI13-18-4k-2, 4"): 441.,
    ("slow", "high", "BI13-18-4k-2, 4"): 0.,
    ("fast", "high", "BI13-18-4k-2, 4"): 501.,
    ("fast", "low", "BI13-18-4k-2, 4"): 500.,

    # GMOS-S New Hamamatsu CCDs + swap (2023)
    # Values from German Gimeno on Jan 22, 2024
    # Valid from 20231214

    #  AMPS 4-1 (CCD1)
    ("slow", "low",  "BI11-41-4k-2, 1"): 331.,
    ("slow", "high", "BI11-41-4k-2, 1"): 0.,
    ("fast", "high", "BI11-41-4k-2, 1"): 2691.,
    ("fast", "low",  "BI11-41-4k-2, 1"): 547.,
    ("slow", "low",  "BI11-41-4k-2, 2"): 312.,
    ("slow", "high", "BI11-41-4k-2, 2"): 0.,
    ("fast", "high", "BI11-41-4k-2, 2"): 2821.,
    ("fast", "low",  "BI11-41-4k-2, 2"): 910.,
    ("slow", "low",  "BI11-41-4k-2, 3"): 360.,
    ("slow", "high", "BI11-41-4k-2, 3"): 0.,
    ("fast", "high", "BI11-41-4k-2, 3"): 2945.,
    ("fast", "low",  "BI11-41-4k-2, 3"): 796.,
    ("slow", "low",  "BI11-41-4k-2, 4"): 512.,
    ("slow", "high", "BI11-41-4k-2, 4"): 0.,
    ("fast", "high", "BI11-41-4k-2, 4"): 2976.,
    ("fast", "low",  "BI11-41-4k-2, 4"): 1033.,
    # AMPS 8-5 (CCD2)
    ("slow", "low",  "BI13-19-4k-3, 1"): 776.,
    ("slow", "high", "BI13-19-4k-3, 1"): 0.,
    ("fast", "high", "BI13-19-4k-3, 1"): 3451.,
    ("fast", "low",  "BI13-19-4k-3, 1"): 3065.,
    ("slow", "low",  "BI13-19-4k-3, 2"): 108.,
    ("slow", "high", "BI13-19-4k-3, 2"): 0.,
    ("fast", "high", "BI13-19-4k-3, 2"): 3193.,
    ("fast", "low",  "BI13-19-4k-3, 2"): 2278.,
    ("slow", "low",  "BI13-19-4k-3, 3"): 158.,
    ("slow", "high", "BI13-19-4k-3, 3"): 0.,
    ("fast", "high", "BI13-19-4k-3, 3"): 3020.,
    ("fast", "low",  "BI13-19-4k-3, 3"): 1856.,
    ("slow", "low",  "BI13-19-4k-3, 4"): 1073.,
    ("slow", "high", "BI13-19-4k-3, 4"): 0.,
    ("fast", "high", "BI13-19-4k-3, 4"): 3584.,
    ("fast", "low",  "BI13-19-4k-3, 4"): 3260.,
    # AMPS 12-9 (CCD3)
    ("slow", "low",  "BI12-34-4k-1, 1"): 514.,
    ("slow", "high", "BI12-34-4k-1, 1"): 0.,
    ("fast", "high", "BI12-34-4k-1, 1"): 3046.,
    ("fast", "low",  "BI12-34-4k-1, 1"): 1890.,
    ("slow", "low",  "BI12-34-4k-1, 2"): 497.,
    ("slow", "high", "BI12-34-4k-1, 2"): 0.,
    ("fast", "high", "BI12-34-4k-1, 2"): 3037.,
    ("fast", "low",  "BI12-34-4k-1, 2"): 1324.,
    ("slow", "low",  "BI12-34-4k-1, 3"): 525.,
    ("slow", "high", "BI12-34-4k-1, 3"): 0.,
    ("fast", "high", "BI12-34-4k-1, 3"): 2963.,
    ("fast", "low",  "BI12-34-4k-1, 3"): 1449.,
    ("slow", "low",  "BI12-34-4k-1, 4"): 507.,
    ("slow", "high", "BI12-34-4k-1, 4"): 0.,
    ("fast", "high", "BI12-34-4k-1, 4"): 3033.,
    ("fast", "low",  "BI12-34-4k-1, 4"): 920.
}

gmosampsBiasBefore20231214 = {
    # Database of GMOS CCD amplifier BIAS properties
    # after 2017-02-24
    # Columns below are given as:
    # READOUT, GAINSTATE, AMPNAME : BIAS
    # GMOS-N: Hamamatsu CCDs
    # AMPS 4-1 (CCD1)
    ("slow", "low", "BI13-20-4k-1, 1"): 446.,
    ("slow", "high", "BI13-20-4k-1, 1"): 0.,
    ("fast", "high", "BI13-20-4k-1, 1"): 501.,
    ("fast", "low", "BI13-20-4k-1, 1"): 501.,
    ("slow", "low", "BI13-20-4k-1, 2"): 448.,
    ("slow", "high", "BI13-20-4k-1, 2"): 0.,
    ("fast", "high", "BI13-20-4k-1, 2"): 502.,
    ("fast", "low", "BI13-20-4k-1, 2"): 499.,
    ("slow", "low", "BI13-20-4k-1, 3"): 456.,
    ("slow", "high", "BI13-20-4k-1, 3"): 0.,
    ("fast", "high", "BI13-20-4k-1, 3"): 502.,
    ("fast", "low", "BI13-20-4k-1, 3"): 501.,
    ("slow", "low", "BI13-20-4k-1, 4"): 456.,
    ("slow", "high", "BI13-20-4k-1, 4"): 0.,
    ("fast", "high", "BI13-20-4k-1, 4"): 500.,
    ("fast", "low", "BI13-20-4k-1, 4"): 499.,
    # AMPS 8-5 (CCD2)
    ("slow", "low", "BI12-09-4k-2, 1"): 448.,
    ("slow", "high", "BI12-09-4k-2, 1"): 0.,
    ("fast", "high", "BI12-09-4k-2, 1"): 500.,
    ("fast", "low", "BI12-09-4k-2, 1"): 501.,
    ("slow", "low", "BI12-09-4k-2, 2"): 451.,
    ("slow", "high", "BI12-09-4k-2, 2"): 0.,
    ("fast", "high", "BI12-09-4k-2, 2"): 500.,
    ("fast", "low", "BI12-09-4k-2, 2"): 500.,
    ("slow", "low", "BI12-09-4k-2, 3"): 452.,
    ("slow", "high", "BI12-09-4k-2, 3"): 0.,
    ("fast", "high", "BI12-09-4k-2, 3"): 500.,
    ("fast", "low", "BI12-09-4k-2, 3"): 501.,
    ("slow", "low", "BI12-09-4k-2, 4"): 439.,
    ("slow", "high", "BI12-09-4k-2, 4"): 0.,
    ("fast", "high", "BI12-09-4k-2, 4"): 500.,
    ("fast", "low", "BI12-09-4k-2, 4"): 499.,
    # AMPS 12-9 (CCD3)
    ("slow", "low", "BI13-18-4k-2, 1"): 456.,
    ("slow", "high", "BI13-18-4k-2, 1"): 0.,
    ("fast", "high", "BI13-18-4k-2, 1"): 501.,
    ("fast", "low", "BI13-18-4k-2, 1"): 499.,
    ("slow", "low", "BI13-18-4k-2, 2"): 456.,
    ("slow", "high", "BI13-18-4k-2, 2"): 0.,
    ("fast", "high", "BI13-18-4k-2, 2"): 502.,
    ("fast", "low", "BI13-18-4k-2, 2"): 499.,
    ("slow", "low", "BI13-18-4k-2, 3"): 456.,
    ("slow", "high", "BI13-18-4k-2, 3"): 0.,
    ("fast", "high", "BI13-18-4k-2, 3"): 501.,
    ("fast", "low", "BI13-18-4k-2, 3"): 501.,
    ("slow", "low", "BI13-18-4k-2, 4"): 441.,
    ("slow", "high", "BI13-18-4k-2, 4"): 0.,
    ("fast", "high", "BI13-18-4k-2, 4"): 501.,
    ("fast", "low", "BI13-18-4k-2, 4"): 500.,
    # GMOS-S Hamamatsu CCDs (new E video boards, 2015)
    #  AMPS 4-1 (CCD1)
    ("slow", "low",  "BI5-36-4k-2, 1"): 500.,
    ("slow", "high", "BI5-36-4k-2, 1"): 0.,
    ("fast", "high", "BI5-36-4k-2, 1"): 2811.,
    ("fast", "low",  "BI5-36-4k-2, 1"): 819.,
    ("slow", "low",  "BI5-36-4k-2, 2"): 897.,
    ("slow", "high", "BI5-36-4k-2, 2"): 0.,
    ("fast", "high", "BI5-36-4k-2, 2"): 2854.,
    ("fast", "low",  "BI5-36-4k-2, 2"): 1040.,
    ("slow", "low",  "BI5-36-4k-2, 3"): 633.,
    ("slow", "high", "BI5-36-4k-2, 3"): 0.,
    ("fast", "high", "BI5-36-4k-2, 3"): 2932.,
    ("fast", "low",  "BI5-36-4k-2, 3"): 856.,
    ("slow", "low",  "BI5-36-4k-2, 4"): 730.,
    ("slow", "high", "BI5-36-4k-2, 4"): 0.,
    ("fast", "high", "BI5-36-4k-2, 4"): 2909.,
    ("fast", "low",  "BI5-36-4k-2, 4"): 841.,
    # AMPS 8-5 (CCD2)
    ("slow", "low",  "BI11-33-4k-1, 1"): 990.,
    ("slow", "high", "BI11-33-4k-1, 1"): 0.,
    ("fast", "high", "BI11-33-4k-1, 1"): 3334.,
    ("fast", "low",  "BI11-33-4k-1, 1"): 2682.,
    ("slow", "low",  "BI11-33-4k-1, 2"): 760.,
    ("slow", "high", "BI11-33-4k-1, 2"): 0.,
    ("fast", "high", "BI11-33-4k-1, 2"): 3272.,
    ("fast", "low",  "BI11-33-4k-1, 2"): 2551.,
    ("slow", "low",  "BI11-33-4k-1, 3"): 495.,
    ("slow", "high", "BI11-33-4k-1, 3"): 0.,
    ("fast", "high", "BI11-33-4k-1, 3"): 3091.,
    ("fast", "low",  "BI11-33-4k-1, 3"): 2114.,
    ("slow", "low",  "BI11-33-4k-1, 4"): 905.,
    ("slow", "high", "BI11-33-4k-1, 4"): 0.,
    ("fast", "high", "BI11-33-4k-1, 4"): 3252.,
    ("fast", "low",  "BI11-33-4k-1, 4"): 2172.,
    # AMPS 12-9 (CCD3)
    ("slow", "low",  "BI12-34-4k-1, 1"): 795.,
    ("slow", "high", "BI12-34-4k-1, 1"): 0.,
    ("fast", "high", "BI12-34-4k-1, 1"): 3078.,
    ("fast", "low",  "BI12-34-4k-1, 1"): 1856.,
    ("slow", "low",  "BI12-34-4k-1, 2"): 470.,
    ("slow", "high", "BI12-34-4k-1, 2"): 0.,
    ("fast", "high", "BI12-34-4k-1, 2"): 3078.,
    ("fast", "low",  "BI12-34-4k-1, 2"): 1322.,
    ("slow", "low",  "BI12-34-4k-1, 3"): 525.,
    ("slow", "high", "BI12-34-4k-1, 3"): 0.,
    ("fast", "high", "BI12-34-4k-1, 3"): 3003.,
    ("fast", "low",  "BI12-34-4k-1, 3"): 1445.,
    ("slow", "low",  "BI12-34-4k-1, 4"): 230.,
    ("slow", "high", "BI12-34-4k-1, 4"): 0.,
    ("fast", "high", "BI12-34-4k-1, 4"): 3058.,
    ("fast", "low",  "BI12-34-4k-1, 4"): 893.
}

gmosampsBiasBefore20170224 = {
        # Database of GMOS CCD amplifier BIAS properties
        # after 2015-08-26 and before 2017-02-24
        #  (This is when the new GMOS-N Hamamatsu CCDs were installed.
        # Columns below are given as:
        # READOUT, GAINSTATE, AMPNAME : BIAS
        # GMOS-N:
        # e2vDD CCDs best amps
        ("slow", "low",  "e2v 10031-23-05, right") : 1176.,
        ("slow", "high", "e2v 10031-23-05, right") : 889.,
        ("fast", "high", "e2v 10031-23-05, right") : 768.,
        ("fast", "low",  "e2v 10031-23-05, right") : 1036.,
        ("slow", "low",  "e2v 10031-01-03, right") : 909.,
        ("slow", "high", "e2v 10031-01-03, right") : 551.,
        ("fast", "high", "e2v 10031-01-03, right") : 429.,
        ("fast", "low",  "e2v 10031-01-03, right") : 757.,
        ("slow", "low",  "e2v 10031-18-04, left")  : 1188.,
        ("slow", "high", "e2v 10031-18-04, left")  : 886.,
        ("fast", "high", "e2v 10031-18-04, left")  : 787.,
        ("fast", "low",  "e2v 10031-18-04, left")  : 1099.,
        # e2vDD CCDs secondary amps
        ("slow", "low",  "e2v 10031-23-05, left")  : 1055.,
        ("slow", "high", "e2v 10031-23-05, left")  : 776.,
        ("fast", "high", "e2v 10031-23-05, left")  : 653.,
        ("fast", "low",  "e2v 10031-23-05, left")  : 917.,
        ("slow", "low",  "e2v 10031-01-03, left")  : 1099.,
        ("slow", "high", "e2v 10031-01-03, left")  : 695.,
        ("fast", "high", "e2v 10031-01-03, left")  : 560.,
        ("fast", "low",  "e2v 10031-01-03, left")  : 934.,
        ("slow", "low",  "e2v 10031-18-04, right") : 1105.,
        ("slow", "high", "e2v 10031-18-04, right") : 827.,
        ("fast", "high", "e2v 10031-18-04, right") : 727.,
        ("fast", "low",  "e2v 10031-18-04, right") : 997.,
        # GMOS-S Hamamatsu CCDs (new E video boards, 2015)
        # AMPS 4-1 (CCD1)
        ("slow", "low",  "BI5-36-4k-2, 1"): 500.,
        ("slow", "high", "BI5-36-4k-2, 1"): 0.,
        ("fast", "high", "BI5-36-4k-2, 1"): 2811.,
        ("fast", "low",  "BI5-36-4k-2, 1"): 819.,
        ("slow", "low",  "BI5-36-4k-2, 2"): 897.,
        ("slow", "high", "BI5-36-4k-2, 2"): 0.,
        ("fast", "high", "BI5-36-4k-2, 2"): 2854.,
        ("fast", "low",  "BI5-36-4k-2, 2"): 1040.,
        ("slow", "low",  "BI5-36-4k-2, 3"): 633.,
        ("slow", "high", "BI5-36-4k-2, 3"): 0.,
        ("fast", "high", "BI5-36-4k-2, 3"): 2932.,
        ("fast", "low",  "BI5-36-4k-2, 3"): 856.,
        ("slow", "low",  "BI5-36-4k-2, 4"): 730.,
        ("slow", "high", "BI5-36-4k-2, 4"): 0.,
        ("fast", "high", "BI5-36-4k-2, 4"): 2909.,
        ("fast", "low",  "BI5-36-4k-2, 4"): 841.,
        # AMPS 8-5 (CCD2)
        ("slow", "low",  "BI11-33-4k-1, 1"): 990.,
        ("slow", "high", "BI11-33-4k-1, 1"): 0.,
        ("fast", "high", "BI11-33-4k-1, 1"): 3334.,
        ("fast", "low",  "BI11-33-4k-1, 1"): 2682.,
        ("slow", "low",  "BI11-33-4k-1, 2"): 760.,
        ("slow", "high", "BI11-33-4k-1, 2"): 0.,
        ("fast", "high", "BI11-33-4k-1, 2"): 3272.,
        ("fast", "low",  "BI11-33-4k-1, 2"): 2551.,
        ("slow", "low",  "BI11-33-4k-1, 3"): 495.,
        ("slow", "high", "BI11-33-4k-1, 3"): 0.,
        ("fast", "high", "BI11-33-4k-1, 3"): 3091.,
        ("fast", "low",  "BI11-33-4k-1, 3"): 2114.,
        ("slow", "low",  "BI11-33-4k-1, 4"): 905.,
        ("slow", "high", "BI11-33-4k-1, 4"): 0.,
        ("fast", "high", "BI11-33-4k-1, 4"): 3252.,
        ("fast", "low",  "BI11-33-4k-1, 4"): 2172.,
        # AMPS 12-9 (CCD3)
        ("slow", "low",  "BI12-34-4k-1, 1"): 795.,
        ("slow", "high", "BI12-34-4k-1, 1"): 0.,
        ("fast", "high", "BI12-34-4k-1, 1"): 3078.,
        ("fast", "low",  "BI12-34-4k-1, 1"): 1856.,
        ("slow", "low",  "BI12-34-4k-1, 2"): 470.,
        ("slow", "high", "BI12-34-4k-1, 2"): 0.,
        ("fast", "high", "BI12-34-4k-1, 2"): 3078.,
        ("fast", "low",  "BI12-34-4k-1, 2"): 1322.,
        ("slow", "low",  "BI12-34-4k-1, 3"): 525.,
        ("slow", "high", "BI12-34-4k-1, 3"): 0.,
        ("fast", "high", "BI12-34-4k-1, 3"): 3003.,
        ("fast", "low",  "BI12-34-4k-1, 3"): 1445.,
        ("slow", "low",  "BI12-34-4k-1, 4"): 230.,
        ("slow", "high", "BI12-34-4k-1, 4"): 0.,
        ("fast", "high", "BI12-34-4k-1, 4"): 3058.,
        ("fast", "low",  "BI12-34-4k-1, 4"): 893.
    }


gmosampsBiasBefore20150826 = {
        # Database of GMOS CCD amplifier BIAS properties
        # after 2006-08-31
        # Columns below are given as:
        # READOUT, GAINSTATE, AMPNAME : BIAS
        # GMOS-N:
        # e2vDD CCDs best amps
        ("slow", "low",  "e2v 10031-23-05, right") : 1176.,
        ("slow", "high", "e2v 10031-23-05, right") : 889.,
        ("fast", "high", "e2v 10031-23-05, right") : 768.,
        ("fast", "low",  "e2v 10031-23-05, right") : 1036.,
        ("slow", "low",  "e2v 10031-01-03, right") : 909.,
        ("slow", "high", "e2v 10031-01-03, right") : 551.,
        ("fast", "high", "e2v 10031-01-03, right") : 429.,
        ("fast", "low",  "e2v 10031-01-03, right") : 757.,
        ("slow", "low",  "e2v 10031-18-04, left")  : 1188.,
        ("slow", "high", "e2v 10031-18-04, left")  : 886.,
        ("fast", "high", "e2v 10031-18-04, left")  : 787.,
        ("fast", "low",  "e2v 10031-18-04, left")  : 1099.,
        # e2vDD CCDs secondary amps
        ("slow", "low",  "e2v 10031-23-05, left")  : 1055.,
        ("slow", "high", "e2v 10031-23-05, left")  : 776.,
        ("fast", "high", "e2v 10031-23-05, left")  : 653.,
        ("fast", "low",  "e2v 10031-23-05, left")  : 917.,
        ("slow", "low",  "e2v 10031-01-03, left")  : 1099.,
        ("slow", "high", "e2v 10031-01-03, left")  : 695.,
        ("fast", "high", "e2v 10031-01-03, left")  : 560.,
        ("fast", "low",  "e2v 10031-01-03, left")  : 934.,
        ("slow", "low",  "e2v 10031-18-04, right") : 1105.,
        ("slow", "high", "e2v 10031-18-04, right") : 827.,
        ("fast", "high", "e2v 10031-18-04, right") : 727.,
        ("fast", "low",  "e2v 10031-18-04, right") : 997.,
        # EEV CCDs best amps
        ("slow", "low",  "EEV 9273-16-03, right") : 738.,
        ("slow", "high", "EEV 9273-16-03, right") : 680.,
        ("fast", "high", "EEV 9273-16-03, right") : 580.,
        ("fast", "low",  "EEV 9273-16-03, right") : 645.,
        ("slow", "low",  "EEV 9273-20-04, right") : 608.,
        ("slow", "high", "EEV 9273-20-04, right") : 561.,
        ("fast", "high", "EEV 9273-20-04, right") : 455.,
        ("fast", "low",  "EEV 9273-20-04, right") : 506.,
        ("slow", "low",  "EEV 9273-20-03, left")  : 370.,
        ("slow", "high", "EEV 9273-20-03, left")  : 295.,
        ("fast", "high", "EEV 9273-20-03, left")  : 224.,
        ("fast", "low",  "EEV 9273-20-03, left")  : 332.,
        # EEV CCDs secondary amps
        ("slow", "low",  "EEV 9273-16-03, left")  : 485.,
        ("slow", "high", "EEV 9273-16-03, left")  : 404.,
        ("fast", "high", "EEV 9273-16-03, left")  : 283.,
        ("fast", "low",  "EEV 9273-16-03, left")  : 361.,
        ("slow", "low",  "EEV 9273-20-04, left")  : 669.,
        ("slow", "high", "EEV 9273-20-04, left")  : 619.,
        ("fast", "high", "EEV 9273-20-04, left")  : 533.,
        ("fast", "low",  "EEV 9273-20-04, left")  : 593.,
        ("slow", "low",  "EEV 9273-20-03, right") : 737.,
        ("slow", "high", "EEV 9273-20-03, right") : 673.,
        ("fast", "high", "EEV 9273-20-03, right") : 570.,
        ("fast", "low",  "EEV 9273-20-03, right") : 635.,
        # GMOS-S:
        # EEV CCDs best amps
        ("slow", "low",  "EEV 8056-20-03, left")  : 602.,
        ("slow", "high", "EEV 8056-20-03, left")  : 547.,
        ("fast", "high", "EEV 8056-20-03, left")  : 443.,
        ("fast", "low",  "EEV 8056-20-03, left")  : 503.,
        ("slow", "low",  "EEV 8194-19-04, left")  : 667.,
        ("slow", "high", "EEV 8194-19-04, left")  : 571.,
        ("fast", "high", "EEV 8194-19-04, left")  : 449.,
        ("fast", "low",  "EEV 8194-19-04, left")  : 535.,
        ("slow", "low",  "EEV 8261-07-04, right") : 644.,
        ("slow", "high", "EEV 8261-07-04, right") : 536.,
        ("fast", "high", "EEV 8261-07-04, right") : 448.,
        ("fast", "low",  "EEV 8261-07-04, right") : 559.,
        # EEV CCDs secondary amps
        ("slow", "low",  "EEV 8056-20-03, right") : 579.,
        ("slow", "high", "EEV 8056-20-03, right") : 488.,
        ("fast", "high", "EEV 8056-20-03, right") : 370.,
        ("fast", "low",  "EEV 8056-20-03, right") : 443.,
        ("slow", "low",  "EEV 8194-19-04, right") : 674.,
        ("slow", "high", "EEV 8194-19-04, right") : 610.,
        ("fast", "high", "EEV 8194-19-04, right") : 494.,
        ("fast", "low",  "EEV 8194-19-04, right") : 571.,
        ("slow", "low",  "EEV 8261-07-04, left")  : 784.,
        ("slow", "high", "EEV 8261-07-04, left")  : 647.,
        ("fast", "high", "EEV 8261-07-04, left")  : 528.,
        ("fast", "low",  "EEV 8261-07-04, left")  : 663.,
        # new GMOS-S EEV CCD: Best/Secondary
        ("slow", "low",  "EEV 2037-06-03, left")  : 714.,
        ("slow", "high", "EEV 2037-06-03, left")  : 634.,
        ("fast", "high", "EEV 2037-06-03, left")  : 517.,
        ("fast", "low",  "EEV 2037-06-03, left")  : 583.,
        ("slow", "low",  "EEV 2037-06-03, right") : 572.,
        ("slow", "high", "EEV 2037-06-03, right") : 492.,
        ("fast", "high", "EEV 2037-06-03, right") : 377.,
        ("fast", "low",  "EEV 2037-06-03, right") : 448.,
        # GMOS-S Hamamatsu CCDs
        # AMPS 4-1 (CCD1)
        ("slow", "low",  "BI5-36-4k-2, 1"): 3218.,
        ("slow", "high", "BI5-36-4k-2, 1"): 0.000,
        ("fast", "high", "BI5-36-4k-2, 1"): 1932.,
        ("fast", "low",  "BI5-36-4k-2, 1"): 1829.,
        ("slow", "low",  "BI5-36-4k-2, 2"): 3359.,
        ("slow", "high", "BI5-36-4k-2, 2"): 0.000,
        ("fast", "high", "BI5-36-4k-2, 2"): 2288.,
        ("fast", "low",  "BI5-36-4k-2, 2"): 2430.,
        ("slow", "low",  "BI5-36-4k-2, 3"): 3270.,
        ("slow", "high", "BI5-36-4k-2, 3"): 0.000,
        ("fast", "high", "BI5-36-4k-2, 3"): 1760.,
        ("fast", "low",  "BI5-36-4k-2, 3"): 1901.,
        ("slow", "low",  "BI5-36-4k-2, 4"): 3316.,
        ("slow", "high", "BI5-36-4k-2, 4"): 0.000,
        ("fast", "high", "BI5-36-4k-2, 4"): 2098.,
        ("fast", "low",  "BI5-36-4k-2, 4"): 2494.,

        # AMPS 8-5 (CCD2)
        ("slow", "low",  "BI11-33-4k-1, 1"): 3025.,
        ("slow", "high", "BI11-33-4k-1, 1"): 0.000,
        ("fast", "high", "BI11-33-4k-1, 1"): 1890.,
        ("fast", "low",  "BI11-33-4k-1, 1"): 2113.,
        ("slow", "low",  "BI11-33-4k-1, 2"): 2942.,
        ("slow", "high", "BI11-33-4k-1, 2"): 0.000,
        ("fast", "high", "BI11-33-4k-1, 2"): 2106.,
        ("fast", "low",  "BI11-33-4k-1, 2"): 2067.,
        ("slow", "low",  "BI11-33-4k-1, 3"): 2948.,
        ("slow", "high", "BI11-33-4k-1, 3"): 0.000,
        ("fast", "high", "BI11-33-4k-1, 3"): 2217.,
        ("fast", "low",  "BI11-33-4k-1, 3"): 2342.,
        ("slow", "low",  "BI11-33-4k-1, 4"): 2985.,
        ("slow", "high", "BI11-33-4k-1, 4"): 0.000,
        ("fast", "high", "BI11-33-4k-1, 4"): 1947.,
        ("fast", "low",  "BI11-33-4k-1, 4"): 1808.,
        # AMPS 12-9 (CCD3)
        ("slow", "low",  "BI12-34-4k-1, 1"): 3162.,
        ("slow", "high", "BI12-34-4k-1, 1"): 0.000,
        ("fast", "high", "BI12-34-4k-1, 1"): 2467.,
        ("fast", "low",  "BI12-34-4k-1, 1"): 2745.,
        ("slow", "low",  "BI12-34-4k-1, 2"): 3113.,
        ("slow", "high", "BI12-34-4k-1, 2"): 0.000,
        ("fast", "high", "BI12-34-4k-1, 2"): 1962.,
        ("fast", "low",  "BI12-34-4k-1, 2"): 1823.,
        ("slow", "low",  "BI12-34-4k-1, 3"): 3071.,
        ("slow", "high", "BI12-34-4k-1, 3"): 0.000,
        ("fast", "high", "BI12-34-4k-1, 3"): 2142.,
        ("fast", "low",  "BI12-34-4k-1, 3"): 2201.,
        ("slow", "low",  "BI12-34-4k-1, 4"): 3084.,
        ("slow", "high", "BI12-34-4k-1, 4"): 0.000,
        ("fast", "high", "BI12-34-4k-1, 4"): 1617.,
        ("fast", "low",  "BI12-34-4k-1, 4"): 1263.,
        # AMPS 12-9 (CCD3)
        # Repeated entries for CCD3 without the last comma
        ("slow", "low",  "BI12-34-4k-1 1"): 3162.,
        ("slow", "high", "BI12-34-4k-1 1"): 0.000,
        ("fast", "high", "BI12-34-4k-1 1"): 2467.,
        ("fast", "low",  "BI12-34-4k-1 1"): 2745.,
        ("slow", "low",  "BI12-34-4k-1 2"): 3113.,
        ("slow", "high", "BI12-34-4k-1 2"): 0.000,
        ("fast", "high", "BI12-34-4k-1 2"): 1962.,
        ("fast", "low",  "BI12-34-4k-1 2"): 1823.,
        ("slow", "low",  "BI12-34-4k-1 3"): 3071.,
        ("slow", "high", "BI12-34-4k-1 3"): 0.000,
        ("fast", "high", "BI12-34-4k-1 3"): 2142.,
        ("fast", "low",  "BI12-34-4k-1 3"): 2201.,
        ("slow", "low",  "BI12-34-4k-1 4"): 3084.,
        ("slow", "high", "BI12-34-4k-1 4"): 0.000,
        ("fast", "high", "BI12-34-4k-1 4"): 1617.,
        ("fast", "low",  "BI12-34-4k-1 4"): 1263.
    }

gmosampsBiasBefore20060831 = {
        # Database of GMOS CCD amplifier GAIN properties
        # before 2006-08-31
        # Columns below are given as:
        # READOUT, GAINSTATE, AMPNAME : BIAS
        # GMOS-N
        # EEV CCDs best amps
        ("slow", "low",  "EEV 9273-16-03, right") : 738.,
        ("slow", "high", "EEV 9273-16-03, right") : 680.,
        ("fast", "high", "EEV 9273-16-03, right") : 580.,
        ("fast", "low",  "EEV 9273-16-03, right") : 645.,
        ("slow", "low",  "EEV 9273-20-04, right") : 608.,
        ("slow", "high", "EEV 9273-20-04, right") : 561.,
        ("fast", "high", "EEV 9273-20-04, right") : 455.,
        ("fast", "low",  "EEV 9273-20-04, right") : 506.,
        ("slow", "low",  "EEV 9273-20-03, left")  : 370.,
        ("slow", "high", "EEV 9273-20-03, left")  : 295.,
        ("fast", "high", "EEV 9273-20-03, left")  : 224.,
        ("fast", "low",  "EEV 9273-20-03, left")  : 332.,
        # EEV CCDs secondary amps
        ("slow", "low",  "EEV 9273-16-03, left")  : 485.,
        ("slow", "high", "EEV 9273-16-03, left")  : 404.,
        ("fast", "high", "EEV 9273-16-03, left")  : 283.,
        ("fast", "low",  "EEV 9273-16-03, left")  : 361.,
        ("slow", "low",  "EEV 9273-20-04, left")  : 669.,
        ("slow", "high", "EEV 9273-20-04, left")  : 619.,
        ("fast", "high", "EEV 9273-20-04, left")  : 533.,
        ("fast", "low",  "EEV 9273-20-04, left")  : 593.,
        ("slow", "low",  "EEV 9273-20-03, right") : 737.,
        ("slow", "high", "EEV 9273-20-03, right") : 673.,
        ("fast", "high", "EEV 9273-20-03, right") : 570.,
        ("fast", "low",  "EEV 9273-20-03, right") : 635.,
        # GMOS-S: New Gain/RN/Bias values (2006oct25) for
        # EEV CCDs best amps
        ("slow", "low",  "EEV 8056-20-03, left")  : 602.,
        ("slow", "high", "EEV 8056-20-03, left")  : 547.,
        ("fast", "high", "EEV 8056-20-03, left")  : 443.,
        ("fast", "low",  "EEV 8056-20-03, left")  : 503.,
        ("slow", "low",  "EEV 8194-19-04, left")  : 579.,
        ("slow", "high", "EEV 8194-19-04, left")  : 497.,
        ("fast", "high", "EEV 8194-19-04, left")  : 392.,
        ("fast", "low",  "EEV 8194-19-04, left")  : 486.,
        ("slow", "low",  "EEV 8261-07-04, right") : 627.,
        ("slow", "high", "EEV 8261-07-04, right") : 532.,
        ("fast", "high", "EEV 8261-07-04, right") : 784.,
        ("fast", "low",  "EEV 8261-07-04, right") : 622.,
        # EEV CCDs secondary amps
        ("slow", "low",  "EEV 8056-20-03, right") : 579.,
        ("slow", "high", "EEV 8056-20-03, right") : 488.,
        ("fast", "high", "EEV 8056-20-03, right") : 370.,
        ("fast", "low",  "EEV 8056-20-03, right") : 443.,
        ("slow", "low",  "EEV 8194-19-04, right") : 614.,
        ("slow", "high", "EEV 8194-19-04, right") : 552.,
        ("fast", "high", "EEV 8194-19-04, right") : 459.,
        ("fast", "low",  "EEV 8194-19-04, right") : 533.,
        ("slow", "low",  "EEV 8261-07-04, left")  : 749.,
        ("slow", "high", "EEV 8261-07-04, left")  : 634.,
        ("fast", "high", "EEV 8261-07-04, left")  : 536.,
        ("fast", "low",  "EEV 8261-07-04, left")  : 663.,
        # new GMOS-S EEV CCD: Best/Secondary
        ("slow", "low",  "EEV 2037-06-03, left")  : 609.,
        ("slow", "high", "EEV 2037-06-03, left")  : 548.,
        ("fast", "high", "EEV 2037-06-03, left")  : 456.,
        ("fast", "low",  "EEV 2037-06-03, left")  : 534.,
        ("slow", "low",  "EEV 2037-06-03, right") : 566.,
        ("slow", "high", "EEV 2037-06-03, right") : 480.,
        ("fast", "high", "EEV 2037-06-03, right") : 369.,
        ("fast", "low",  "EEV 2037-06-03, right") : 439.
    }

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
    # ("detector ID", "filter"): zeropoint
    #
    # GMOS-N original EEV detectors
    # Values from http://www.gemini.edu/sciops/instruments/gmos/calibration?q=node/10445  20111021
    ('EEV 9273-20-04', "u"): 25.47,
    ('EEV 9273-16-03', "u"): 25.47,
    ('EEV 9273-20-03', "u"): 25.47,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "u"): 25.47,
    ('EEV 9273-20-04', "g"): 27.95,
    ('EEV 9273-16-03', "g"): 27.95,
    ('EEV 9273-20-03', "g"): 27.95,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "g"): 27.95,
    ('EEV 9273-20-04', "r"): 28.20,
    ('EEV 9273-16-03', "r"): 28.20,
    ('EEV 9273-20-03', "r"): 28.20,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "r"): 28.20,
    ('EEV 9273-20-04', "i"): 27.94,
    ('EEV 9273-16-03', "i"): 27.94,
    ('EEV 9273-20-03', "i"): 27.94,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "i"): 27.94,
    ('EEV 9273-20-04', "z"): 26.78,
    ('EEV 9273-16-03', "z"): 26.78,
    ('EEV 9273-20-03', "z"): 26.78,
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', "z"): 26.78,
    #
    # GMOS-N New (Nov 2011) deep depletion E2V CCDs.
    # Bogus values made up by PH for u-band
    # New values from post M2 coating 20150925, data from
    #   Oct 8, 2015 to Nov 1, 2015 - MPohlen
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "u"): 10.00,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "u"): 10.00,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "u"): 10.00,
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "u"): 10.00,
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "g"): 28.248,  # 0.048
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "g"): 28.248,  # 0.048
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "g"): 28.248,  # 0.048
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "g"): 28.248,  # 0.048
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "r"): 28.334,  # 0.066
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "r"): 28.334,  # 0.066
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "r"): 28.334,  # 0.066
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "r"): 28.334,  # 0.066
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "i"): 28.399,  # 0.088
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "i"): 28.399,  # 0.088
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "i"): 28.399,  # 0.088
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "i"): 28.399,  # 0.088
    ('e2v DD ML2AR CCD42-90-1-F43 10031-23-05', "z"): 27.656,  # 0.076
    ('e2v DD ML2AR CCD42-90-1-F43 10031-01-03', "z"): 27.656,  # 0.076
    ('e2v DD ML2AR CCD42-90-1-F43 10031-18-04', "z"): 27.656,  # 0.076
    ('e2v 10031-23-05,10031-01-03,10031-18-04', "z"): 27.656,  # 0.076

    # GMOS-N Hamamatsu CCDs
    # Values provided by ?? on ??
    #
    # CCDr = BI13-20-4k-1
    # CCDg = BI12-09-4k-2
    # CCDb = BI13-18-4k-2
    ('BI13-20-4k-1', 'g'): 28.09,  # 0.000,
    ('BI12-09-4k-2', 'g'): 28.09,  # 0.00,
    ('BI13-18-4k-2', 'g'): 28.09,  # 0.000,
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2', 'g'): 28.09,  # 0.00
    ('BI13-20-4k-1', 'r'): 28.35,  # 0.000,
    ('BI12-09-4k-2', 'r'): 28.35,  # 0.000,
    ('BI13-18-4k-2', 'r'): 28.35,  # 0.000,
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2', 'r'): 28.35,  # 0.00
    ('BI13-20-4k-1', 'i'): 28.42,  # 0.000,
    ('BI12-09-4k-2', 'i'): 28.42,  # 0.000,
    ('BI13-18-4k-2', 'i'): 28.42,  # 0.000,
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2', 'i'): 28.42,  # 0.00
    ('BI13-20-4k-1', 'z'): 28.19,  # 0.000,
    ('BI12-09-4k-2', 'z'): 28.19,  # 0.000,
    ('BI13-18-4k-2', 'z'): 28.19,  # 0.000,
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2', 'z'): 28.19,  # 0.00,
    ('BI13-20-4k-1', 'OVI'): 23.8,  # 0.000,
    ('BI12-09-4k-2', 'OVI'): 23.8,  # 0.000,
    ('BI13-18-4k-2', 'OVI'): 23.8,  # 0.000,
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2', 'OVI'): 23.8,  # 0.00,
    ('BI13-20-4k-1', 'OVIC'): 23.8,  # 0.000,
    ('BI12-09-4k-2', 'OVIC'): 23.8,  # 0.000,
    ('BI13-18-4k-2', 'OVIC'): 23.8,  # 0.000,
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2', 'OVIC'): 23.8,  # 0.00,

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
    # Values on the website based on data taken between Jun 2014 and Dec 2015.
    # https://www.gemini.edu/instrumentation/gmos/calibrations#PhotStand
    #
    # Numbers commented out are the color terms. Zeropoint uncertainties are
    # listed as +/-0.011 mag and color term uncertainties +/- 0.023 mag.
    #
    # The combined zeropoint is simply the CCD2 zeropoint, since this will
    # apply to post-flatfielded data, and the flatfield normalization is
    # based solely on the CCD2 level.
    #
    # CCDr = BI5-36-4k-2
    # CCDg = BI11-33-4k-1
    # CCDb = BI12-34-4k-1
    ('BI5-36-4k-2', 'u'):  24.138,  # 0.025,
    ('BI11-33-4k-1', 'u'): 24.103,  # 0.045,
    ('BI12-34-4k-1', 'u'): 24.165,  # 0.015,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'u'): 24.10,
    ('BI5-36-4k-2', 'g'):  28.002,  # 0.044,
    ('BI11-33-4k-1', 'g'): 27.985,  # 0.059,
    ('BI12-34-4k-1', 'g'): 27.984,  # 0.041,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'g'): 27.99,
    ('BI5-36-4k-2', 'r'):  28.244,  # -0.022,
    ('BI11-33-4k-1', 'r'): 28.227,  # 0.004,
    ('BI12-34-4k-1', 'r'): 28.235,  # 0.000,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'r'): 28.23,
    ('BI5-36-4k-2', 'i'):  28.229,  # -0.020,
    ('BI11-33-4k-1', 'i'): 28.209,  # -0.012,
    ('BI12-34-4k-1', 'i'): 28.238,  # -0.024,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'i'): 28.21,
    ('BI5-36-4k-2', 'z'):  28.035,  # -0.068,
    ('BI11-33-4k-1', 'z'): 28.020,  # -0.005,
    ('BI12-34-4k-1', 'z'): 28.018,  # -0.055,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'z'): 28.02,
    ('BI5-36-4k-2', 'OVI'): 23.1,
    ('BI11-33-4k-1', 'OVI'): 23.1,
    ('BI12-34-4k-1', 'OVI'): 23.1,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'OVI'): 23.1,
    ('BI5-36-4k-2', 'OVIC'): 23.1,
    ('BI11-33-4k-1', 'OVIC'): 23.1,
    ('BI12-34-4k-1', 'OVIC'): 23.1,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'OVIC'): 23.1,
}

# Dictionary of the pixel scale for all GMOS sites and detectors
# All values in arcseconds per pixel. Keyed by instrument and DETTYPE
gmosPixelScales = {
    # GMOS-N:
    # EEV CCDs
    ("GMOS-N", "SDSU II CCD"): 0.0727,
    # e2v CCDs
    ("GMOS-N", "SDSU II e2v DD CCD42-90"): 0.07288,
    # Hamamatsu CCDs
    ("GMOS-N", "S10892-N"): 0.0807,

    # GMOS-S:
    # EEV CCDs
    ("GMOS-S", "SDSU II CCD"): 0.073,
    # Hamamatsu CCDs
    ("GMOS-S", "S10892"): 0.0800
}


# Associate ROI names with the standard ROIs sections
# The ROIs here are given in unbinned pixels, in Python section format,
# meaning (y1, y2, x1, x2) and 0-indexed.
# Format is: { "ROI Name" : [(list of), (rois for), (this ROI name)]}
gmosRoiSettings = {
    "Full Frame": [
        # EEV / e2vDD CCDs
        (0, 4608, 0, 6144),
        # Hamamatsu
        (0, 4224, 0, 6144),
    ],
    "CCD2": [
        # EEV / e2vDD CCDs
        (0, 4608, 2048, 4096),
        # Hamamatsu
        (0, 4224, 2048, 4096),
    ],
    "Central Spectrum": [
        # EEV / e2vDD CCDs (adjustment circa 2010)
        (1792, 2816, 0, 6144),
        (1791, 2815, 0, 6144),
        # Hamamatsu
        (1624, 2648, 0, 6144),
    ],
    "Central Stamp": [
        # EEV / e2vDD CCDs
        (2154, 2454, 2922, 3222),
        # Hamamatsu
        (1986, 2286, 2922, 3222),
        # GMOS-N Hamamatsu
        (1986, 2294, 2922, 3222),
    ],
}

# # lists the valid ROIs for each ROI name
# # The ROIs here are given in physical (ie unbinned) pixels, and are 1- based.
# # We provide a list of rois for each name as the definitions get changed once
# # once in a while and we don't change the name.
# ##M The addition of the Hamamatsu values may actually be a bit of a hack
# # Format is: { "ROI Name" : [(list of), (rois for), (this ROI name)]}
# gmosRoiSettings = {
#     "Full Frame" : [
#         # EEV / e2vDD CCDs
#         (1, 6144, 1, 4608),
#         # Hamamatsu GMOS-S  !!KL!! and GMOS-N?  Looks okay.
#         (1, 6144, 1, 4224)
#         ],
#     "CCD2" :[
#         # EEV / e2vDD CCDs
#         (2049, 4096, 1, 4608),
#         # Hamamatsu GMOS-S  !!KL!! and GMOS-N?  Looks okay.
#         (2049, 4096, 1, 4224)
#     ],
#     "Central Spectrum" : [
#         # This got adjusted by 1 pixel sometime circa 2010
#         (1, 6144, 1793, 2816),
#         (1, 6144, 1792, 2815),
#         # Hamamatsu GMOS-S  !!KL!! and GMOS-N?
#         (1, 6144, 1625, 2648)
#     ],
#     "Central Stamp" : [
#         # EEV and e2vDD CCDs
#         (2923, 3222, 2155, 2454),
#         # GMOS-S Hamamatsu CCDs  !!KL!! and GMOS-N?  Looks okay.
#         (2923, 3222, 1987, 2286)
#     ]
# }

# GMOS read modes. Dict for EEV previously defined in GMOS_Descriptors.py
# for descriptor read_mode().
# 'default' applies for both EEV and the super old e2v CCDs.
#
# It is unclear whether there is an 'Engineering' read mode for
# Hamamatsu CCDs. This mode is not defined in the requirements document,
# GMOS-S Hamamatsu CCD Upgrades, v2.0 - 15 December 2014 (M Simpson).
# The mode is left in here for JIC purposes as the final possible combination
# [ gain_setting, read_speed_setting ].
# 04-06-2015, kra
# CJS: Inverted key/value order of entries for AD2.0
read_mode_map = {
    "default": {
            ("low", "slow"): "Normal",
            ("high", "fast"): "Bright",
            ("low", "fast"): "Acquisition",
            ("high", "slow"): "Engineering",
    },
    "Hamamatsu": {
            ("low", "slow"): "Normal",
            ("high", "fast"): "Bright",
            ("low", "fast"): "Acquisition",
            ("high", "slow"): "Engineering",
    }
}

# Keyed by : ampname for e2vDD and Hamamatsu CCDs. The old EEV CCDs
# do not have a well-defined saturation level.
# These limits are in electrons, for 1x1 unbiased data.
# 2014-09-04: These values are for slow / low (science) mode - MS
gmosThresholds = {
    # GMOS-N e2v DD CCDs
    "GMOS + e2v DD CCD42-90": {
        "e2v 10031-23-05, left": 110900.0,
        "e2v 10031-23-05, right": 105100.0,
        "e2v 10031-01-03, left": 115500.0,
        "e2v 10031-01-03, right": 108900.0,
        "e2v 10031-18-04, left": 115700.0,
        "e2v 10031-18-04, right": 109200.0,
        },
    # GMOS-S Hamamatsu CCDs ##M FULL_WELL
    "GMOS + Hamamatsu": {
        "BI5-36-4k-2, 1": 108289.0,
        "BI5-36-4k-2, 2": 113419.0,
        "BI5-36-4k-2, 3": 111421.0,
        "BI5-36-4k-2, 4": 106558.0,
        "BI11-33-4k-1, 1": 109060.0,
        "BI11-33-4k-1, 2": 110814.0,
        "BI11-33-4k-1, 3": 110155.0,
        "BI11-33-4k-1, 4": 113986.0,
        "BI12-34-4k-1, 1": 99539.4,
        "BI12-34-4k-1, 2": 98963.6,
        "BI12-34-4k-1, 3": 98813.4,
        "BI12-34-4k-1, 4": 105683.0,
        },
    "GMOS + Hamamatsu_new": {
        "BI5-36-4k-2, 1": 120435.,
        "BI5-36-4k-2, 2": 121390.,
        "BI5-36-4k-2, 3": 121652.,
        "BI5-36-4k-2, 4": 118871.,
        "BI11-33-4k-1, 1": 121234.,
        "BI11-33-4k-1, 2": 119157.,
        "BI11-33-4k-1, 3": 125719.,
        "BI11-33-4k-1, 4": 123307.,
        "BI12-34-4k-1, 1": 106966.,
        "BI12-34-4k-1, 2": 114605.,
        "BI12-34-4k-1, 3": 112054.,
        "BI12-34-4k-1, 4": 118424.,
        },
    # GMOS-N Hamamatsu CCDs  (from JS on 2017-06-29)
    "GMOS-N + Hamamatsu": {
        "BI13-20-4k-1, 1": 129000,
        "BI13-20-4k-1, 2": 129000,
        "BI13-20-4k-1, 3": 129000,
        "BI13-20-4k-1, 4": 129000,
        "BI12-09-4k-2, 1": 124000,
        "BI12-09-4k-2, 2": 121000,
        "BI12-09-4k-2, 3": 123000,
        "BI12-09-4k-2, 4": 123000,
        "BI13-18-4k-2, 1": 123000,
        "BI13-18-4k-2, 2": 125000,
        "BI13-18-4k-2, 3": 124000,
        "BI13-18-4k-2, 4": 126000,
        },
    # GMOS-S Hamamatsu CCDs - 2023 upgrade (from GG on 2024-01-18)
    "GMOS + Ham-2": {
        "BI11-41-4k-2, 1": 131823.,
        "BI11-41-4k-2, 2": 130389.,
        "BI11-41-4k-2, 3": 130550.,
        "BI11-41-4k-2, 4": 125326.,
        "BI13-19-4k-3, 1": 115493.,
        "BI13-19-4k-3, 2": 113239.,
        "BI13-19-4k-3, 3": 115900.,
        "BI13-19-4k-3, 4": 114230.,
        "BI12-34-4k-1, 1": 105466.,
        "BI12-34-4k-1, 2": 112550.,
        "BI12-34-4k-1, 3": 111018.,
        "BI12-34-4k-1, 4": 116744.,
        },
}

# Column 1 from gmos$data/gratingeq.dat
# Could replace with polynomial coefficients
gratingeq = (1.629168086997, 1.6150523031634, 1.6004445587539, 1.5853493034291,
             1.56977113535, 1.5537147997774, 1.5371851876267, 1.5201873339779,
             1.5027264165419, 1.4848077540831, 1.4664368047999, 1.4476191646612,
             1.4283605657026, 1.4086668742798, 1.3885440892821, 1.3679983403045,
             1.3470358857814, 1.3256631110794, 1.3038865265527, 1.2817127655598,
             1.2591485824433, 1.2362008504717, 1.2128765597467, 1.1891828150729,
             1.1651268337947, 1.1407159435968, 1.115957580273, 1.0908592854604,
             1.0654287043429, 1.0396735833217, 1.0136017676563, 0.98722119907411,
             0.96053991335197, 0.93356603786793, 0.90630778912574, 0.87877347025202,
             0.85097146846701, 0.82291025252974, 0.79459837015842, 0.76604444542665,
             0.7372571761365, 0.70824533116903, 0.67901774781327, 0.6495833290742,
             0.61995104096088, 0.59012990975533, 0.56012901926295, 0.52995750804561,
             0.49962456663787, 0.46913943474751, 0.438511398441, 0.40774978731487,
             0.37686397165385, 0.34586335957656, 0.31475739416973, 0.28355555061173,
             0.25226733328635, 0.22090227288766, 0.18946992351687, 0.15797985977208,
             0.12644167383172, 0.094864972532733, 0.063259374444201, 0.03163450693745,
             3.2534573368714E-9)
