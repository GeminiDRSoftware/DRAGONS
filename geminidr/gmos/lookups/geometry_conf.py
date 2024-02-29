# GMOS geometry_conf.py module containing information
# for Transform-based tileArrays/mosaicDetectors

# for tileArrays(): key=detector_name(), value=gap (unbinned pixels)
tile_gaps = {
    # GMOS-N
    'EEV9273-16-03EEV9273-20-04EEV9273-20-03': 37,
    'e2v 10031-23-05,10031-01-03,10031-18-04': 37,
    'BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2': 67,

    # GMOS-S
    'EEV2037-06-03EEV8194-19-04EEV8261-07-04': 37,
    'EEV8056-20-03EEV8194-19-04EEV8261-07-04': 37,
    'BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1': 61,
    'BI11-41-4k-2,BI13-19-4k-3,BI12-34-4k-1': 61,
}

# for mosaicDetectors(): key=detector_name(), value=dict
# In this dict, each CCD is keyed by the DETSEC coords (x,y) at its bottom-left
# and can contain entries for: "shift" (x,y) -- (0,0) if absent
#                               "rotation" (degrees counterclockwise) -- 0 if absent
#                               "magnification" -- 1.0 if absent
#                               "shape" (unbinned pixels) -- "default_shape" if absent
#
# The shifts are centre-to-centre differences between the CCDs. Rotations are
# performed about the centre of each CCD.
geometry = {
    # GMOS-N
    'EEV9273-16-03EEV9273-20-04EEV9273-20-03': {'default_shape': (2048, 4608),
                                                (0, 0): {'shift': (-2087.50,-1.58),
                                                         'rotation': -0.004},
                                                (2048, 0): {},
                                                (4096, 0): {'shift': (2088.8723, -1.86),
                                                            'rotation': -0.046}
                                                         },
    'e2v 10031-23-05,10031-01-03,10031-18-04': {'default_shape': (2048, 4608),
                                                (0, 0): {'shift': (-2087.7,-0.749),
                                                         'rotation': -0.009},
                                                (2048, 0): {},
                                                (4096, 0): {'shift': (2087.8014, 2.05),
                                                            'rotation': -0.003}
                                                },
    'BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2': {'default_shape': (2048, 4224),
                                               (0, 0): {'shift': (-2115.95, -0.21739),
                                                        'rotation': -0.004},
                                               (2048, 0): {},
                                               (4096, 0): {'shift': (2115.48, 0.1727),
                                                           'rotation': -0.00537}
                                               },

# GMOS-S
    'EEV8056-20-03EEV8194-19-04EEV8261-07-04': {'default_shape': (2048, 4608),
                                                (0, 0): {'shift': (-2086.44, 5.46),
                                                         'rotation': -0.01},
                                                (2048, 0): {},
                                                (4096, 0): {'shift': (2092.53, 9.57),
                                                            'rotation': 0.02}
                                                },
    'EEV2037-06-03EEV8194-19-04EEV8261-07-04': {'default_shape': (2048, 4608),
                                                (0, 0): {'shift': (-2086.49, -0.22),
                                                         'rotation': 0.011},
                                                (2048, 0): {},
                                                (4096, 0): {'shift': (2089.31, 2.04),
                                                            'rotation': 0.012}
                                                },
    'BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1': {'default_shape': (2048, 4224),
                                              (0, 0): {'shift': (-2110.2, 0.71),
                                                       'rotation': 0.},
                                              (2048, 0): {},
                                              (4096, 0): {'shift': (2109., -0.73),
                                                          'rotation': 0.}
                                              },
    'BI11-41-4k-2,BI13-19-4k-3,BI12-34-4k-1': {'default_shape': (2048, 4224),
                                              (0, 0): {'shift': (-2112.5, -1.2),
                                                       'rotation': 0.},
                                              (2048, 0): {},
                                              (4096, 0): {
                                                  'shift': (2110.8, 1.2),
                                                  'rotation': 0.}
                                              },
}
