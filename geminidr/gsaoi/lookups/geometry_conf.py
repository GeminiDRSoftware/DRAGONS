# GSAOI geometry_conf.py module containing information
# for Transform-based tileArrays/mosaicDetectors

# for tileArrays(): key=detector_name(), value=(xgap, ygap) (unbinned pixels)
tile_gaps = {
    'GSAOI': (145, 140)
}

# for mosaicDetectors(): key=detector_name(), value=dict
# In this dict, each physical detector is keyed by the DETSEC coords (x,y) at its bottom-left
# and can contain entries for: "shift" (x,y) -- (0,0) if absent
#                               "rotation" (degrees counterclockwise) -- 0 if absent
#                               "magnification" -- 1.0 if absent
#                               "shape" (unbinned pixels) -- "default_shape" if absent
#
# The shifts are centre-to-centre differences between the detectors. Rotations are
# performed about the centre of each detector.
geometry = {
    'GSAOI': {'default_shape': (2048, 2048),
              (0, 0): {},
              (2048, 0): {'shift': (2198.604018, -1.248663),
                          'rotation': 0.033606},
              (0, 2048): {'shift': (0.029961, 2177.102256),
                          'rotation': -0.582767},
              (2048, 2048): {'shift': (2192.420524, 2175.722921),
                             'rotation': -0.769542}
              }
}
