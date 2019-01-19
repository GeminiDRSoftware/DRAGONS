# GSAOI instrument geometry configuration parameters for 
# gem_mosaic_function() in gemMosaicFunction.py

# for new tileArrays: (xgap, ygap)
tile_gaps = {
    'GSAOI': (145, 140)
}

# Shifts are (x, y) and include detector offsets
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
