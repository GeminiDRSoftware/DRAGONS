# GSAOI instrument geometry configuration parameters for 
# gem_mosaic_function() in gemMosaicFunction.py

# for new tileArrays
tile_gaps = {
    'GSAOI': (140, 145)
}

# gaps: (x_gap, y_gap)
gaps_tile = {
    # Gaps between the blocks in the mosaic grid, (0,0) (col,row) is the lower left.
    
    # dettype, bin/unbinned
    ('GSAOI', 'unbinned'): { (0, 0): (0, 0),     (1, 0): (153-8, 0),
                             (0, 1): (0, 148-8), (1, 1): (153-8, 148-8)
    }
}

# gaps: (x_gap, y_gap) applied when mosaicking corrected blocks.
gaps_transform = {
    # dettype, bin/unbinned
    ('GSAOI', 'unbinned'):{(0, 0): (0, 0), (1, 0): (111, 0),
                           (0, 1): (0, 92), (1, 1): (105, 90)
    }
}

blocksize = {
    ('GSAOI', 'unbinned'): (2048, 2048),
}

#  (x_shift, y_shift) for detectors (1,2,3,4).
shift = {
    ('GSAOI','unbinned'): [(0., 0.), (43.604018-4.0, -1.248663),
                           (0.029961, 41.102256-4.0), (43.420524-4.0, 41.722921-4.0)],
}

# unbinned, units are Degrees.
rotation = {
    ('GSAOI', 'unbinned'): (0.0, 0.033606, -0.582767, -0.769542),
}

# (x_mag,y_mag)
magnification = {
     ('GSAOI', 'unbinned'): [(1., 1.), (1., 1.), (1., 1.0), (1., 1.)],
}

# unbinned
chip_gaps = {
    ('GSAOI','unbinned'): [(2040, 2193, 1, 4219), (1, 4219, 2035, 2192)],
}            

mosaic_grid = {
    ('GSAOI', 'unbinned'): (2,2),
}

# Values could be 'linear','poly3','poly5','spline3'
interpolator= {
    'SCI': 'linear',
    'DQ': 'linear',
    'VAR': 'linear',
    'OBJMASK': 'linear',
}

# (0-based). Reference detector position, (x,y) in the mosaic grid.
ref_block = {'ref_block': (0, 0), }
