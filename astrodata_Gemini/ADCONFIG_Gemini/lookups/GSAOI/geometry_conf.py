
# GSAOI instrument geometry configuration parameters for 
# gem_mosaic_function() in gemMosaicFunction.py

gaps = {    # gaps: (x_gap, y_gap)
    # Instr,  dettype,       detector, bin/unbinned
    
    #('GSAOI',None,'GSAOI','unbinned'): (139,139),
    ('GSAOI',None,'GSAOI','unbinned'): (153, 148) #Average (vertical,horizontal)
                                                 # gap between arrays (trimmed, tiling only

}

blocksize = {  
    ('GSAOI',None,'GSAOI','unbinned'): (2048,2048),
}

shift = {                #  (x_shift, y_shift) for detectors (1,2,3,4).
    #('GSAOI',None,'GSAOI','unbinned'):  [(43.604018,-1.248663),(0.,0.),
    ('GSAOI',None,'GSAOI','unbinned'):  [(0.,0.),(43.604018,-1.248663),
                   (0.029961, 41.102256), (43.420524, 41.722921)],
}

rotation = {    # unbinned
    ('GSAOI',None,'GSAOI','unbinned'): (0.0, -0.033606, 0.582767, 0.769542),
    #('GSAOI',None,'GSAOI','unbinned'): (-0.033606, 0.0, 0.582767, 0.769542),
}

magnification = {   # (x_mag,y_mag)
     ('GSAOI',None,'GSAOI','unbinned'): [(1.,1.), (1.0013,1.0013), (1.0052,1.0052), (1.0159,1.0159)],
     #('GSAOI',None,'GSAOI','unbinned'): [(1.0013,1.0013), (1.,1.), (1.0052,1.0052), (1.0159,1.0159)],
}


chip_gaps = {  # unbinned
    ('GSAOI',None,'GSAOI','unbinned'): [(2040, 2193, 1, 4219), (1, 4219, 2035, 2192)],
}            

mosaic_grid = {
    ('GSAOI',None,'GSAOI','unbinned'): (2,2),
}


interpolator= {  # Values could be 'linear','poly3','poly5','spline3'
              'SCI':'linear','DQ':'linear', 'VAR':'linear','OBJMASK':'linear',
}

ref_block = { 'ref_block':(1,1) ,    # reference detector position (x,y) in the mosaic grid.
#ref_block = { 'ref_block':(2,1) ,    # reference detector position (x,y) in the mosaic grid.
}
