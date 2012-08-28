
# Mosaic task
#------------

# GMOS instruments (N,S) geometry configuration parameters for 
# gem_mosaic_function() in gemMosaicFunction.py

gap_dict_bin =   {(0,0):(0,0), (1,0):(36,0), (2,0):(36,0)}
gap_dict_unbin = {(0,0):(0,0), (1,0):(37,0), (2,0):(37,0)}
gaps = {   # gaps: (x_gap, y_gap)  One tuple per detector
    # Instr,  dettype,       detector, bin/unbinned

    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'): gap_dict_unbin,
    ('GMOS-S','SDSU II CCD','GMOS + Blue1',           'unbinned'): gap_dict_unbin,
    ('GMOS-N','SDSU II CCD','GMOS + e2v DD CCD42-90', 'unbinned'): gap_dict_unbin,
    ('GMOS-N','S10892-01',  'GMOS + S10892-01',       'unbinned'): gap_dict_unbin,
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'): gap_dict_unbin,

    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','binned'): gap_dict_bin,
    ('GMOS-S','SDSU II CCD','GMOS + Blue1',           'binned'): gap_dict_bin,
    ('GMOS-N','SDSU II CCD','GMOS + e2v DD CCD42-90', 'binned'): gap_dict_bin,
    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned'):           gap_dict_bin,
    ('GMOS-N','SDSU II CCD','GMOS + Red1','binned'):             gap_dict_bin,
    ('GMOS-N','S10892-01',  'GMOS + S10892-01',       'binned'): gap_dict_bin,    # Hamamatsu
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','binned'): gap_dict_bin,

}

gaps_tile = gaps
gaps_transform = gaps

shift = {                #  (x_shift, y_shift) for detectors (1,2,3).
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','unbinned'):[(-1.44,5.46),(0.,0.),(7.53,9.57)],  
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','binned'):  [(-2.44,5.46),(0.,0.),(7.53,9.57)],  
    #  
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'):[(-1.49,-0.22),(0.,0.),(4.31,2.04)], 
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','binned'):  [(-2.49,-0.22),(0.,0.),(5.31,2.04)], 

    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned') : [(-2.50,-1.58),(0.,0.),(3.8723,-1.86)],
    ('GMOS-N','SDSU II CCD','GMOS + Red1','binned') :   [(-3.50,-1.58),(0.,0.),(4.8723,-1.86)],

    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'): [(-2.7,-0.749),(0.,0.), (2.8014,2.0500)],
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','binned'): [(-3.7,-0.749),(0.,0.), (3.8014,2.0500)],
    # Hamamatsu
    ('GMOS-N','S10892-01','GMOS + S10892-01','unbinned'):[(-2.50,-1.58),(0.,0.),(3.8723,-1.86)],  
    ('GMOS-N','S10892-01','GMOS + S10892-01','binned'):  [(-3.50,-1.58),(0.,0.),(4.8723,-1.86)],  
}

rotation = {    # unbinned
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','unbinned'): (-0.01,  0.0, 0.02),
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned') : (0.011,0.0,0.012),

    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned'): (-0.004, 0.0,-0.046),
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'):  (-0.009, 0.0,-0.003),
    # Hamamatsu
    ('GMOS-N','S10892-01','GMOS + S10892-01','unbinned'): (-0.004, 0.0,-0.046),
}


chip_gaps = { 
    ('GMOS-S',
     'SDSU II CCD', 
     'GMOS + Blue1',
     'unbinned'):    [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],

    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'):
                [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],

    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned') :
                [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],

    ('GMOS-N','SDSU II CCD','GMOS + Red1',            'unbinned'): 
                [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'):
                [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],
    # Hamamatsu
    ('GMOS-N','S10892-01','GMOS + S10892-01',         'unbinned'):
               [(2046, 2086, 1, 4224), (4133, 4176, 1, 4224)],
}            

blocksize = {  
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'): (2048,4608),
    ('GMOS-S','SDSU II CCD','GMOS + Blue1',           'unbinned'): (2048,4608),
    ('GMOS-N','SDSU II CCD','GMOS + Red1',            'unbinned'): (2048,4608),
    ('GMOS-N','SDSU II CCD','GMOS + e2v DD CCD42-90', 'unbinned'): (2048,4608),
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'): (2048,4608),
    ('GMOS-N','S10892-01',  'GMOS + S10892-01',       'unbinned'): (2048,4224),
}

mosaic_grid = {
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','unbinned') :  (3,1),
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'):  (3,1),

    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned') : (3,1), 
    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned') :   (3,1), 
    ('GMOS-N','S10892-01','GMOS + S10892-01','unbinned') : (3,1), 
}

magnification = {
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','unbinned') : [(1.,1.),(1.,1.),(1.,1.)],
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'):  [(1.,1.),(1.,1.),(1.,1.)],

    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned') :[(1.,1.),(1.,1.),(1.,1.)],
    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned') :  [(1.,1.),(1.,1.),(1.,1.)], 
    ('GMOS-N','S10892-01','GMOS + S10892-01','unbinned') : [(1.,1.),(1.,1.),(1.,1.)] 
}

interpolator= {  # Values could be 'linear','nearest','spline2','spline3','spline4'
              'SCI':'linear','DQ':'linear', 'VAR':'linear','OBJMASK':'linear',
}
ref_block = { 'ref_block':(2,1)    # Reference detector (x,y) position in the mosaic grid
}
