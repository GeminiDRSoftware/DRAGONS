
from copy import deepcopy

# Mosaic task
#------------

# GMOS instruments (N,S) geometry configuration parameters for 
# gem_mosaic_function() in gemMosaicFunction.py

# Specifications below for GMOS-S Hamamatsu CCDs are taken from
# the DPSG Document:
#
#    GMOS-S Hamamatsu CCD Upgrades
#    DPSG Software Requirments
#    Mark Simpson
#    V2.0 -- 15 December 2014
#    Document ID: DPSG-SRS-101_GMOSSHamCCDReqs
#
# Added here 07 May 2015, kra.

gap_dict_bin =   {(0,0):(0,0), (1,0):(36,0), (2,0):(36,0)}
gap_dict_unbin = {(0,0):(0,0), (1,0):(37,0), (2,0):(37,0)}

hamgap_dict_bin = {(0,0):(0,0), (1,0):(60,0), (2,0):(60,0)}
hamgap_dict_unbin = {(0,0):(0,0), (1,0):(61,0), (2,0):(61,0)}

# NOTE: The gap sets used for GMOS-N Hamamatsu may not be correct, as the 
# keys below are simply using the gap sets for the old CCDs.
# - kra, 07-05-15

# gaps: (x_gap, y_gap)  One tuple per detector
#     Instr,      dettype,      detector,          bin/unbinned
# --------------------------------------------------------------
gaps = { 
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'): gap_dict_unbin,
    ('GMOS-S','SDSU II CCD','GMOS + Blue1',           'unbinned'): gap_dict_unbin,
    ('GMOS-N','SDSU II CCD','GMOS + e2v DD CCD42-90', 'unbinned'): gap_dict_unbin,
    ('GMOS-N','S10892-01',  'GMOS + S10892-01',       'unbinned'): gap_dict_unbin, # Hamamatsu ? May not be correct
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'): gap_dict_unbin,
    ('GMOS-S','S10892',    'GMOS + Hamamatsu',        'unbinned'): hamgap_dict_unbin,  # Hamamatsu

    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','binned'): gap_dict_bin,
    ('GMOS-S','SDSU II CCD','GMOS + Blue1',           'binned'): gap_dict_bin,
    ('GMOS-N','SDSU II CCD','GMOS + e2v DD CCD42-90', 'binned'): gap_dict_bin,
    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned'):           gap_dict_bin,
    ('GMOS-N','SDSU II CCD','GMOS + Red1','binned'):             gap_dict_bin,
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','binned'): gap_dict_bin,

    ('GMOS-N','S10892-01',  'GMOS + S10892-01',       'binned'): gap_dict_bin,    # Hamamatsu ? May not be correct
    ('GMOS-S','S10892',     'GMOS + Hamamatsu',       'binned'): hamgap_dict_bin  # Hamamatsu

}

gaps_tile = gaps
gaps_transform = deepcopy(gaps)

shift = {                #  (x_shift, y_shift) for detectors (1,2,3).
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','unbinned'):[(-1.44,5.46),(0.,0.),(7.53,9.57)],  
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','binned'):  [(-2.44,5.46),(0.,0.),(7.53,9.57)],  
    #  
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'):[(-1.49,-0.22),(0.,0.),(4.31,2.04)], 
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','binned'):  [(-2.49,-0.22),(0.,0.),(5.31,2.04)], 

    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned') : [(-2.50,-1.58),(0.,0.),(3.8723,-1.86)],
    ('GMOS-N','SDSU II CCD','GMOS + Red1','binned') :   [(-3.50,-1.58),(0.,0.),(4.8723,-1.86)],

    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'): [(-2.7,-0.749),(0.,0.),(2.8014,2.0500)],
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','binned'): [(-3.7,-0.749),(0.,0.),(3.8014,2.0500)],

    # Hamamatsu
    ('GMOS-N','S10892-01','GMOS + S10892-01','unbinned'):[(-2.50,-1.58),(0.,0.),(3.8723,-1.86)],
    ('GMOS-N','S10892-01','GMOS + S10892-01','binned'):  [(-3.50,-1.58),(0.,0.),(4.8723,-1.86)],

    ('GMOS-S','S10892','GMOS + Hamamatsu','unbinned'):[(-1.2,0.71),(0.,0.),(0.,-0.73)],
    ('GMOS-S','S10892','GMOS + Hamamatsu','binned'):  [(-2.4,0.0),(0.,0.),(0.,0.)]
}

rotation = {    # unbinned
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','unbinned'): (-0.01,  0.0, 0.02),
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned') : (0.011,0.0,0.012),

    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned'): (-0.004, 0.0,-0.046),
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'): (-0.009, 0.0,-0.003),

    # Hamamatsu
    ('GMOS-N','S10892-01','GMOS + S10892-01','unbinned'): (-0.004, 0.0,-0.046),
    ('GMOS-S','S10892','GMOS + Hamamatsu','unbinned'):    (0.,0.,0.)
}


chip_gaps = { 
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','unbinned'): 
              [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'):
              [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],

    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned'): 
              [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'):
              [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],

    # Hamamatsu
    ('GMOS-N','S10892-01','GMOS + S10892-01','unbinned'):
               [(2046, 2086, 1, 4224), (4133, 4176, 1, 4224)],

    # from GIRAF gmos/data/chipgaps.dat, Rev1.7
    ('GMOS-S','S10892','GMOS + Hamamatsu','unbinned'):
               [(2025, 2130, 1, 4176), (4140, 4240, 1, 4176)] 
}

blocksize = {  
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'): (2048,4608),
    ('GMOS-S','SDSU II CCD','GMOS + Blue1',           'unbinned'): (2048,4608),
    ('GMOS-N','SDSU II CCD','GMOS + Red1',            'unbinned'): (2048,4608),
    ('GMOS-N','SDSU II CCD','GMOS + e2v DD CCD42-90', 'unbinned'): (2048,4608),
    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned'): (2048,4608),
    ('GMOS-N','S10892-01',  'GMOS + S10892-01',       'unbinned'): (2048,4224),
    ('GMOS-S','S10892',     'GMOS + Hamamatsu',       'unbinned'): (2048,4224)  # ??
}

mosaic_grid = {
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','unbinned') :  (3,1),
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'):  (3,1),

    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned') : (3,1), 
    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned') :   (3,1), 

    # Hamamatsu
    ('GMOS-N','S10892-01','GMOS + S10892-01','unbinned') : (3,1),
    ('GMOS-S','S10892','GMOS + Hamamatsu','unbinned'): (3,1)
}

magnification = {
    ('GMOS-S','SDSU II CCD','GMOS + Blue1','unbinned') : [(1.,1.),(1.,1.),(1.,1.)],
    ('GMOS-S','SDSU II CCD','GMOS + Blue1 + new CCD1','unbinned'):  [(1.,1.),(1.,1.),(1.,1.)],

    ('GMOS-N','SDSU II e2v DD CCD42-90','GMOS + e2v DD CCD42-90','unbinned') :[(1.,1.),(1.,1.),(1.,1.)],
    ('GMOS-N','SDSU II CCD','GMOS + Red1','unbinned') :  [(1.,1.),(1.,1.),(1.,1.)], 
    ('GMOS-N','S10892-01','GMOS + S10892-01','unbinned') : [(1.,1.),(1.,1.),(1.,1.)],
    ('GMOS-S','S10892','GMOS + Hamamatsu','unbinned'): [(1.,1.),(1.,1.),(1.,1.)]
}

interpolator= {  # Values could be 'linear','nearest','spline2','spline3','spline4'
              'SCI':'linear','DQ':'linear', 'VAR':'linear','OBJMASK':'linear',
}
ref_block = { 'ref_block':(1,0)    # (0-base). Reference detector (x,y) position 
                                   # in the mosaic grid
}
