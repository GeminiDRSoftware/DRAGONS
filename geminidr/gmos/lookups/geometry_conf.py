
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

gap_dict_bin = {(0,0):(0,0), (1,0):(36,0), (2,0):(36,0)}
gap_dict_unbin = {(0,0):(0,0), (1,0):(37,0), (2,0):(37,0)}

hamgap_dict_bin = {(0,0):(0,0), (1,0):(60,0), (2,0):(60,0)}
hamgap_dict_unbin = {(0,0):(0,0), (1,0):(61,0), (2,0):(61,0)}

gmosn_hamgap_dict_bin = {(0,0):(0,0), (1,0):(66,0), (2,0):(66,0)}
gmosn_hamgap_dict_unbin = {(0,0):(0,0), (1,0):(67,0), (2,0):(67,0)}

# NOTE: The gap sets used for GMOS-N Hamamatsu may not be correct, as the
# keys below are simply using the gap sets for the old CCDs.
# - kra, 07-05-15
#
# Update to add GMOS DETECTOR value 'GMOS + Hamamatsu', to fields.
# This was the name early after install of the Hamamatsu ccds until
# final characterisation on 07-09-2015.
# - kra, 05-07-2017.
#
# gaps: (x_gap, y_gap)  One tuple per detector
#     Instr,      dettype,      detector,          bin/unbinned
# --------------------------------------------------------------
gaps = {
    # GMOS-N
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', 'unbinned'): gap_dict_unbin,
    ('e2v 10031-23-05,10031-01-03,10031-18-04', 'unbinned'): gap_dict_unbin,
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2' , 'unbinned'): gmosn_hamgap_dict_unbin,

    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', 'binned'): gap_dict_bin,
    ('e2v 10031-23-05,10031-01-03,10031-18-04', 'binned'): gap_dict_bin,
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2' , 'binned'): gmosn_hamgap_dict_bin,

    # GMOS-S
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04', 'unbinned'): gap_dict_unbin,
    ('EEV8056-20-03EEV8194-19-04EEV8261-07-04', 'unbinned'): gap_dict_unbin,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1',   'unbinned'): hamgap_dict_unbin,

    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04', 'binned'): gap_dict_bin,
    ('EEV8056-20-03EEV8194-19-04EEV8261-07-04', 'binned'): gap_dict_bin,
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1',   'binned'): hamgap_dict_bin,

    # Previous table with 'human readable' keys. DETID has a unique value for each chip set.
    # DETID keys above, previous keys follow.
    #('GMOS-N', 'SDSU II CCD', 'GMOS + e2v DD CCD42-90', 'unbinned'): gap_dict_unbin,
    #('GMOS-N', 'SDSU II e2v DD CCD42-90', 'GMOS + e2v DD CCD42-90', 'unbinned'): gap_dict_unbin,
    #('GMOS-N', 'S10892-N',    'GMOS-N + Hamamatsu',     'unbinned'): gmosn_hamgap_dict_unbin,

    # ('GMOS-S', 'SDSU II CCD', 'GMOS + Blue1 + new CCD1', 'unbinned'): gap_dict_unbin,
    # ('GMOS-S', 'SDSU II CCD', 'GMOS + Blue1',            'unbinned'): gap_dict_unbin,
    # ('GMOS-S', 'S10892',      'GMOS + Hamamatsu',        'unbinned'): hamgap_dict_unbin,
    # ('GMOS-S', 'S10892',      'GMOS + Hamamatsu_new',    'unbinned'): hamgap_dict_unbin,

    # ('GMOS-N', 'SDSU II CCD', 'GMOS + e2v DD CCD42-90', 'binned'): gap_dict_bin,
    # ('GMOS-N', 'SDSU II CCD', 'GMOS + Red1',            'binned'): gap_dict_bin,
    # ('GMOS-N', 'SDSU II e2v DD CCD42-90', 'GMOS + e2v DD CCD42-90', 'binned'): gap_dict_bin,
    # ('GMOS-N', 'S10892-N',    'GMOS-N + Hamamatsu',     'binned'): gmosn_hamgap_dict_unbin,

    # ('GMOS-S', 'SDSU II CCD', 'GMOS + Blue1 + new CCD1', 'binned'): gap_dict_bin,
    # ('GMOS-S', 'SDSU II CCD', 'GMOS + Blue1',            'binned'): gap_dict_bin,
    # ('GMOS-S', 'S10892',      'GMOS + Hamamatsu',        'binned'): hamgap_dict_bin,
    # ('GMOS-S', 'S10892',      'GMOS + Hamamatsu_new',    'binned'): hamgap_dict_bin
}

gaps_tile = gaps
gaps_transform = deepcopy(gaps)

#  (x_shift, y_shift) for detectors (1,2,3).
shift = {
    # GMOS-N
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03','unbinned'): [(-2.50,-1.58), (0.,0.), (3.8723, -1.86)],
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03',  'binned'): [(-3.50,-1.58), (0.,0.), (4.8723, -1.86)],

    ('e2v 10031-23-05,10031-01-03,10031-18-04','unbinned'): [(-2.7,-0.749), (0.,0.), (2.8014, 2.0500)],
    ('e2v 10031-23-05,10031-01-03,10031-18-04',  'binned'): [(-3.7,-0.749), (0.,0.), (3.8014, 2.0500)],

    # GMOS-S
    ('EEV8056-20-03EEV8194-19-04EEV8261-07-04','unbinned'): [(-1.44,5.46), (0.,0.), (7.53, 9.57)],
    ('EEV8056-20-03EEV8194-19-04EEV8261-07-04',  'binned'): [(-2.44,5.46), (0.,0.), (7.53, 9.57)],

    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04','unbinned'): [(-1.49,-0.22), (0.,0.), (4.31, 2.04)],
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04',  'binned'): [(-2.49,-0.22), (0.,0.), (5.31, 2.04)],

    # Hamamatsu
    # GMOS-N
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2','unbinned'): [(-0.95,-0.21739), (0.,0.), (0.48, 0.1727)],
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2',  'binned'): [(-1.95,-0.21739), (0.,0.), (1.48, 0.1727)],

    # GMOS-S
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1','unbinned'): [(-1.2,0.71), (0.,0.), (0., -0.73)],
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1',  'binned'): [(-2.4,0.71), (0.,0.), (0., -0.73)],

}

rotation = {
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03','unbinned'): (-0.004, 0.0, -0.046),
    ('e2v 10031-23-05,10031-01-03,10031-18-04','unbinned'): (-0.009, 0.0, -0.003),

    ('EEV8056-20-03EEV8194-19-04EEV8261-07-04','unbinned'): (-0.01,  0.0, 0.02),
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04','unbinned') : (0.011, 0.0, 0.012),

    # GMOS-N Hamamatsu
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2','unbinned'): (-0.004, 0.0, -0.00537),
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2',  'binned'): (-0.004, 0.0, -0.00537),

    # GMOS-S Hamamatsu
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'unbinned'): (0., 0., 0.),
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1',   'binned'): (0., 0., 0.),
}

chip_gaps = {
    # GMOS-N
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03','unbinned'): [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],
    ('e2v 10031-23-05,10031-01-03,10031-18-04','unbinned'): [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],

    # GMOS-S
    ('EEV8056-20-03EEV8194-19-04EEV8261-07-04','unbinned'): [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04','unbinned'): [(2046, 2086, 1, 4608), (4133, 4176, 1, 4608)],

    # GMOS-N Hamamatsu
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2','unbinned'):[(2032, 2130, 1, 4176), (4147, 4245, 1, 4176)],

    # GMOS-S Hamamatsu (from GIRAF gmos/data/chipgaps.dat, Rev1.7)
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1', 'unbinned'):[(2025, 2130, 1, 4176), (4140, 4240, 1, 4176)],
}

blocksize = {
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04', 'unbinned'): (2048,4608),
    ('EEV8056-20-03EEV8194-19-04EEV8261-07-04', 'unbinned'): (2048,4608),
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', 'unbinned'): (2048,4608),
    ('e2v 10031-23-05,10031-01-03,10031-18-04', 'unbinned'): (2048,4608),
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2',  'unbinned'): (2048,4224),
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1',   'unbinned'): (2048,4224),
}

mosaic_grid = {
    ('EEV8056-20-03EEV8194-19-04EEV8261-07-04','unbinned'): (3,1),
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04','unbinned'): (3,1),

    ('e2v 10031-23-05,10031-01-03,10031-18-04','unbinned'): (3,1),
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03','unbinned'): (3,1),

    # Hamamatsu
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2', 'unbinned'): (3,1),
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1',  'unbinned'): (3,1),
}

magnification = {
    ('EEV8056-20-03EEV8194-19-04EEV8261-07-04', 'unbinned'): [(1.,1.),(1.,1.),(1.,1.)],
    ('EEV2037-06-03EEV8194-19-04EEV8261-07-04', 'unbinned'): [(1.,1.),(1.,1.),(1.,1.)],

    ('e2v 10031-23-05,10031-01-03,10031-18-04', 'unbinned'): [(1.,1.),(1.,1.),(1.,1.)],
    ('EEV9273-16-03EEV9273-20-04EEV9273-20-03', 'unbinned'): [(1.,1.),(1.,1.),(1.,1.)],
    ('BI13-20-4k-1,BI12-09-4k-2,BI13-18-4k-2',  'unbinned'): [(1.,1.),(1.,1.),(1.,1.)],
    ('BI5-36-4k-2,BI11-33-4k-1,BI12-34-4k-1',   'unbinned'): [(1.,1.),(1.,1.),(1.,1.)],
}

# Values could be 'linear','nearest','spline2','spline3','spline4'
interpolator= { 'SCI': 'linear', 'DQ': 'linear', 'VAR': 'linear', 'OBJMASK': 'linear' }

# (0-base). Ref detector (x,y) position in mosaic grid
ref_block = { 'ref_block': (1, 0) }
