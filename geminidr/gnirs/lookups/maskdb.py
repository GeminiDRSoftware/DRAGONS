# gnirs/maskdb.py
#
# This file contains the bad pixel mask (BPMs), illumination mask,
# and mask definition file (MDF) lookup tables for GNIRS

# Cut-on and cut-off wavelengths (um) of GNIRS order-blocking filters, based on conservative transmissivity (1%),
# or inter-order minima in the flats (determined using the GN filter set)
#
# Instruction for finding filter cut-on and cut-off wvls (same as for F2):
# https://docs.google.com/document/d/1LVTUFWkXJygkRUvqjsFm_7VZnqy7I4fy/edit?usp=sharing&ouid=106387637049533476653&rtpof=true&sd=true
bl_filter_range_dict = {'X_G0518': (1.01, 1.19), # GN filters
                        'J_G0517': (1.15, 1.385),
                        'H_G0516': (1.46, 1.84),
                        'K_G0515': (1.89, 2.54),
                        'L_G0527': (2.77, 4.44),
                        'M_G0514': (4.2, 6.0),
                        'X_G0506': (1.01, 1.19), # GS filter (assumed to be the same as the GN filters)
                        'J_G0505': (1.15, 1.385),
                        'H_G0504': (1.46, 1.84),
                        'K_G0503': (1.89, 2.54),
                        'L_G0502': (2.77, 4.44),
                        'M_G0501': (4.2, 6.0)}
