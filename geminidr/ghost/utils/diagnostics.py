#!/usr/bin/env python3

"""A script to do diagnostic checks on reduced ghost data.

It displays several windows showing how good the fits are and a few other things

This script should be ran from within the finished reduction folder, which
should contain the 'calibrations/' directory that are needed and finished
reduction files (extracted profiles/barycentric corrected)
"""

from __future__ import division, print_function
import numpy as np
from geminidr.ghost import polyfit
import glob
import sys
import astropy.io.fits as pyfits
import pylab as pl
from cycler import cycler
import input_locations

user='joao'

def plot_arcs(arc_data, thar_spec, w_map, title):
    """ Function used to plot two panels, one containing the extracted arc
    with the ThAr lamp spectrum superimposed, and one containing the difference
    between the two, to look for particularly bad regions of the fit.

    Parameters
    ----------

    arc_data: 
    """
    pl.rc('axes', prop_cycle=(cycler('color', ['b', 'r'])))
    f, axes = pl.subplots(3, 1, sharex='all')
    f.suptitle(title)
    # We always have 3 objects
    for obj in range(3):
        axes[obj].plot(w_map.T, arc_data[:, :, obj].T)
        axes[obj].set_title('Object %s' % (str(obj + 1)))
        thar_range = np.where((thar_spec[0] > w_map.min())
                               & (thar_spec[0] < w_map.max()))
        thar_spec[1] = thar_spec[1] * (arc_data[:, :, obj].max() /
                                        thar_spec[1][thar_range].max())
        axes[obj].plot(thar_spec[0][thar_range], thar_spec[1][thar_range],
                       ls='-.',
                       color='green')

    pl.show()


# Let's start by checking the fits. We use the same method as the slider adjust
# and let the user check things individually.


flat_list = glob.glob('calibrations/processed_flat/*flat.fits')
arc_list = glob.glob('calibrations/processed_arc/*arc.fits')

modes = ['high', 'std']
cams = ['blue', 'red']

# Now cycle through available modes. or just the ones required
# by detecting particular keywords in the sys arguments.
if len(sys.argv) > 1:
    if 'high' in sys.argv:
        modes = ['high']
    if 'std' in sys.argv:
        modes = ['std']
    if 'red' in sys.argv:
        cams = ['red']
    if 'blue' in sys.argv:
        cams = ['blue']


for mode in modes:
    for cam in cams:
        files = input_locations.Files(user=user, mode=mode, cam=cam)
        ghost = polyfit.ghost.GhostArm(cam, mode=mode)
        print('Inspecting flat and arc fits from the %s camera in %s mode' %
              (cam, mode))
        # Find the default models for things not inspected
        rotparams = files.rotparams
        specparams = files.specparams
        spatparams = files.spatparams
        wparams = files.waveparams

        flat_file_location = [value for value in flat_list if cam in value
                              and mode in value][0]
        flat_file = pyfits.open(flat_file_location)
        print('Inspecting file %s' % (flat_file_location))
        xparams = flat_file['XMOD'].data

        dummy = ghost.spectral_format_with_matrix(xparams, wparams,
                                                  spatparams, specparams,
                                                  rotparams)

        flat_conv = ghost.slit_flat_convolve(flat_file['SCI'].data)
        plot_title = 'Convolution plot for camera %s in %s mode.' % (cam, mode)
        adjusted_params=ghost.manual_model_adjust(flat_conv,
                                                    model = 'position',
                                                    xparams = xparams,
                                                    percentage_variation = 10,
                                                    title = plot_title)

        plot_title='Regular flat for camera %s in %s mode.' % (cam, mode)
        adjusted_params=ghost.manual_model_adjust(flat_file['SCI'].data,
                                                    model='position',
                                                    xparams=xparams,
                                                    percentage_variation=10,
                                                    title=plot_title)

        # Now the arcs
        arcs_list=[value for value in arc_list
                   if cam in value and mode in value]
        for arc in arcs_list:
            print('Inspecting file %s' % (arc))
            arc_file=pyfits.open(arc)
            wparams=arc_file['WFIT'].data
            arc_data=arc_file['SCI'].data
            dummy=ghost.spectral_format_with_matrix(xparams, wparams,
                                                  spatparams, specparams,
                                                  rotparams)
            plot_title='Arc %s with superimposed template in green.' % (arc)
            thar_spec = files.thar_spectrum(files.arclinefile)
            plot_arcs(arc_data, thar_spec, ghost.w_map, title=plot_title)
