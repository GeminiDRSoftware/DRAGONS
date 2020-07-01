#!/usr/bin/env python
"""
Tests for the `createSlitIllumination` primitive. The primitive itself is
defined in :mod:`~geminidr.core.primitives_spect` but these tests use GMOS Spect
data.
"""
import itertools
import logging
import os
import pytest
import time
import warnings

from copy import copy, deepcopy

import astrodata
import astrofaker
import gemini_instruments
import ipywidgets as widgets
import matplotlib as mpl
import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy import visualization as vis
from astropy.modeling import fitting, models
from astrodata.testing import download_from_archive
from cycler import cycler
from gempy.utils import logutils
from gempy.library import astromodels, transform
from geminidr.core.primitives_spect import _transpose_if_needed
from geminidr.gmos import primitives_gmos_longslit
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS
from ipywidgets import interact, interactive, interact_manual, fixed
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from recipe_system.utils.reduce_utils import normalize_ucals
from recipe_system.reduction.coreReduce import Reduce


@pytest.mark.gmosls
@pytest.mark.preprocessed
def test_create_slit_illumination():
    pass


def create_inputs_recipe():
    """
    Creates input data for tests using pre-processed twilight flat data and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """

    associated_calibrations = {
        "S20190204S0006.fits": {
            "bias": ["S20190203S0110.fits",
                     "S20190203S0109.fits",
                     "S20190203S0108.fits",
                     "S20190203S0107.fits",
                     "S20190203S0106.fits"],
        }
    }

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/test_create_slit_illumination/"
    path = os.path.join(root_path, module_path)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, cals in associated_calibrations.items():

        print('Downloading files...')
        tflat_path = download_from_archive(filename)
        bias_path = [download_from_archive(f) for f in cals['bias']]

        tflat_ad = astrodata.open(tflat_path)
        data_label = tflat_ad.data_label()

        print('Reducing BIAS for {:s}'.format(data_label))
        logutils.config(file_name='log_bias_{}.txt'.format(data_label))
        bias_reduce = Reduce()
        bias_reduce.files.extend(bias_path)
        bias_reduce.runr()
        bias_master = bias_reduce.output_filenames.pop()
        del bias_reduce

        print('Reducing twilight flat:')
        logutils.config(file_name='log_sflat.txt')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            p = primitives_gmos_longslit.GMOSLongslit([tflat_ad])

            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            p.biasCorrect(bias=bias_master)
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.stackFrames(suffix="_sflat")

            os.chdir("inputs/")

            # Write non-mosaicked data
            tflat = p.writeOutputs()[0]

            # Write mosaicked data
            p = primitives_gmos_longslit.GMOSLongslit([deepcopy(tflat)])
            p.mosaicDetectors(suffix="_msflat")
            p.writeOutputs()

            os.chdir("../")


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
