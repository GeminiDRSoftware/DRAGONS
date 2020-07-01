#!/usr/bin/env python
"""
Tests for the `createSlitIllumination` primitive. The primitive itself is
defined in :mod:`~geminidr.core.primitives_spect` but these tests use GMOS Spect
data.
"""
import os
import pytest
import warnings

from copy import deepcopy

import astrodata

from astrodata.testing import download_from_archive
from gempy.utils import logutils
from geminidr.gmos import primitives_gmos_longslit
from recipe_system.reduction.coreReduce import Reduce


@pytest.mark.gmosls
@pytest.mark.preprocessed
@pytest.mark.parametrize("ad", ["S20190204S0006_mtflat.fits"], indirect=True)
def test_create_slit_illumination_with_mosaicked_data(ad, change_working_dir, request):
    """
    Test that can run `createSlitIllumination` in mosaicked data.
    """
    plot = request.config.getoption("--do-plots")

    with change_working_dir():
        print("Running tests inside folder:\n  {}".format(os.getcwd()))
        p = primitives_gmos_longslit.GMOSLongslit([ad])
        slit_illum_ad = p.createSlitIllumination(border=10, debug_plot=plot)[0]

        if plot:
            os.makedirs("plots", exist_ok=True)
            os.rename(slit_illum_ad.filename.replace(".fits", ".png"),
                      os.path.join("plots", ad.filename.replace(".fits", ".png")))

        for ext, slit_ext in zip(ad, slit_illum_ad):
            assert {ext.shape == slit_ext.shape}


@pytest.mark.gmosls
@pytest.mark.preprocessed
@pytest.mark.parametrize("ad", ["S20190204S0006_tflat.fits"], indirect=True)
def test_create_slit_illumination_with_multi_extension_data(ad, change_working_dir, request):
    """
    Test that can run `createSlitIllumination` in multi-extension data.
    """
    plot = request.config.getoption("--do-plots")

    with change_working_dir():
        print("Running tests inside folder:\n  {}".format(os.getcwd()))
        p = primitives_gmos_longslit.GMOSLongslit([ad])
        slit_illum_ad = p.createSlitIllumination(border=10, debug_plot=plot)[0]

        if plot:
            os.makedirs("plots", exist_ok=True)
            os.rename(slit_illum_ad.filename.replace(".fits", ".png"),
                      os.path.join("plots", ad.filename.replace(".fits", ".png")))

        for ext, slit_ext in zip(ad, slit_illum_ad):
            assert {ext.shape == slit_ext.shape}


# --- Helper functions and fixtures -------------------------------------------
@pytest.fixture
def ad(request, path_to_inputs):
    """
    Returns the pre-processed spectrum file.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the `distortionDetermine`
        primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad


# -- Recipe to create pre-processed data ---------------------------------------
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
            p.stackFrames()

            os.chdir("inputs/")

            # Write non-mosaicked data
            tflat = p.writeOutputs(suffix="_tflat", strip=True)[0]

            # Write mosaicked data
            p = primitives_gmos_longslit.GMOSLongslit([deepcopy(tflat)])
            p.mosaicDetectors()
            p.writeOutputs(suffix="_mtflat", strip=True)

            os.chdir("../")


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
