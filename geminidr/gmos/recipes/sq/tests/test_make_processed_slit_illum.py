#!/usr/bin/env python
"""
Tests that make sure that the `makeProcessedSlitIllum` works as expected.
"""
import os
import pytest

import astrodata
import gemini_instruments
import numpy as np

from astrodata.testing import download_from_archive
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals
from recipe_system.testing import ref_ad_factory


# Ensures repeatability of the tests
np.random.seed(0)

datasets = ["S20190204S0006.fits"]

associated_calibrations = {
    "S20190204S0006.fits": "S20190203S0110_bias.fits",
}


@pytest.mark.gmosls
@pytest.mark.integration_test
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("processed_slit_illum", datasets, indirect=True)
def test_make_processed_slit_illum(processed_slit_illum, ref_ad_factory):

    assert "_slitIllum" in processed_slit_illum.filename

    ref_ad = ref_ad_factory(processed_slit_illum.filename)

    for ext, ext_ref in zip(processed_slit_illum, ref_ad):
        np.testing.assert_allclose(ext.data, ext_ref.data, atol=1)


@pytest.fixture(scope='module')
def processed_slit_illum(change_working_dir, path_to_inputs, request):
    """
    Returns the processed slit illumination function that will be analysed.

    Parameters
    ----------
    change_working_dir : pytest.fixture
        Fixture that changes the working directory (see :mod:`astrodata.testing`).
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the `applyQECorrection`.
    """
    twi_filename = request.param
    twi_path = download_from_archive(twi_filename)
    twi_ad = astrodata.from_file(twi_path)

    print(twi_ad.tags)

    master_bias = os.path.join(
        path_to_inputs, associated_calibrations[twi_filename])

    assert os.path.exists(master_bias)

    calibration_files = ['processed_bias:{}'.format(master_bias)]

    with change_working_dir():
        print("Reducing SLITILLUM in folder:\n  {}".format(os.getcwd()))
        logutils.config(
            file_name='log_flat_{}.txt'.format(twi_ad.data_label()))

        reduce = Reduce()
        reduce.files.extend([twi_path])
        reduce.mode = 'sq'
        reduce.recipename = 'makeProcessedSlitIllum'
        reduce.ucals = normalize_ucals(calibration_files)
        reduce.runr()

        _processed_twi_filename = reduce.output_filenames.pop()
        _processed_twi = astrodata.from_file(_processed_twi_filename)

    return _processed_twi


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

    _associated_calibrations = {
        "S20190204S0006.fits": {
            "bias": ["S20190203S0110.fits",
                     "S20190203S0109.fits",
                     "S20190203S0108.fits",
                     "S20190203S0107.fits",
                     "S20190203S0106.fits"],
        }
    }

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/recipes/sq/test_make_processed_slit_illum/inputs"
    path = os.path.join(root_path, module_path)

    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    print('Current working directory:\n    {:s}'.format(os.getcwd()))
    for filename, cals in _associated_calibrations.items():

        print('Downloading files...')
        bias_path = [download_from_archive(f) for f in cals['bias']]

        print('Reducing BIAS for {:s}'.format(filename))
        logutils.config(
            file_name='log_bias_{}.txt'.format(filename.replace(".fits", "")))

        bias_reduce = Reduce()
        bias_reduce.files.extend(bias_path)
        bias_reduce.runr()


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
