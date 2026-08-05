#!/usr/bin/env python
import os

import numpy as np
import pytest
import shutil

import astrodata
# noinspection PyUnresolvedReferences
import gemini_instruments
import geminidr
from astrodata.testing import download_from_archive
# I need these when I'm debugging as a standalone pytest
# from astrodata.testing import change_working_dir, path_to_outputs, path_to_refs, path_to_inputs, path_to_test_data
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

# noinspection PyUnresolvedReferences
from recipe_system.testing import reduce_bias, ref_ad_factory

datasets = [
    "S20180707S0043.fits",  # B600 at 0.520 um
    "S20190502S0096.fits",  # B600 at 0.525 um
    "S20200122S0020.fits",  # B600 at 0.520 um
    "N20200101S0055.fits",  # B1200 at 0.495 um
    # "S20180410S0120.fits",  # B1200 at 0.595 um  # Scattered light?
    # "S20190410S0053.fits",  # B1200 at 0.463 um  # Scattered light?
]

associated_calibrations = {
    "S20180707S0043.fits": "S20180707S0187_bias.fits",
    "S20190502S0096.fits": "S20190502S0221_bias.fits",
    "S20200122S0020.fits": "S20200121S0170_bias.fits",
    "N20200101S0055.fits": "N20200101S0240_bias.fits",
    "S20180410S0120.fits": "S20180410S0132_bias.fits",
    "S20190410S0053.fits": "S20190410S0297_bias.fits",
}


# -- Tests --------------------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.parametrize("processed_flat", datasets, indirect=True)
@pytest.mark.preprocessed_data
def test_processed_flat_has_median_around_one(processed_flat):
    """
    Tests if the processed flat contains values around one.

    Parameters
    ----------
    processed_flat : pytest.fixture
        Fixture containing an instance of the processed flat.
    """
    for ext in processed_flat:
        data = np.ma.masked_array(ext.data, mask=ext.mask)
        np.testing.assert_almost_equal(np.ma.median(data.ravel()), 1.0, decimal=3)


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.parametrize("processed_flat", datasets, indirect=True)
@pytest.mark.preprocessed_data
def test_processed_flat_has_small_std(processed_flat):
    """
    Tests if the processed flat has a small standard deviation.

    Parameters
    ----------
    processed_flat : pytest.fixture
        Fixture containing an instance of the processed flat.
        """
    for ext in processed_flat:
        data = np.ma.masked_array(ext.data, mask=ext.mask)
        np.testing.assert_array_less(np.std(data.ravel()), 0.1)


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.parametrize("processed_flat", datasets, indirect=True)
@pytest.mark.preprocessed_data
def test_regression_processed_flat(processed_flat, ref_ad_factory):
    """
    Regression tests for the standard makeProcessedFlat recipe.

    Parameters
    ----------
    processed_flat : pytest.fixture
        Fixture containing an instance of the processed flat.
    ref_ad_factory : pytest.fixture
        Fixture containing a function that will receive the input file an return
        the path to the reference data.
    """
    ref_flat = ref_ad_factory(processed_flat.filename)
    for ext, ext_ref in zip(processed_flat, ref_flat):
        astrodata.testing.assert_most_equal(ext.mask, ext_ref.mask, 200)
        astrodata.testing.assert_most_close(
            ext.data[ext.mask==0], ext_ref.data[ext_ref.mask==0],
            200, atol=0.01)


# -- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope='module')
def processed_flat(change_working_dir, path_to_inputs, request):
    """
    Returns the processed flat that will be analysed.

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
    np.random.seed(0)

    flat_filename = request.param
    flat_path = download_from_archive(flat_filename)
    flat_raw = astrodata.open(flat_path)

    master_bias = os.path.join(path_to_inputs, associated_calibrations[flat_filename])
    calibration_files = ['processed_bias:{}'.format(master_bias)]

    with change_working_dir():
        print("Reducing FLATs in folder:\n  {}".format(os.getcwd()))
        logutils.config(
            file_name='log_flat_{}.txt'.format(flat_raw.data_label()))

        # Allow retrieval of BPM from archive
        with open("test.cfg", "w") as f:
            f.write("[calibs]\n")
            f.write("databases = https://archive.gemini.edu get\n")

        reduce = Reduce()
        reduce.files.extend([flat_path])
        reduce.mode = 'ql'
        reduce.ucals = normalize_ucals(calibration_files)
        reduce.config_file = 'test.cfg'
        reduce.runr()

        # Clean up duplicated files
        shutil.rmtree('calibrations/')

        _processed_flat_filename = reduce.output_filenames.pop()
        _processed_flat = astrodata.open(_processed_flat_filename)

    return _processed_flat


# -- Recipe to create pre-processed data ---------------------------------------
def create_master_bias_for_tests():
    """
    Creates input bias data for tests.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """
    root_path = os.path.join("./dragons_test_inputs/")
    module_path = f"geminidr/gmos/recipes/ql/{__file__.split('.')[0]}/"
    path = os.path.join(root_path, module_path, "inputs/")

    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    associated_biases = {
        "S20180707S0043.fits": ["S20180707S0187.fits",
                                "S20180707S0188.fits",
                                "S20180707S0189.fits",
                                "S20180707S0190.fits",
                                "S20180707S0191.fits"],
        "S20190502S0096.fits": ["S20190502S0221.fits",
                                "S20190502S0222.fits",
                                "S20190502S0223.fits",
                                "S20190502S0224.fits",
                                "S20190502S0225.fits"],
        "S20200122S0020.fits": ["S20200121S0170.fits",
                                "S20200121S0171.fits",
                                "S20200121S0172.fits",
                                "S20200121S0173.fits",
                                "S20200121S0174.fits"],
        "N20200101S0055.fits": ["N20200101S0240.fits",
                                "N20200101S0241.fits",
                                "N20200101S0242.fits",
                                "N20200101S0243.fits",
                                "N20200101S0244.fits"],
        "S20180410S0120.fits": ["S20180410S0132.fits",
                                "S20180410S0133.fits",
                                "S20180410S0134.fits",
                                "S20180410S0135.fits",
                                "S20180410S0136.fits"],
        "S20190410S0053.fits": ["S20190410S0297.fits",
                                "S20190410S0298.fits",
                                "S20190410S0299.fits",
                                "S20190410S0300.fits",
                                "S20190410S0301.fits"],
    }

    for filename, bias_files in associated_biases.items():

        print('Downloading files...')
        sci_path = download_from_archive(filename)
        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        bias_paths = [download_from_archive(f) for f in bias_files]

        print('Reducing BIAS for {:s}'.format(data_label))
        logutils.config(file_name='log_bias_{}.txt'.format(data_label))
        bias_reduce = Reduce()
        bias_reduce.files.extend(bias_paths)
        bias_reduce.runr()
        
        shutil.rmtree("calibrations/")


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_master_bias_for_tests()
    else:
        pytest.main()
