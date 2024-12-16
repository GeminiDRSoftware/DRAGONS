#!/usr/bin/python
"""
Tests related to GMOS Long-slit Spectroscopy data reduction.
"""
import glob
import os
import pytest

from astrodata.testing import download_from_archive
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

datasets = {

    'GN-2017A-FT-19-15': {
        "arcs": ["N20170601S0850.fits"],
        "bias": [f"N20170602S{i:04d}.fits" for i in range(590, 595)],
        "flat": ["N20170601S0459.fits"],
        "sci": [f"N20170601S{i:04d}.fits" for i in range(456, 459)],
        "std": ["N20170529S0168.fits"],
        "user_pars": [],
    },

    # DO NOT UNCOMMENT THESE WITHOUT REDUCING TO 1 ARC AND 1 FLAT PER TEST

    # "GN-2018B-Q-108-27": {
    #     "arcs": ["N20181011S0220.fits"],
    #     "bias": ["N20181011S0609.fits", "N20181011S0610.fits",
    #              "N20181011S0611.fits", "N20181011S0612.fits",
    #              "N20181011S0613.fits"],
    #     "flat": ["N20181011S0219.fits", "N20181011S0225.fits"],
    #     "sci": [f"N20181011S{i:04d}.fits" for i in range(221, 225)],
    #     "user_pars": [],
    # }

    # 'GS-2016B-Q-54-32': {
    #     "arcs": ["S20170103S0149.fits", "S20170103S0152.fits"],
    #     "bias": [f"S20170103S{i:04d}.fits" for i in range(216, 221)],
    #     "flat": ["S20170103S0153.fits"],
    #     "sci": [f"S20170103S{i:04d}.fits" for i in (147, 148, 150, 151)],
    #     "std": [],
    #     "user_pars": [()],
    # }
}


@pytest.mark.slow
@pytest.mark.integration_test
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("test_case", datasets.keys())
def test_reduce_ls_spect(change_working_dir, keep_data, test_case):
    """
    Tests that we can run all the data reduction steps on a complete dataset.

    Parameters
    ----------
    change_working_dir : fixture
        Change the current work directory using a context manager.
    keep_data : fixture
        Keep pre-stack data? (Uses a lot of disk space)
    test_case : str
        Test parameter containing the key to the `datasets` dictionary.
    """
    with change_working_dir(test_case):

        cals = []

        # Reducing bias
        bias_filenames = datasets[test_case]["bias"]
        bias_paths = [download_from_archive(f) for f in bias_filenames]
        cals = reduce(bias_paths, f"bias_{test_case}", cals, save_to="processed_bias")

        # Reducing arcs
        arcs_filenames = datasets[test_case]["arcs"]
        arcs_paths = [download_from_archive(f) for f in arcs_filenames]
        cals = reduce(arcs_paths, f"flat_{test_case}", cals, save_to="processed_arc")

        # Reducing flats
        flat_filenames = datasets[test_case]["flat"]
        flat_paths = [download_from_archive(f) for f in flat_filenames]
        cals = reduce(flat_paths, f"flat_{test_case}", cals, save_to="processed_flat")

        # Reducing standard stars
        if "std" in datasets[test_case]:
            if len(datasets[test_case]["std"]) > 0:
                std_filenames = datasets[test_case]["std"]
                std_paths = [download_from_archive(f) for f in std_filenames]
                cals = reduce(std_paths, f"std_{test_case}", cals,
                              save_to="processed_standard")

        # Reducing science
        if "sci" in datasets[test_case]:

            std_fname = [c.replace("processed_standard:", "")
                         for c in cals if "processed_standard" in c]

            if std_fname:
                datasets[test_case]["user_pars"] += \
                    [("fluxCalibrate:standard", std_fname.pop())]

            sci_filenames = datasets[test_case]["sci"]
            sci_paths = [download_from_archive(f) for f in sci_filenames]
            _ = reduce(sci_paths, f"sci_{test_case}", cals,
                       user_pars=datasets[test_case]["user_pars"])

            if not keep_data:
                print(' Deleting pre-stack files.')
                [os.remove(f) for f in glob.glob("*_CRMasked.fits")]
                [os.remove(f) for f in glob.glob("*_align.fits")]


# -- Helper functions ---------------------------------------------------------
def reduce(file_list, label, calib_files, recipe_name=None, save_to=None,
           user_pars=None):
    """
    Helper function used to prevent replication of code.

    Parameters
    ----------
    file_list : list
        List of files that will be reduced.
    label : str
        Labed used on log files name.
    calib_files : list
        List of calibration files properly formatted for DRAGONS Reduce().
    recipe_name : str, optional
        Name of the recipe used to reduce the data.
    save_to : str, optional
        Stores the calibration files locally in a list.
    user_pars : list, optional
        List of user parameters

    Returns
    -------
    list : an updated list of calibration files
    """
    objgraph = pytest.importorskip("objgraph")

    logutils.get_logger().info("\n\n\n")
    logutils.config(file_name=f"test_image_{label}.log")

    r = Reduce()
    r.files = file_list
    r.ucals = normalize_ucals(calib_files)
    r.uparms = user_pars
    r.mode = 'ql'  # for 3.0.x

    if recipe_name:
        r.recipename = recipe_name

    r.runr()

    if save_to:
        [calib_files.append(
            "{}:{}".format(save_to, os.path.join("calibrations", save_to, f)))
            for f in r.output_filenames]
        [os.remove(f) for f in r.output_filenames]

    # check that we are not leaking objects
    assert len(objgraph.by_type('NDAstroData')) == 0, ("Leaking objects",
           [x.shape for x in objgraph.by_type('NDAstroData')])

    return calib_files


# -- Custom configuration -----------------------------------------------------
@pytest.fixture(scope='module')
def keep_data(request):
    """
    By default, the tests will delete pre-stack files to save disk space. If one
    needs to keep them for debugging, one can pass --keep-data argument to the
    command line call to force the tests to keep this data.

    Parameters
    ----------
    request : fixture
        Represents the test that calls this function.
    """
    return request.config.getoption("--keep-data")


if __name__ == '__main__':
    pytest.main()
