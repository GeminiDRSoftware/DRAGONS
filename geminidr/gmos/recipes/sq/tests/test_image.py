#!/usr/bin/env python

import glob
import pytest
import os

from astrodata.testing import download_from_archive
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

from gempy.utils import logutils

datasets = {

    "GN_HAM_2x2_z-band": {
        "bias": [f"N20170912S{n:04d}.fits" for n in range(295, 300)] +
                [f"N20170914S{n:04d}.fits" for n in range(481, 486)] +
                [f"N20170915S{n:04d}.fits" for n in range(337, 342)],
        "flat": [f"N20170915S{n:04d}.fits" for n in range(274, 288)],
        "sci": [f"N20170913S{n:04d}.fits" for n in range(153, 159)],
        "ucals": [],
    },

    "GN_EEV_2x2_g-band": {
        # Only three files to avoid memory errors or to speed up the test
        "bias": [f"N20020214S{n:03d}.fits" for n in range(22, 27)][:3],
        "flat": [f"N20020211S{n:03d}.fits" for n in range(156, 160)][:3],
        "sci": [f"N20020214S{n:03d}.fits" for n in range(59, 64)][:3],
        "ucals": [],
    },

    "GS_HAM_1x1_i-band": {
        "bias": [f"S20171204S{n:04d}.fits" for n in range(22, 27)] +
                [f"S20171206S{n:04d}.fits" for n in range(128, 133)],
        "flat": [f"S20171206S{n:04d}.fits" for n in range(120, 128)],
        "sci": [f"S20171205S{n:04d}.fits" for n in range(62, 77)],
        "ucals": [
            ('stackFrames:memory', 1),
            # ('addDQ:user_bpm', 'fixed_bpm_1x1_FullFrame.fits'),
            ('adjustWCSToReference:rotate', True),
            ('adjustWCSToReference:scale', True),
            ('resampleToCommonFrame:interpolator', 'spline3')]
    },

    "GS_HAM_2x2_i-band_std": {
        "bias": [f"S20171204S{n:04d}.fits" for n in range(37, 42)],
        "flat": [f"S20171120S{n:04d}.fits" for n in range(131, 140)],
        "std": ["S20171205S0077.fits"],
        "ucals": [
            ('stackFrames:memory', 1),
            # ('addDQ:user_bpm', 'fixed_bpm_2x2_FullFrame.fits'),
            ('resampleToCommonFrame:interpolator', 'spline3')
        ]
    },

}


#@pytest.mark.skip("Can't run in Jenkins - Need more investigation")
@pytest.mark.gmosimage
@pytest.mark.integration_test
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("test_case", list(datasets.keys())[:1])
def test_reduce_image(change_working_dir, keep_data, test_case):
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
        cals = run_reduce(bias_paths, f"bias_{test_case}", cals, save_to="processed_bias")

        # Reducing flats
        flat_filenames = datasets[test_case]["flat"]
        flat_paths = [download_from_archive(f) for f in flat_filenames]
        cals = run_reduce(flat_paths, f"flat_{test_case}", cals, save_to="processed_flat")

        # Reducing standard stars
        if "std" in datasets[test_case]:
            std_filenames = datasets[test_case]["std"]
            std_paths = [download_from_archive(f) for f in std_filenames]
            cals = run_reduce(std_paths, f"std_{test_case}", cals)

        # Reducing science
        if "sci" in datasets[test_case]:
            sci_filenames = datasets[test_case]["sci"]
            sci_paths = [download_from_archive(f) for f in sci_filenames]
            for recipe_name in (None, "reduceSeparateCCDs",
                                "reduceSeparateCCDsCentral"):
                cals = run_reduce(
                    sci_paths, f"fringe_{test_case}", cals,
                    recipe_name='makeProcessedFringe', save_to="processed_fringe")
                run_reduce(sci_paths, f"sci_{test_case}", cals,
                           recipe_name=recipe_name,
                           user_pars=datasets[test_case]["ucals"])
                if not keep_data:
                    print(' Deleting pre-stack files.')
                    [os.remove(f) for f in glob.glob("*_CRMasked.fits")]
                    [os.remove(f) for f in glob.glob("*_align.fits")]
                suffix = recipe_name or "default"
                [os.rename(f, f.replace(".fits", f"_{suffix}.fits"))
                 for f in glob.glob("*.fits")]


# -- Helper functions ---------------------------------------------------------
def run_reduce(file_list, label, calib_files, recipe_name=None, save_to=None,
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
    #objgraph = pytest.importorskip("objgraph")

    logutils.get_logger().info("\n\n\n")
    logutils.config(file_name=f"test_image_{label}.log")
    r = Reduce()
    r.files = file_list
    r.ucals = normalize_ucals(calib_files)
    r.uparms = user_pars

    if recipe_name:
        r.recipename = recipe_name

    r.runr()

    if save_to:
        calib_files.append("{}:{}".format(
            save_to, os.path.join("calibrations", save_to, r._output_filenames[0])))
        [os.remove(f) for f in r._output_filenames]

    # check that we are not leaking objects
    #assert len(objgraph.by_type('NDAstroData')) == 0

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
