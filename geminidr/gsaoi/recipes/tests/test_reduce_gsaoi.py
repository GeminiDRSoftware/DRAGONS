#!/usr/bin/env python
import os
import pytest

from astrodata.testing import download_from_archive
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

datasets = {

    "GS-2017A-Q-29-13": {
        "science": [f"S20170505S{i:04d}.fits" for i in range(95, 98)],
        "dflat_on": [f"S20170505S{i:04d}.fits" for i in range(30, 33)],
        "dflat_off": [f"S20170505S{i:04d}.fits" for i in range(72, 75)],
        "dark": [f"S20150609S{i:04d}.fits" for i in range(22, 25)],
        "hflat": [f"S20171208S{i:04d}.fits" for i in range(53, 56)],
        "user_pars": [],
    }

}


@pytest.mark.slow
@pytest.mark.integration_test
@pytest.mark.gsaoi
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("test_case", datasets.keys())
def test_reduce_image(change_working_dir, test_case):
    """
    Test that DRAGONS can reduce GSAOI data.

    Parameters
    ----------
    change_working_dir : fixture
        Allows easy folder manipulation.
    test_case : str
        Group ID related to the input data.
    """
    with change_working_dir(test_case):

        cals = []

        # Reducing dark -----
        dark_filenames = datasets[test_case]["dark"]
        dark_paths = [download_from_archive(f) for f in dark_filenames]
        _, cals = reduce(dark_paths, f"dark_{test_case}", cals,
                         save_to="processed_dark")

        # Create BPM -----
        hflat_filenames = datasets[test_case]["hflat"] + datasets[test_case]["dark"]
        hflat_paths = [download_from_archive(f) for f in hflat_filenames]
        bpm_filename, cals = reduce(hflat_paths, f"bpm_{test_case}", cals,
                                    recipe_name='makeProcessedBPM')

        # Reduce flats -----
        flat_filenames = (datasets[test_case]["dflat_on"] +
                          datasets[test_case]["dflat_on"])
        flat_paths = [download_from_archive(f) for f in flat_filenames]
        _, cals = reduce(flat_paths, f"flat_{test_case}", cals,
                         save_to="processed_flat")

        # Reduce Science ---
        datasets[test_case]["user_pars"].append(('addDQ:user_bpm', bpm_filename))
        sci_filenames = datasets[test_case]["science"]
        sci_paths = [download_from_archive(f) for f in sci_filenames]
        _, _ = reduce(sci_paths, f"sci_{test_case}", cals,
                      user_pars=datasets[test_case]["user_pars"])


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
    str : Output reduced file.
    list : An updated list of calibration files.
    """
    objgraph = pytest.importorskip("objgraph")

    logutils.get_logger().info("\n\n\n")
    logutils.config(file_name=f"test_image_{label}.log")
    r = Reduce()
    r.files = file_list
    r.ucals = normalize_ucals(calib_files)
    r.uparms = user_pars

    if recipe_name:
        r.recipename = recipe_name

    r.runr()
    output_file = r.output_filenames[0]

    if save_to:
        calib_files.append("{}:{}".format(
            save_to, os.path.join("calibrations", save_to, r.output_filenames[0])))
        [os.remove(f) for f in r.output_filenames]

    # check that we are not leaking objects
    assert len(objgraph.by_type('NDAstroData')) == 0, ("Leaking objects",
           [x.shape for x in objgraph.by_type('NDAstroData')],
           [x.filename for x in objgraph.by_type("AstroDataGsaoi")],
           f'{label=}')

    return output_file, calib_files


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
