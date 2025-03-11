#!/usr/bin/env python3
"""
End-to-end integration tests for GNIRS Long-slit data reduction.
"""

import os

import numpy as np
import pytest

import astrodata
from astrodata.testing import download_from_archive
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals


# -- Datasets -----------------------------------------------------------------

datasets = {
    "GN-2012B-Q-56-40": {
        "arcs": ['N20121212S0231_arc.fits'],
        "flats": [f'N20121212S{i:04d}.fits' for i in range(233, 238)],
        "sci":  [f"N20121212S{i:04d}.fits" for i in range(242, 246)],
        "user_pars": {}
        },
    "GN-2017A-Q-87-284": {
        "arcs": ['N20170609S0136_arc.fits'],
        "flats": [f'N20170609S{i:04d}.fits' for i in range(131, 136)],
        "sci":  [f"N20170609S{i:04d}.fits" for i in range(127, 131)],
        "user_pars": {}
        },
    "GN-2017B-Q-85-216": {
        "arcs": ['N20180201S0065_arc.fits'],
        "flats": [f'N20180201S{i:04d}.fits' for i in range(60, 66)],
        "sci":  [f"N20180201S{i:04d}.fits" for i in range(52, 56)],
        "user_pars": {}
        },
    "GN-2017B-Q-81-99": {
        "arcs": ['N20180114S0121_arc.fits'],
        "flats": [f'N20180114S{i:04d}.fits' for i in range(125, 133)],
        "sci":  [f"N20180114S{i:04d}.fits" for i in range(121, 125)],
        "user_pars": {}
        },
    "GN-2019A-FT-207-46": {
        "arcs": ['N20190410S0212_arc.fits'],
        "flats": [f'N20190410S{i:04d}.fits' for i in range(220, 228)],
        "sci":  [f"N20190410S{i:04d}.fits" for i in range(212, 216)],
        "user_pars": {}
        },
    "GN-2020A-Q-227-36": {
        "arcs": ['N20200707S0097_arc.fits'],
        "flats": [f'N20200707S{i:04d}.fits' for i in range(101, 111)],
        "sci":  [f'N20200707S{i:04d}.fits' for i in range(97, 101)],
        "user_pars": {}
        }
    }

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsls
@pytest.mark.slow
@pytest.mark.integration_test
@pytest.mark.dragons_remote_data
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("test_case", datasets.keys())
def test_reduce_ls_spect(path_to_inputs, path_to_refs, change_working_dir,
                         keep_data, test_case):
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

        # Reducing flats
        flat_filenames = datasets[test_case]["flats"]
        flat_paths = [download_from_archive(f) for f in flat_filenames]
        cals = reduce(flat_paths, f"flat_{test_case}", cals, save_to="processed_flat")

        # Use a preprocessed arc since we might need to use one from sky lines
        arcs_filenames = datasets[test_case]["arcs"]
        arcs_paths = os.path.join(path_to_inputs, arcs_filenames[0])
        cals.append(f'processed_arc:{arcs_paths}')

        # Reducing science frames
        sci_filenames = datasets[test_case]["sci"]
        sci_paths = [download_from_archive(f) for f in sci_filenames]
        output = reduce(sci_paths, f"sci_{test_case}", cals,
                        user_pars=datasets[test_case]["user_pars"],
                        return_output=True)
        ad_out_2d = astrodata.open(output[0].replace("1D", "2D"))
        ad_out_1d = astrodata.open(output[0])
        ad_ref_1d = astrodata.open(os.path.join(path_to_refs, output[0]))

        # Check fewer than 4 apertures extracted
        assert len(ad_out_2d[0].APERTURE) < 4
        # Check first trace is basically vertical
        assert abs(ad_out_2d[0].APERTURE[0]["c1"]) < 0.5
        # Check that the WCS matches with the reference.
        astrodata.testing.ad_compare(ad_out_1d, ad_ref_1d,
                                     compare=["wcs"], rtol=1e-7)
        # Check that counts agree with the reference.
        np.testing.assert_allclose(ad_out_1d[0].data, ad_out_1d[0].data)

# -- Helper functions ---------------------------------------------------------
def reduce(file_list, label, calib_files, recipe_name=None, save_to=None,
           user_pars=None, return_output=False):
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

    if recipe_name:
        r.recipename = recipe_name

    r.runr()

    if save_to:
        [calib_files.append(
            "{}:{}".format(save_to, os.path.join("calibrations", save_to, f)))
            for f in r.output_filenames]
        [os.remove(f) for f in r.output_filenames]

    # check that we are not leaking objects
    assert len(objgraph.by_type('NDAstroData')) == 0

    if return_output:
        return r.output_filenames

    return calib_files


# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='function')
def gnirs_files(files):
    return [astrodata.open(download_from_archive(f) for f in files)]

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
