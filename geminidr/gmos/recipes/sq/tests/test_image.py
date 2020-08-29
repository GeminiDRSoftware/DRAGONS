#!/usr/bin/env python

import glob
import pytest
import os

import astrodata
import gemini_instruments

from astrodata.testing import download_from_archive
from gempy.adlibrary import dataselect
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
        "ucals": [('stackFrames:memory', 1),
                  # ('addDQ:user_bpm', 'fixed_bpm_1x1_FullFrame.fits'),
                  ('adjustWCSToReference:rotate', True),
                  ('adjustWCSToReference:scale', True),
                  ('resampleToCommonFrame:interpolator', 'spline3')]
    },

    "GS_HAM_2x2_i-band_std": {
        "bias": [f"S20171204S{n:04d}.fits" for n in range(37, 42)],
        "flat": [f"S20171120S{n:04d}.fits" for n in range(131, 140)],
        "std": ["S20171205S0077.fits"],
        "ucals": [('stackFrames:memory', 1),
                  # ('addDQ:user_bpm', 'fixed_bpm_2x2_FullFrame.fits'),
                  ('resampleToCommonFrame:interpolator', 'spline3')]
    },

}


# ToDo - @bquint - Perform clean up after running tests
@pytest.mark.skip("Using too much space - enable clean up")
@pytest.mark.integtest
@pytest.mark.parametrize("test_case", datasets.keys())
def test_reduce_image(change_working_dir, test_case):
    """
    Tests that we can run all the data reduction steps on a complete dataset.

    Parameters
    ----------
    change_working_dir : fixture
    test_case : str
    """
    with change_working_dir(test_case):

        cals = []

        # Reducing bias
        bias_filenames = datasets[test_case]["bias"]
        bias_paths = [download_from_archive(f) for f in bias_filenames]
        cals = reduce(bias_paths, f"bias_{test_case}", cals, save_to="processed_bias")

        # Reducing flats
        flat_filenames = datasets[test_case]["flat"]
        flat_paths = [download_from_archive(f) for f in flat_filenames]
        cals = reduce(flat_paths, f"flat_{test_case}", cals, save_to="processed_flat")

        # Reducing standard stars
        if "std" in datasets[test_case]:
            std_filenames = datasets[test_case]["std"]
            std_paths = [download_from_archive(f) for f in std_filenames]
            cals = reduce(std_paths, f"std_{test_case}", cals)

        # Reducing science
        if "sci" in datasets[test_case]:
            sci_filenames = datasets[test_case]["sci"]
            sci_paths = [download_from_archive(f) for f in sci_filenames]
            cals = reduce(
                sci_paths, f"fringe_{test_case}", cals,
                recipe_name='makeProcessedFringe', save_to="processed_fringe")
            _ = reduce(
                sci_paths, f"sci_{test_case}", cals,
                user_pars=datasets[test_case]["ucals"])


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

    logutils.config(file_name=f"test_image_{label}.log")
    r = Reduce()
    r.files = file_list
    r.ucals = normalize_ucals(r.files, calib_files)
    r.uparms = user_pars

    if recipe_name:
        r.recipename = recipe_name

    r.runr()

    if save_to:
        calib_files.append(f'{save_to}:{r.output_filenames[0]}')

    # check that we are not leaking objects
    assert len(objgraph.by_type('NDAstroData')) == 0

    return calib_files


if __name__ == '__main__':
    pytest.main()
