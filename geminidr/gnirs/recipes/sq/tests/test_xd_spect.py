#!/usr/bin/env python3
"""
End-to-end integration tests for GNIRS Cross-dispersed data reduction.
"""

import os

import numpy as np
import objgraph
import pytest

from astropy.coordinates import SkyCoord

import astrodata
from astrodata.testing import download_from_archive
from gempy.library import astromodels as am
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

# -- Datasets -----------------------------------------------------------------

datasets = {
    # 32 l/mm Shortblue SXD
    "GN-2021A-Q-215": {
        "arcs": ["N20210129S0324.fits"],
        "flats": [f"N20210129S{i:04d}.fits" for i in range(304, 324)],
        "pinholes": [f"N20210129S{i:04d}.fits" for i in (386, 388, 390, 391, 393)],
        "sci": [f"N20210129S{i:04d}.fits" for i in range(296, 304)],
        "user_pars": {"telluricCorrect:do_cal": "skip",
                      "fluxCalibrate:do_cal": "skip"}
    },
    # 10 l/mm Longblue SXD
    "GN-2013B-Q-41": {
        "arcs": ["N20130821S0301.fits"],
        "flats": [f"N20130821S{i:04d}.fits" for i in range(302, 318)],
        "pinholes": ["N20130821S0556.fits"],
        "sci": [f"N20130821S{i:04d}.fits" for i in range(322, 326)],
        "user_pars": {"telluricCorrect:do_cal": "skip",
                      "fluxCalibrate:do_cal": "skip"}
    },
    # 111 l/mm Shortblue SXD
    "GN-2024A-Q-115": {
        "arcs": ["N20240219S0228.fits"],
        "flats": [f"N20240219S{i:04d}.fits" for i in range(215, 227)],
        "pinholes": [f"N20240219S{i:04d}.fits" for i in range(288, 292)],
        "sci": [f"N20240219S{i:04d}.fits" for i in range(211, 215)],
        "user_pars": {"telluricCorrect:do_cal": "skip",
                      "fluxCalibrate:do_cal": "skip"}
    },
}

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsxd
@pytest.mark.slow
@pytest.mark.integration_test
@pytest.mark.dragons_remote_data
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("single_wave_scale", (False, True))
@pytest.mark.parametrize("test_case", datasets.keys())
def test_reduce_xd_spect(path_to_refs, change_working_dir,
                         keep_data, single_wave_scale, test_case):
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
        upars = datasets[test_case]["user_pars"]

        # Reducing flats
        flat_filenames = datasets[test_case]["flats"]
        flat_paths = [download_from_archive(f) for f in flat_filenames]
        cals = reduce(flat_paths, f"flat_{test_case}", cals, save_to="processed_flat")

        # Reduce pinholes
        pinhole_filenames = datasets[test_case]["pinholes"]
        if pinhole_filenames: # In case there are no pinholes for a dataset
            pinhole_paths = [download_from_archive(f) for f in pinhole_filenames]
            cals.extend(reduce(pinhole_paths, f"pinhole_{test_case}", cals,
                               save_to="processed_pinhole"))

        # Use a preprocessed arc since we might need to use one from sky lines
        # CJS: Except we don't, so let's not do this because it can cause the tests
        # to fail if there are other unrelated changes
        arcs_filenames = datasets[test_case]["arcs"]
        arcs_paths = [download_from_archive(f) for f in arcs_filenames]
        cals.extend(reduce(arcs_paths, f"arcs_{test_case}", cals,
                           user_pars=upars,
                           save_to="processed_arc"))

        # Reducing science frames
        sci_filenames = datasets[test_case]["sci"]
        sci_paths = [download_from_archive(f) for f in sci_filenames]
        upars = datasets[test_case]["user_pars"]
        upars['resampleToCommonFrame:single_wave_scale'] = single_wave_scale
        if single_wave_scale:
            upars['resampleToCommonFrame:output_wave_scale'] = 'loglinear'
        output = reduce(sci_paths, f"sci_{test_case}", cals,
                        user_pars=upars,
                        return_output=True)
        ad_out_2d = astrodata.open(output[0].replace("1D", "2D"))
        ad_out_1d = astrodata.open(output[0])

        arc = astrodata.open(cals[-1].split(":")[1])

        # Check for a sensible number of apertures
        assert len(ad_out_2d[0].APERTURE) < 3

        # Check that the pixel scale in the 2D image is correct (approx)
        # TODO: issue with non-square pixels for AO data
        if not ad_out_2d.is_ao():
            for ext in ad_out_2d:
                ymid, xmid = [0.5 * length for length in ext.shape]
                c1 = SkyCoord(*ext.wcs(xmid, ymid)[1:], unit="deg")
                c2 = SkyCoord(*ext.wcs(xmid+1, ymid)[1:], unit="deg")
                print(ext.id, c1.separation(c2).arcsec, ext.pixel_scale())
                assert c1.separation(c2).arcsec == pytest.approx(ext.pixel_scale(), rel=0.05)

        # Check that the wavelength scale in the 1D spectrum is correct
        arc = astrodata.open(cals[-1].split(":")[1])
        arc_waves = np.empty((len(arc), 2))
        for i, ext in enumerate(arc):
            wave = am.get_named_submodel(ext.wcs.forward_transform, "WAVE")
            arc_waves[i] = [wave(1021), wave(0)]
        for i, ext in enumerate(ad_out_1d):
            min_wave, max_wave = ext.wcs(0), ext.wcs(ext.shape[0]-1)
            print(i, min_wave, max_wave)
            if single_wave_scale:
                assert min_wave == pytest.approx(arc_waves[-1, 0], abs=1)
                assert max_wave == pytest.approx(arc_waves[0, 1], abs=1)
            else:
                assert min_wave == pytest.approx(arc_waves[i, 0], abs=1)
                assert max_wave == pytest.approx(arc_waves[i, 1], abs=1)

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
