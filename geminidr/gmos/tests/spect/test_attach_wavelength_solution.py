#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` and `ad_ref` fixtures to be called and the AstroData object returned.
"""
import numpy as np
import os
import pytest
from copy import deepcopy

import astrodata
import geminidr

from geminidr.gmos import primitives_gmos_longslit
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory

# Test parameters --------------------------------------------------------------
# A more scientifically-complete reduction of some of the same data is done by
# test_distortion_correct & test_distortion_correct_with_wavelength_solution.
associated_calibrations = {
    "N20100115S0312.fits": {              # GN B600:0.500 EEV
        'arc': "N20100115S0346.fits",
    },
    "S20200116S0104.fits": {              # GS R400:0.850 HAM CS
        'arc': "S20200116S0357.fits",
    },
    "N20211008S0368.fits": {              # GN B600:0.495 HAM CS+FULL
        'arc': "N20211015S0382.fits",
    #   'arc': "N20211008S0408.fits",  # Alternative CS arc
    },
}
datasets = [(key.replace('.fits', '_varAdded.fits'),
             cals['arc'].replace('.fits', '_arc.fits'))
            for key, cals in associated_calibrations.items()]

fixed_test_parameters_for_determine_distortion = {
    "fwidth": None,
    "id_only": False,
    "max_missed": 5,
    "max_shift": 0.05,
    "min_snr": 5.,
    "nsum": 10,
    "spatial_order": 3,
    "spectral_order": 4,
}


def compare_frames(frame1, frame2):
    """Compare the important stuff of two CoordinateFrame instances"""
    for attr in ("naxes", "axes_type", "axes_order", "unit", "axes_names"):
        assert getattr(frame1, attr) == getattr(frame2, attr)


# Tests Definitions ------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad, arc_ad", datasets, indirect=True)
def test_regression_in_attach_wavelength_solution(ad, arc_ad, change_working_dir, ref_ad_factory):
    """
    Runs the `attachWavelengthSolution` primitive on preprocessed data and
    compares its WCS with the one in the reference file.
    """
    with change_working_dir():

        logutils.config(
            file_name='log_regression_{:s}.txt'.format(ad.data_label()))

        p = primitives_gmos_longslit.GMOSLongslit([deepcopy(ad)])
        p.viewer = geminidr.dormantViewer(p, None)
        p.attachWavelengthSolution(arc=arc_ad)
        calibrated_ad = p.writeOutputs()[0]

    ref_ad = ref_ad_factory(calibrated_ad.filename)
    for ext, ext_ref in zip(calibrated_ad, ref_ad):
        # Compare the output WCS, making sure it includes the same frames and
        # produces the same World co-ordinates. Don't require it to be
        # structured identically to the reference, because there's more than
        # one equivalent way that attachWavelengthSolution could assemble
        # different WCS components with origin shifts; also, it's unhelpful
        # for the comparison to be brittle if the input files later need
        # regenerating with a differently-structured but equivalent wavelength
        # solution, when the same references could possibly be re-used.
        for f in ext_ref.wcs.available_frames:
            compare_frames(getattr(ext_ref.wcs, f), getattr(ext.wcs, f))
        idx = np.meshgrid(*(np.arange(dim) for dim in reversed(ext_ref.shape)))
        world, world_ref = ext.wcs(*idx), ext_ref.wcs(*idx)
        # Require roughly 0.1 pix precision in wavelength & <0.1" spatially:
        np.testing.assert_allclose(world[0], world_ref[0], atol=0.005)
        np.testing.assert_allclose(world[1:3], world_ref[1:3], atol=1e-5)
        inv, inv_ref = ext.wcs.invert(*world), ext_ref.wcs.invert(*world_ref)
        # np.testing.assert_allclose(inv, inv_ref, atol=0.05)
        np.testing.assert_allclose(inv, idx, atol=0.05)

@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad, arc_ad", datasets, indirect=True)
def test_regression_in_attach_wavelength_solution_to_mosaic(ad, arc_ad, change_working_dir, ref_ad_factory):
    """
    Runs the `attachWavelengthSolution` primitive on preprocessed, mosaicked
    data and compares its WCS with the one in the reference file.
    """
    with change_working_dir():

        logutils.config(
            file_name='log_regression_{:s}.txt'.format(ad.data_label()))

        if ad.detector_roi_setting() != arc_ad.detector_roi_setting():
            pytest.skip('Can\'t use arc of different ROI for mosaicked data')

        p = primitives_gmos_longslit.GMOSLongslit([deepcopy(ad)])
        p.viewer = geminidr.dormantViewer(p, None)
        p.mosaicDetectors()
        p.attachWavelengthSolution(arc=arc_ad)
        calibrated_ad = p.writeOutputs(
            strip=True, suffix="_mosaic_wavelengthSolutionAttached"
        )[0]

    ref_ad = ref_ad_factory(calibrated_ad.filename)
    for ext, ext_ref in zip(calibrated_ad, ref_ad):
        # Do the same comparison as in the above test on the mosicked data:
        for f in ext_ref.wcs.available_frames:
            compare_frames(getattr(ext_ref.wcs, f), getattr(ext.wcs, f))
        idx = np.meshgrid(*(np.arange(dim) for dim in reversed(ext_ref.shape)))
        world, world_ref = ext.wcs(*idx), ext_ref.wcs(*idx)
        np.testing.assert_allclose(world[0], world_ref[0], atol=0.005)
        np.testing.assert_allclose(world[1:3], world_ref[1:3], atol=1e-5)
        inv, inv_ref = ext.wcs.invert(*world), ext_ref.wcs.invert(*world_ref)
        np.testing.assert_allclose(inv, idx, atol=0.05)


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
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
        Input spectrum processed up to right before the
        `attachWavelengthSolution` primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad


@pytest.fixture(scope='function')
def arc_ad(path_to_inputs, request):
    """
    Returns the master arc used during the data-set data reduction.

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
        Master arc.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        print(f"Reading input arc: {path}")
        arc_ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return arc_ad


# -- Recipe to create pre-processed data, including wave cal ------------------
def create_inputs_recipe():
    """
    Creates input data for tests using pre-processed standard star and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """
    import os
    from astrodata.testing import download_from_archive
    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS
    from recipe_system.reduction.coreReduce import Reduce
    from recipe_system.utils.reduce_utils import normalize_ucals

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    # This wants refactoring a bit so there's less duplication WRT QECorrect
    # and any variants of the distortionCorrect tests.
    for filename, cals in associated_calibrations.items():
        print(filename)
        # print(cals)

        sci_path = download_from_archive(filename)
        arc_path = download_from_archive(cals['arc'])

        sci_ad = astrodata.open(sci_path)
        arc_ad = astrodata.open(arc_path)
        data_label = sci_ad.data_label()

        logutils.config(file_name='log_arc_{}.txt'.format(data_label))

        p = primitives_gmos_longslit.GMOSLongslit([arc_ad])
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True, poisson_noise=False)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True, read_noise=False)
        p.mosaicDetectors()
        p.makeIRAFCompatible()
        p.determineWavelengthSolution(order=4., lsigma=2.5, hsigma=2.5,
                                      min_sep=10., interactive=False)
        p.determineDistortion(**fixed_test_parameters_for_determine_distortion)
        p.storeProcessedArc()

        os.chdir("inputs")
        processed_arc = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed arc to:\n'
              '    {:s}'.format(processed_arc.filename))

        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = primitives_gmos_longslit.GMOSLongslit([sci_ad])
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        # p.biasCorrect(bias=bias_master)  # inc. in test_distortion_correct
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)

        os.chdir("inputs")
        processed_sci = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed sci to:\n'
              '    {:s}'.format(processed_sci.filename))


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
