#!/usr/bin/env python
"""
Tests related to GNIRS Long-slit Spectroscopy Arc primitives.

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

from geminidr.gnirs import primitives_gnirs_longslit
from gempy.utils import logutils
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from geminidr.gnirs.tests.longslit import CREATED_INPUTS_PATH_FOR_TESTS

# Test parameters --------------------------------------------------------------

associated_calibrations = {
    "N20211221S0094.fits": {              # GNIRS LS 111 l/mm LongCom, K-band
        'flat': ["N20211221S0095.fits"],
        'arcs': ["N20211221S0101.fits"],
    }
}

input_pars = [
    ("N20211221S0094_flatCorrected.fits", dict()),
]

sci_datasets = [key.replace('.fits', '_flatCorrected.fits')
                for key in associated_calibrations]
arc_datasets = [cals['arcs'][0].replace('.fits', '_arc.fits')
                for cals in associated_calibrations.values()]
datasets = [(key.replace('.fits', '_flatCorrected.fits'),
             cals['arcs'][0].replace('.fits', '_arc.fits'))
            for key, cals in associated_calibrations.items()]


def compare_frames(frame1, frame2):
    """Compare the important stuff of two CoordinateFrame instances"""
    for attr in ("naxes", "axes_type", "axes_order", "unit", "axes_names"):
        assert getattr(frame1, attr) == getattr(frame2, attr)


# Tests Definitions ------------------------------------------------------------
@pytest.mark.skip
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad, arc_ad", datasets, indirect=True)
def test_regression_in_distortion_correct(ad, arc_ad, change_working_dir, ref_ad_factory):
    """
    Compares the structure of WCS and pixel values of a frame that was flatCorrected,
    wavelengthSolutionAttached and distortion corrected using an associated arc,
    with one for which wavelength solution and distortion correction was done using
    a "processed_arc" created from the same science frame made using
    makeWavecalFromSkyAbsorption recipe.
    """
    with change_working_dir():

        logutils.config(
            file_name='log_regression_{:s}.txt'.format(ad.data_label()))

        p = primitives_gnirs_longslit.GNIRSLongslit([deepcopy(ad)])
        p.viewer = geminidr.dormantViewer(p, None)
        p.attachWavelengthSolution(arc=arc_ad)
        p.copyInputs(instream="main", outstream="with_distortion_model")
        p.distortionCorrect()
        p.findApertures()
        p.determineWavelengthSolution(absorption=True, min_snr=1)
        p.transferDistortionModel(source="with_distortion_model")
        processed_arc = p.writeOutputs()[0]

        p = primitives_gnirs_longslit.GNIRSLongslit([deepcopy(ad)])
        p.attachWavelengthSolution(arc=processed_arc)
        p.distortionCorrect()
        dist_corrected_ad = p.writeOutputs()[0]

    ref_ad = ref_ad_factory(dist_corrected_ad.filename)
    for ext, ext_ref in zip(dist_corrected_ad, ref_ad):
        data = np.ma.masked_invalid(ext.data)
        ref_data = np.ma.masked_invalid(ext_ref.data)
        np.testing.assert_allclose(data, ref_data, atol=1)
        # Compare output WCS as well as pixel values (by evaluating it at the
        # ends of the ranges, since there are multiple ways of constructing an
        # equivalent WCS, eg. depending on the order of various Shift models):
        for f in ext_ref.wcs.available_frames:
            compare_frames(getattr(ext_ref.wcs, f), getattr(ext.wcs, f))
        corner1, corner2 = (0, 0), tuple(v-1 for v in ext_ref.shape[::-1])
        world1, world2 = ext_ref.wcs(*corner1), ext_ref.wcs(*corner2)
        np.testing.assert_allclose(ext.wcs(*corner1), world1, rtol=1e-6)
        np.testing.assert_allclose(ext.wcs(*corner2), world2, rtol=1e-6)
        # The inverse is not highly accurate and transforming back & forth can
        # just exceed even a tolerance of 0.01 pix, but it's best to compare
        # with the true corner values rather than the same results from the
        # reference, otherwise it would be too easy to overlook a problem with
        # the inverse when someone checks the reference file.
        np.testing.assert_allclose(ext.wcs.invert(*world1),
                                   corner1, atol=0.02)
        np.testing.assert_allclose(ext.wcs.invert(*world2),
                                   corner2, atol=0.02)



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
        Input spectrum processed up to right before the `distortionCorrect`
        primitive.
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


# -- Recipe to create pre-processed data (no wave cal, just distortion) -------
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
    from geminidr.gnirs.tests.longslit import CREATED_INPUTS_PATH_FOR_TESTS
    from recipe_system.reduction.coreReduce import Reduce

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, cals in associated_calibrations.items():
        print(filename)
        # print(cals)

        sci_path = download_from_archive(filename)
        flat_paths = [download_from_archive(f) for f in cals['flat']]
        arc_paths = [download_from_archive(f) for f in cals['arcs']]

        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        logutils.config(file_name='log_flat_{}.txt'.format(data_label))
        flat_reduce = Reduce()
        flat_reduce.files.extend(flat_paths)
        flat_reduce.runr()
        processed_flat = flat_reduce.output_filenames.pop()
        del flat_reduce

        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        arc_reduce = Reduce()
        arc_reduce.files.extend(arc_paths)
        arc_reduce.runr()
        del arc_reduce

        logutils.config(file_name='log_sci_{}.txt'.format(data_label))
        p = primitives_gnirs_longslit.GNIRSLongslit([sci_ad])
        p.prepare()
        p.addDQ()
        p.ADUToElectrons()
        p.addVAR(read_noise=True, poisson_noise=True)
        p.flatCorrect(flat=processed_flat)
        os.chdir("inputs")
        processed_sci = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed sci to:\n'
              '    {:s}'.format(processed_sci.filename))

def create_refs_recipe():
    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("refs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))
    for sci, arc in zip(sci_datasets, arc_datasets):
        ad = astrodata.open(os.path.join('inputs', sci))
        p = GNIRSLongslit([ad])
        p.attachWavelengthSolution(arc=arc)
        p.distortionCorrect()
        os.chdir('refs/')
        p.writeOutputs()
        os.chdir('..')


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    if "--create-refs" in sys.argv[1:]:
        create_refs_recipe()
    else:
        pytest.main()
