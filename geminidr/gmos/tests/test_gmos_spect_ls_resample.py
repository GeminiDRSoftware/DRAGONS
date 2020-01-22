"""
Regression tests for GMOS LS `resampleToCommonFrame`.
"""

import os

import numpy as np
import pytest

import geminidr
from geminidr.gmos import primitives_gmos_spect

# Test parameters -------------------------------------------------------------
test_datasets = [
    # Input Filename
    ("N20190427S0123_extracted.fits",
     "N20190427S0266_distortionDetermined.fits"),  # R400 525
    ("N20190427S0126_extracted.fits",
     "N20190427S0267_distortionDetermined.fits"),  # R400 625
    ("N20190427S0127_extracted.fits",
     "N20190427S0268_distortionDetermined.fits"),  # R400 725
]

ref_datasets = [
    "_".join(f[0].split("_")[:-1]) + "_extracted.fits"
    for f in test_datasets
]


@pytest.fixture(scope='module')
def refpath(path_to_refs):
    testpath = __name__.split('.')
    testpath.remove('tests')
    testpath = os.path.join(path_to_refs, *testpath)
    return testpath


# Local Fixtures and Helper Functions -----------------------------------------
@pytest.fixture(scope='module')
def ad(request, ad_factory, path_to_outputs, refpath):
    """
    Loads existing input FITS files as AstroData objects, runs the
    `extract1DSpectra` primitive on it, and return the output object containing
    the extracted 1d spectrum. This makes tests more efficient because the
    primitive is run only once, instead of N x Numbes of tests.

    If the input file does not exist, this fixture raises a IOError.

    If the input file does not exist and PyTest is called with the
    `--force-preprocess-data`, this fixture looks for cached raw data and
    process it. If the raw data does not exist, it is then cached via download
    from the Gemini Archive.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    ad_factory : fixture
        Custom fixture defined in the `conftest.py` file that loads cached
        data, or download and/or process it if needed.
    path_to_outputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to
        the output folder.

    Returns
    -------
    AstroData
        Object containing Wavelength Solution table.

    Raises
    ------
    IOError
        If the input file does not exist and if --force-preprocess-data
        is False.

    """
    print('\n\n Running test inside folder:\n  {}'.format(refpath))
    fname, arcname = request.param

    # create reduced arc
    arcfile = os.path.join(refpath, arcname)
    adarc = ad_factory(arcfile, preprocess_arc_recipe)
    if not os.path.exists(arcfile):
        adarc.write(arcfile)

    # create input for this test
    adfile = os.path.join(refpath, fname)
    ad = ad_factory(adfile, preprocess_recipe, arc=adarc)
    if not os.path.exists(adfile):
        ad.write(adfile)

    tests_failed_before_module = request.session.testsfailed
    yield ad

#     if request.session.testsfailed > tests_failed_before_module:
#         _dir = os.path.join(path_to_outputs, os.path.dirname(fname))
#         os.makedirs(_dir, exist_ok=True)

#         fname_out = os.path.join(_dir, ad_out.filename)
#         ad_out.write(filename=fname_out, overwrite=True)
#         print('\n Saved file to:\n  {}\n'.format(fname_out))


def preprocess_arc_recipe(ad, path):
    """
    Recipe used to generate arc data for `extract1DSpectra` tests.

    Parameters
    ----------
    ad : AstroData
        Input raw arc data loaded as AstroData.
    path : str
        Path that points to where the input data is cached.

    Returns
    -------
    AstroData
        Pre-processed arc data.

    """
    p = primitives_gmos_spect.GMOSSpect([ad])
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.mosaicDetectors()
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    ad = p.determineDistortion()[0]
    return ad


def preprocess_recipe(ad, path, arc):
    """Recipe used to generate input data.

    Parameters
    ----------
    ad : AstroData
        Input raw arc data loaded as AstroData.
    path : str
        Path that points to where the input data is cached.
    arc : AstroData
        Distortion corrected arc loaded as AstroData.

    Returns
    -------
    AstroData
        Pre-processed arc data.

    """
    p = primitives_gmos_spect.GMOSSpect([ad])
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.mosaicDetectors()
    p.distortionCorrect(arc=arc)
    p.findSourceApertures(max_apertures=1)
    p.skyCorrectFromSlit()
    p.traceApertures()
    ad = p.extract1DSpectra()[0]
    return ad


# Tests Definitions -----------------------------------------------------------
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, ad_ref", zip(test_datasets, ref_datasets),
                         indirect=True)
def test_resample_to_common_frame(ad, ad_ref):
    pass
#     assert ad[0].data.ndim == 1
#     np.testing.assert_allclose(ad[0].data, ad_ref[0].data, atol=1e-3)
