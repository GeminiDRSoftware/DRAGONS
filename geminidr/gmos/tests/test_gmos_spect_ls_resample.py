"""
Regression tests for GMOS LS `resampleToCommonFrame`.
"""

import os

import numpy as np
import pytest

from geminidr.gmos import primitives_gmos_spect

# Test parameters -------------------------------------------------------------
test_datasets = [
    # Input Filename
    ("S20190808S0048_extracted.fits",              # R400 : 0.740
     "S20190808S0167_distortionDetermined.fits"),  #
    ("S20190808S0049_extracted.fits",              # R400 : 0.760
     "S20190808S0168_distortionDetermined.fits"),  #
    # ("S20190808S0052_extracted.fits",              # R400 : 0.650
    #  "S20190808S0165_distortionDetermined.fits"),  #
    ("S20190808S0053_extracted.fits",              # R400 : 0.850
     "S20190808S0169_distortionDetermined.fits"),  #
]


@pytest.fixture(scope='module')
def refpath(path_to_refs):
    testpath = __name__.split('.')
    testpath.remove('tests')
    testpath = os.path.join(path_to_refs, *testpath)
    return testpath


# Local Fixtures and Helper Functions -----------------------------------------
@pytest.fixture()
def adinputs(ad_factory, refpath):
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

    """
    print('\nRunning test inside folder:\n  {}'.format(refpath))
    adinputs = []

    for fname, arcname in test_datasets:
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
        adinputs.append(ad)

    print('')
    yield adinputs


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
def test_resample_and_linearize(adinputs):
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    adout = p.resampleToCommonFrame(dw=0.15)
    # we get 3 ad objects with one spectrum
    assert len(adout) == 3
    assert {len(ad) for ad in adout} == {1}
    assert {ad[0].shape[0] for ad in adout} == {3868}


@pytest.mark.preprocessed_data
def test_resample_linearize_trim_and_stack(adinputs):
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    adout = p.resampleToCommonFrame(dw=0.15, trim_data=True)
    # we get 3 ad objects with one spectrum
    assert len(adout) == 3
    assert {len(ad) for ad in adout} == {1}
    assert {ad[0].shape[0] for ad in adout} == {2428}

    adout = p.stackFrames()
    assert len(adout) == 1
    assert len(adout[0]) == 1
    assert adout[0][0].shape[0] == 2428
