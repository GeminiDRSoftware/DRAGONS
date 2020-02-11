"""
Regression tests for GMOS LS `resampleToCommonFrame`.
"""

import numpy as np
import os
import pytest

from geminidr.gmos import primitives_gmos_spect

# Test parameters -------------------------------------------------------------
test_datasets = [
    ("S20190808S0048_skyCorrected.fits",           # R400 : 0.740
     "S20190808S0167_distortionDetermined.fits"),  #
    ("S20190808S0049_skyCorrected.fits",           # R400 : 0.760
     "S20190808S0168_distortionDetermined.fits"),  #
    ("S20190808S0053_skyCorrected.fits",           # R400 : 0.850
     "S20190808S0169_distortionDetermined.fits"),  #
]

test_datasets2 = [
    ("S20161022S0051_skyCorrected.fits",           # R400 : 0.750
     "S20161003S0280_distortionDetermined.fits"),  #
    ("S20161022S0052_skyCorrected.fits",           # R400 : 0.750
     "S20161003S0280_distortionDetermined.fits"),  #
]


@pytest.fixture(scope='module')
def refpath(path_to_refs):
    testpath = __name__.split('.')
    testpath.remove('tests')
    testpath = os.path.join(path_to_refs, *testpath)
    return testpath


# Local Fixtures and Helper Functions -----------------------------------------
def prepare_data(ad_factory, refpath, dataset):
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

    for fname, arcname in dataset:
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
    return adinputs


@pytest.fixture()
def adinputs(ad_factory, refpath):
    yield prepare_data(ad_factory, refpath, test_datasets)


@pytest.fixture()
def adinputs2(ad_factory, refpath):
    yield prepare_data(ad_factory, refpath, test_datasets2)


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
    ad = p.skyCorrectFromSlit()[0]
    return ad


# Tests Definitions -----------------------------------------------------------

def _check_params(records, expected):
    for record in records:
        if record.message.startswith('Resampling and linearizing'):
            assert expected in record.message


@pytest.mark.preprocessed_data
def test_correlation(adinputs, caplog):
    # introduce fake offsets
    for i, ad in enumerate(adinputs[1:], start=1):
        ad[0].data = np.roll(ad[0].data, 10 * i, axis=0)
        ad[0].mask = np.roll(ad[0].mask, 10 * i, axis=0)
        ad.phu['QOFFSET'] -= 10 * i * ad.pixel_scale()

    p = primitives_gmos_spect.GMOSSpect(adinputs)
    adout = p.adjustSlitOffsetToReference()

    assert adout[1].phu['SLITOFF'] == -10
    assert adout[2].phu['SLITOFF'] == -20

    # adout = p.resampleToCommonFrame(dw=0.15)
    # # we get 3 ad objects with one spectrum
    # assert len(adout) == 3
    # assert {len(ad) for ad in adout} == {1}
    # assert {ad[0].shape[0] for ad in adout} == {3868}
    # _check_params(caplog.records, 'w1=508.343 w2=1088.323 dw=0.150 npix=3868')
    # assert 'ALIGN' in adout[0].phu


@pytest.mark.preprocessed_data
def test_header_offset(adinputs2, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs2)
    adout = p.adjustSlitOffsetToReference(method='offsets')

    for rec in caplog.records:
        assert not rec.message.startswith('WARNING - Offset from correlation')
    assert np.isclose(adout[1].phu['SLITOFF'], 93.75)


@pytest.mark.preprocessed_data
def test_header_offset_fallback(adinputs2, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs2)
    adout = p.adjustSlitOffsetToReference()

    assert caplog.records[3].message.startswith(
        'WARNING - Offset from correlation (0) is too big compared to the '
        'header offset (93.74999999999996). Using this one instead')
    assert np.isclose(adout[1].phu['SLITOFF'], 93.75)
