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


# Local Fixtures and Helper Functions -----------------------------------------
def prepare_data(ad_factory, new_path_to_inputs, dataset):
    """ Generate input data for the tests if needed (and if
    --force-preprocess-data is set), using preprocess recipe below.
    """
    print('\nRunning test inside folder:\n  {}'.format(new_path_to_inputs))
    adinputs = []

    for fname, arcname in dataset:
        # create reduced arc
        arcfile = os.path.join(new_path_to_inputs, arcname)
        adarc = ad_factory(arcfile, preprocess_arc_recipe)
        if not os.path.exists(arcfile):
            adarc.write(arcfile)

        # create input for this test
        adfile = os.path.join(new_path_to_inputs, fname)
        ad = ad_factory(adfile, preprocess_recipe, arc=adarc)
        if not os.path.exists(adfile):
            ad.write(adfile)
        adinputs.append(ad)

    print('')
    return adinputs


@pytest.fixture()
def adinputs(ad_factory, new_path_to_inputs):
    yield prepare_data(ad_factory, new_path_to_inputs, test_datasets)


@pytest.fixture()
def adinputs2(ad_factory, new_path_to_inputs):
    yield prepare_data(ad_factory, new_path_to_inputs, test_datasets2)


def preprocess_arc_recipe(ad, path):
    """ Recipe used to generate _distortionDetermined files from raw arc."""
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
    """Recipe used to generate _skyCorrected files from raw data. """
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
        ad.phu['QOFFSET'] += 10 * i * ad.pixel_scale()

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
    assert np.isclose(adout[1].phu['SLITOFF'], -93.75)


@pytest.mark.preprocessed_data
def test_header_offset_fallback(adinputs2, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs2)
    adout = p.adjustSlitOffsetToReference()

    assert caplog.records[3].message.startswith(
        'WARNING - Offset from correlation (0) is too big compared to the '
        'header offset (-93.74999999999996). Using this one instead')
    assert np.isclose(adout[1].phu['SLITOFF'], -93.75)
