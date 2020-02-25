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
    ("N20180106S0025_skyCorrected.fits",           # B600 : 0.555
     "N20180115S0264_distortionDetermined.fits"),  #
    ("N20180106S0026_skyCorrected.fits",           # B600 : 0.555
     "N20180115S0264_distortionDetermined.fits"),  #
    ("N20180106S0028_skyCorrected.fits",           # B600 : 0.555
     "N20180115S0264_distortionDetermined.fits"),  #
    ("N20180106S0029_skyCorrected.fits",           # B600 : 0.555
     "N20180115S0264_distortionDetermined.fits"),  #
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


def add_fake_offset(adinputs, offset=10):
    # introduce fake offsets
    for i, ad in enumerate(adinputs[1:], start=1):
        ad[0].data = np.roll(ad[0].data, offset * i, axis=0)
        ad[0].mask = np.roll(ad[0].mask, offset * i, axis=0)
        ad.phu['QOFFSET'] += offset * i * ad.pixel_scale()


@pytest.mark.preprocessed_data
def test_correlation(adinputs, caplog):
    add_fake_offset(adinputs, offset=10)
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    adout = p.adjustSlitOffsetToReference()

    assert adout[1].phu['SLITOFF'] == -10
    assert adout[2].phu['SLITOFF'] == -20

    p.resampleToCommonFrame(dw=0.15)
    _check_params(caplog.records, 'w1=508.198 w2=1088.323 dw=0.150 npix=3869')

    ad = p.stackFrames()[0]
    assert ad[0].shape == (512, 3869)

    caplog.clear()
    ad = p.findSourceApertures()[0]
    assert len(ad[0].APERTURE) == 1
    assert caplog.records[3].message == 'Found sources at rows: 260.7'

    ad = p.extract1DSpectra()[0]
    assert ad[0].shape == (3869,)


@pytest.mark.preprocessed_data
def test_correlation_and_trim(adinputs, caplog):
    add_fake_offset(adinputs, offset=10)
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    adout = p.adjustSlitOffsetToReference()

    assert adout[1].phu['SLITOFF'] == -10
    assert adout[2].phu['SLITOFF'] == -20

    p.resampleToCommonFrame(dw=0.15, trim_data=True)
    _check_params(caplog.records, 'w1=614.666 w2=978.802 dw=0.150 npix=2429')

    ad = p.stackFrames()[0]
    assert ad[0].shape == (512, 2429)

    caplog.clear()
    ad = p.findSourceApertures()[0]
    assert len(ad[0].APERTURE) == 1
    assert caplog.records[3].message == 'Found sources at rows: 260.4'

    ad = p.extract1DSpectra()[0]
    assert ad[0].shape == (2429,)


@pytest.mark.preprocessed_data
def test_correlation_and_w1_w2(adinputs, caplog):
    add_fake_offset(adinputs, offset=10)
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    adout = p.adjustSlitOffsetToReference()

    assert adout[1].phu['SLITOFF'] == -10
    assert adout[2].phu['SLITOFF'] == -20

    p.resampleToCommonFrame(dw=0.15, w1=700, w2=850)
    _check_params(caplog.records, 'w1=700.000 w2=850.000 dw=0.150 npix=1001')

    adstack = p.stackFrames()
    assert adstack[0][0].shape == (512, 1001)


@pytest.mark.preprocessed_data
def test_correlation_non_linearize(adinputs, caplog):
    add_fake_offset(adinputs, offset=10)
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    adout = p.adjustSlitOffsetToReference()

    assert adout[1].phu['SLITOFF'] == -10
    assert adout[2].phu['SLITOFF'] == -20

    p.resampleToCommonFrame()
    _check_params(caplog.records, 'w1=508.198 w2=1088.323 dw=0.151 npix=3841')
    caplog.clear()
    adout = p.resampleToCommonFrame(dw=0.15)
    assert 'ALIGN' in adout[0].phu
    _check_params(caplog.records, 'w1=508.198 w2=1088.232 dw=0.150 npix=3868')

    adstack = p.stackFrames()
    assert adstack[0][0].shape == (512, 3868)


@pytest.mark.preprocessed_data
def test_header_offset(adinputs2, caplog):
    """Test that the offset is correctly read from the headers."""
    p = primitives_gmos_spect.GMOSSpect(adinputs2)
    adout = p.adjustSlitOffsetToReference(method='offsets')

    for rec in caplog.records:
        assert not rec.message.startswith('WARNING - Offset from correlation')

    assert np.isclose(adout[0].phu['SLITOFF'], 0)
    assert np.isclose(adout[1].phu['SLITOFF'], -92.9368)
    assert np.isclose(adout[2].phu['SLITOFF'], -92.9368)
    assert np.isclose(adout[3].phu['SLITOFF'], 0)


@pytest.mark.preprocessed_data
def test_header_offset_fallback(adinputs2, caplog):
    """For this dataset the correlation method fails, and give an offset very
    different from the header one. So we check that the fallback to the header
    offset works.
    """
    p = primitives_gmos_spect.GMOSSpect(adinputs2)
    adout = p.adjustSlitOffsetToReference()

    assert caplog.records[3].message.startswith(
        'WARNING - Offset from correlation (0) is too big compared to the '
        'header offset (-92.93680297397756). Using this one instead')

    assert np.isclose(adout[0].phu['SLITOFF'], 0)
    assert np.isclose(adout[1].phu['SLITOFF'], -92.9368)
    assert np.isclose(adout[2].phu['SLITOFF'], -92.9368)
    assert np.isclose(adout[3].phu['SLITOFF'], 0)
