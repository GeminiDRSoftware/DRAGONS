"""
Regression tests for GMOS LS `resampleToCommonFrame`.
"""

import os
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


# Local Fixtures and Helper Functions -----------------------------------------
@pytest.fixture()
def adinputs(ad_factory, new_path_to_inputs):
    """ Generate input data for the tests if needed (and if
    --force-preprocess-data is set), using preprocess recipe below.
    """
    print('\nRunning test inside folder:\n  {}'.format(new_path_to_inputs))
    adinputs = []
    for fname, arcname in test_datasets:
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
    yield adinputs


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
    """Recipe used to generate _extracted files from raw data. """
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

def _check_params(records, expected):
    for record in records:
        if record.message.startswith('Resampling and linearizing'):
            assert expected in record.message


@pytest.mark.preprocessed_data
def test_resample_and_linearize(adinputs, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    adout = p.resampleToCommonFrame(dw=0.15)
    # we get 3 ad objects with one spectrum
    assert len(adout) == 3
    assert {len(ad) for ad in adout} == {1}
    assert {ad[0].shape[0] for ad in adout} == {3869}
    _check_params(caplog.records, 'w1=508.198 w2=1088.323 dw=0.150 npix=3869')
    assert 'ALIGN' in adout[0].phu


@pytest.mark.preprocessed_data
def test_resample_and_linearize_with_w1_w2(adinputs, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    p.resampleToCommonFrame(dw=0.15, w1=700, w2=850)
    _check_params(caplog.records, 'w1=700.000 w2=850.000 dw=0.150 npix=1001')


@pytest.mark.preprocessed_data
def test_resample_and_linearize_with_npix(adinputs, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    p.resampleToCommonFrame(dw=0.15, w1=700, npix=1001)
    _check_params(caplog.records, 'w1=700.000 w2=850.000 dw=0.150 npix=1001')


@pytest.mark.preprocessed_data
def test_resample_error_with_all(adinputs, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    expected_error = "Maximum 3 of w1, w2, dw, npix must be specified"
    with pytest.raises(ValueError, match=expected_error):
        p.resampleToCommonFrame(dw=0.15, w1=700, w2=850, npix=1001)


@pytest.mark.preprocessed_data
def test_resample_linearize_trim_and_stack(adinputs, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    adout = p.resampleToCommonFrame(dw=0.15, trim_data=True)
    # we get 3 ad objects with one spectrum
    assert len(adout) == 3
    assert {len(ad) for ad in adout} == {1}
    assert {ad[0].shape[0] for ad in adout} == {2429}
    _check_params(caplog.records, 'w1=614.666 w2=978.802 dw=0.150 npix=2429')

    adout = p.stackFrames()
    assert len(adout) == 1
    assert len(adout[0]) == 1
    assert adout[0][0].shape[0] == 2429


@pytest.mark.preprocessed_data
def test_resample_only(adinputs, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    p.resampleToCommonFrame()
    _check_params(caplog.records, 'w1=508.198 w2=1088.323 dw=0.151 npix=3841')
    caplog.clear()
    adout = p.resampleToCommonFrame(dw=0.15)
    assert 'ALIGN' in adout[0].phu
    _check_params(caplog.records, 'w1=508.198 w2=1088.232 dw=0.150 npix=3868')


@pytest.mark.preprocessed_data
def test_resample_only_and_trim(adinputs, caplog):
    p = primitives_gmos_spect.GMOSSpect(adinputs)
    p.resampleToCommonFrame(trim_data=True)
    _check_params(caplog.records, 'w1=614.666 w2=978.802 dw=0.151 npix=2407')
    caplog.clear()
    adout = p.resampleToCommonFrame(dw=0.15)
    assert 'ALIGN' in adout[0].phu
    _check_params(caplog.records, 'w1=614.574 w2=978.648 dw=0.150 npix=2429')
