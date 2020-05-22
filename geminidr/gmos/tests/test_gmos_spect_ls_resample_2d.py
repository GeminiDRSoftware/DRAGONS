"""
Regression tests for GMOS LS `resampleToCommonFrame`.
"""

import numpy as np
import os
import pytest

import astrodata
import gemini_instruments
from astrodata import testing
from geminidr.gmos import primitives_gmos_spect
from gempy.utils import logutils

# Test parameters -------------------------------------------------------------
test_datasets = [
    "S20190808S0048_skyCorrected.fits",  # R400 at 0.740 um
    "S20190808S0049_skyCorrected.fits",  # R400 at 0.760 um
    "S20190808S0053_skyCorrected.fits",  # R400 at 0.850 um
]

test_datasets2 = [
    "N20180106S0025_skyCorrected.fits",  # B600 at 0.555 um
    "N20180106S0026_skyCorrected.fits",  # B600 at 0.555 um
    "N20180106S0028_skyCorrected.fits",  # B600 at 0.555 um
    "N20180106S0029_skyCorrected.fits",  # B600 at 0.555 um
]


# Tests Definitions -----------------------------------------------------------
@pytest.mark.gmosls
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
    ad = p.findSourceApertures(max_apertures=1)[0]
    assert len(ad[0].APERTURE) == 1
    assert caplog.records[3].message == 'Found sources at rows: 260.8'

    ad = p.extract1DSpectra()[0]
    assert ad[0].shape == (3869,)


@pytest.mark.gmosls
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
    ad = p.findSourceApertures(max_apertures=1)[0]
    assert len(ad[0].APERTURE) == 1
    assert caplog.records[3].message == 'Found sources at rows: 260.8'

    ad = p.extract1DSpectra()[0]
    assert ad[0].shape == (2429,)


@pytest.mark.gmosls
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


@pytest.mark.gmosls
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


@pytest.mark.gmosls
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


@pytest.mark.gmosls
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


# Local Fixtures and Helper Functions -----------------------------------------
def _check_params(records, expected):
    for record in records:
        if record.message.startswith('Resampling and linearizing'):
            assert expected in record.message


def add_fake_offset(adinputs, offset=10):
    # introduce fake offsets
    for i, ad in enumerate(adinputs[1:], start=1):
        ad[0].data = np.roll(ad[0].data, offset * i, axis=0)
        ad[0].mask = np.roll(ad[0].mask, offset * i, axis=0)
        ad[0].mask = np.roll(ad[0].variance, offset * i, axis=0)
        ad.phu['QOFFSET'] += offset * i * ad.pixel_scale()


@pytest.fixture(scope='function')
def adinputs(path_to_inputs):
    return [astrodata.open(os.path.join(path_to_inputs, f))
            for f in test_datasets]


@pytest.fixture(scope='function')
def adinputs2(path_to_inputs):
    return [astrodata.open(os.path.join(path_to_inputs, f))
            for f in test_datasets2]


# -- Recipe to create pre-processed data ---------------------------------------
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
    from recipe_system.reduction.coreReduce import Reduce
    from gempy.utils import logutils

    from astrodata.testing import get_associated_calibrations

    associated_calibrations = {
        "S20190808S0048.fits": 'S20190808S0167.fits',
        "S20190808S0049.fits": 'S20190808S0168.fits',
        "S20190808S0053.fits": 'S20190808S0169.fits',
        "N20180106S0025.fits": 'N20180115S0264.fits',
        "N20180106S0026.fits": 'N20180115S0264.fits',
        "N20180106S0028.fits": 'N20180115S0264.fits',
        "N20180106S0029.fits": 'N20180115S0264.fits',
    }

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/test_gmos_spect_ls_resample_2d/"
    path = os.path.join(root_path, module_path)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("./inputs", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for fname, arc_fname in associated_calibrations.items():

        sci_path = download_from_archive(fname)
        arc_path = download_from_archive(arc_fname)

        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing ARC for {:s}'.format(data_label))
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))

        if os.path.exists(arc_fname.replace('.fits', '_distortionDetermined.fits')):
            arc = astrodata.open(arc_fname.replace('.fits', '_distortionDetermined.fits'))
        else:
            p = primitives_gmos_spect.GMOSSpect([astrodata.open(arc_path)])
            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.mosaicDetectors()
            p.makeIRAFCompatible()
            p.determineWavelengthSolution()
            p.determineDistortion()
            arc = p.writeOutputs().pop()

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = primitives_gmos_spect.GMOSSpect([sci_ad])
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

        os.chdir("inputs/")
        _ = p.writeOutputs().pop()
        os.chdir("../")


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
