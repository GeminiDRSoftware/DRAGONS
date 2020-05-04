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
    "S20190808S0048.fits",  # R400 at 0.740 um
    "S20190808S0049.fits",  # R400 at 0.760 um
    "S20190808S0053.fits",  # R400 at 0.850 um
]

test_datasets2 = [
    "N20180106S0025.fits",  # B600 at 0.555 um
    "N20180106S0026.fits",  # B600 at 0.555 um
    "N20180106S0028.fits",  # B600 at 0.555 um
    "N20180106S0029.fits",  # B600 at 0.555 um
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
    ad = p.findSourceApertures()[0]
    assert len(ad[0].APERTURE) == 1
    assert caplog.records[3].message == 'Found sources at rows: 260.7'

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
    ad = p.findSourceApertures()[0]
    assert len(ad[0].APERTURE) == 1
    assert caplog.records[3].message == 'Found sources at rows: 260.4'

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
        ad.phu['QOFFSET'] += offset * i * ad.pixel_scale()


@pytest.fixture(scope='function')
def adinputs(request, get_input_ad):
    pre_process = request.config.getoption("--force-preprocess-data")
    adinputs = [get_input_ad(f, pre_process) for f in test_datasets]
    return adinputs


@pytest.fixture(scope='function')
def adinputs2(request, get_input_ad):
    pre_process = request.config.getoption("--force-preprocess-data")
    adinputs = [get_input_ad(f, pre_process) for f in test_datasets2]
    return adinputs


@pytest.fixture(scope='module')
def get_input_ad(cache_file_from_archive, path_to_inputs, reduce_arc, reduce_data):
    """
    Reads the input data or cache/process it in a temporary folder.

    Parameters
    ----------
    cache_file_from_archive : pytest.fixture
        Path to where the data will be temporarily cached.
    path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    reduce_arc : pytest.fixture
        Recipe used to reduce ARC data.
    reduce_data : pytest.fixture
        Recipe to reduce the data up to the step before
        `determineWavelengthSolution`.

    Returns
    -------
    function : factory that reads existing input data or call the data
        reduction recipe.
    """
    def _get_input_ad(basename, should_preprocess):
        input_fname = basename.replace('.fits', '_skyCorrected.fits')
        input_path = os.path.join(path_to_inputs, input_fname)

        if os.path.exists(input_path):
            print(" Load existing input file: {:s}".format(basename))
            input_data = astrodata.open(input_path)

        elif should_preprocess:
            print(" Caching input file: {:s}".format(basename))
            filename = cache_file_from_archive(basename)
            ad = astrodata.open(filename)
            cals = testing.get_associated_calibrations(basename)
            print(" Calibrations: ", cals)

            cals = [cache_file_from_archive(c)
                    for c in cals[cals.caltype.str.contains('arc')].filename.values]
            print(" Downloaded calibrations: {:s}".format("\n ".join(cals)))

            master_arc = reduce_arc(ad.data_label(), cals)
            input_data = reduce_data(ad, master_arc)

        else:
            raise IOError(
                'Could not find input file:\n' +
                '  {:s}\n'.format(input_path) +
                '  Run pytest with "--force-preprocess-data" to get it')

        return input_data
    return _get_input_ad


@pytest.fixture(scope='module')
def reduce_arc(change_working_dir):
    """ Recipe used to generate _distortionDetermined files from raw arc."""
    def _reduce_arc(dlabel, arc_fnames):
        with change_working_dir():
            # Use config to prevent duplicated outputs when running Reduce via API
            logutils.config(file_name='log_arc_{}.txt'.format(dlabel))

            p = primitives_gmos_spect.GMOSSpect(
                [astrodata.open(f) for f in arc_fnames])

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
    return _reduce_arc


@pytest.fixture(scope='module')
def reduce_data(change_working_dir):
    """Recipe used to generate _skyCorrected files from raw data. """
    def _reduce_data(ad, arc):
        with change_working_dir():
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
            ad = p.writeOutputs()[0]
        return ad
    return _reduce_data
