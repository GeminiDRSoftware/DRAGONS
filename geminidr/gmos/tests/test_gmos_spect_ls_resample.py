"""
Regression tests for GMOS LS `resampleToCommonFrame`.
"""

import os
import pytest

import astrodata
import gemini_instruments
import geminidr
from astrodata import testing
from geminidr.gmos import primitives_gmos_spect

# Test parameters -------------------------------------------------------------
# Each test input filename contains the original input filename with
# "_extracted" suffix.
test_datasets = [
    "S20190808S0048.fits",  # R400 at 0.740 um
    "S20190808S0049.fits",  # R400 at 0.760
    # "S20190808S0052.fits",  # R400 : 0.650
    "S20190808S0053.fits",  # R400 at 0.850
]


# Tests Definitions -----------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_and_linearize(input_ad_list, caplog):

    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
    ads = p.resampleToCommonFrame(dw=0.15)

    assert len(ads) == 3
    assert {len(ad) for ad in ads} == {1}
    assert {ad[0].shape[0] for ad in ads} == {3869}
    assert {'ALIGN' in ad[0].phu for ad in ads}
    _check_params(caplog.records, 'w1=508.198 w2=1088.323 dw=0.150 npix=3869')


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_and_linearize_with_w1_w2(input_ad_list, caplog):
    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
    p.resampleToCommonFrame(dw=0.15, w1=700, w2=850)
    _check_params(caplog.records, 'w1=700.000 w2=850.000 dw=0.150 npix=1001')


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_and_linearize_with_npix(input_ad_list, caplog):
    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
    p.resampleToCommonFrame(dw=0.15, w1=700, npix=1001)
    _check_params(caplog.records, 'w1=700.000 w2=850.000 dw=0.150 npix=1001')


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_error_with_all(input_ad_list, caplog):
    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
    expected_error = "Maximum 3 of w1, w2, dw, npix must be specified"
    with pytest.raises(ValueError, match=expected_error):
        p.resampleToCommonFrame(dw=0.15, w1=700, w2=850, npix=1001)


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_linearize_trim_and_stack(request, get_input_ad, input_ad_list, caplog):
    force_pre_process = request.config.getoption("--force-preprocess-data")
    list_of_input_ad = [get_input_ad(f, force_pre_process) for f in test_datasets]

    assert len(list_of_input_ad) == len(input_ad_list)

    fnames = [ad.filename for ad in input_ad_list]
    for ad in list_of_input_ad:
        assert ad.filename in fnames

    p = primitives_gmos_spect.GMOSSpect(list_of_input_ad)
    ads = p.resampleToCommonFrame(dw=0.15, trim_data=True)

    assert len(ads) == len(test_datasets)
    assert {ad[0].shape[0] for ad in ads} == {2429}
    _check_params(caplog.records, 'w1=614.666 w2=978.802 dw=0.150 npix=2429')

    adout = p.stackFrames()
    assert len(adout) == 1
    assert len(adout[0]) == 1
    assert adout[0][0].shape[0] == 2429


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_only(request, get_input_ad, caplog):
    force_pre_process = request.config.getoption("--force-preprocess-data")
    list_of_input_ad = [get_input_ad(f, force_pre_process) for f in test_datasets]

    p = primitives_gmos_spect.GMOSSpect(list_of_input_ad)
    p.resampleToCommonFrame()
    _check_params(caplog.records, 'w1=508.198 w2=1088.323 dw=0.151 npix=3841')

    caplog.clear()
    adout = p.resampleToCommonFrame(dw=0.15)
    assert 'ALIGN' in adout[0].phu
    _check_params(caplog.records, 'w1=508.198 w2=1088.232 dw=0.150 npix=3868')


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_only_and_trim(request, get_input_ad, caplog):
    force_pre_process = request.config.getoption("--force-preprocess-data")
    list_of_input_ad = [get_input_ad(f, force_pre_process) for f in test_datasets]

    p = primitives_gmos_spect.GMOSSpect(list_of_input_ad)
    p.resampleToCommonFrame(trim_data=True)
    _check_params(caplog.records, 'w1=614.666 w2=978.802 dw=0.151 npix=2407')

    caplog.clear()
    adout = p.resampleToCommonFrame(dw=0.15)
    assert 'ALIGN' in adout[0].phu
    _check_params(caplog.records, 'w1=614.574 w2=978.648 dw=0.150 npix=2429')


# Local Fixtures and Helper Functions -----------------------------------------
def _check_params(records, expected):
    for record in records:
        if record.message.startswith('Resampling and linearizing'):
            assert expected in record.message


@pytest.fixture(scope='module')
def input_ad_list(request, get_input_ad):
    pre_process = request.config.getoption("--force-preprocess-data")
    ad_list = [get_input_ad(f, pre_process) for f in test_datasets]
    print(ad_list)
    return ad_list


@pytest.fixture(scope='module')
def get_input_ad(cache_path, new_path_to_inputs, reduce_arc, reduce_data):
    """
    Reads the input data or cache-and-process it in a temporary folder.

    Parameters
    ----------
    cache_path : pytest.fixture
        Path to where the data will be temporarily cached.
    new_path_to_inputs : pytest.fixture
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
        input_fname = basename.replace('.fits', '_extracted.fits')
        input_path = os.path.join(new_path_to_inputs, input_fname)

        if should_preprocess:
            print(" Caching input file: {:s}".format(basename))
            filename = cache_path(basename)
            ad = astrodata.open(filename)
            cals = testing.get_associated_calibrations(basename)

            cals = [cache_path(c)
                    for c in cals[cals.caltype.str.contains('arc')].filename.values]
            print(" Downloaded calibrations: {:s}".format("\n ".join(cals)))

            master_arc = reduce_arc(ad.data_label(), cals)
            input_data = reduce_data(ad, master_arc)

        elif os.path.exists(input_path):
            input_data = astrodata.open(input_path)

        else:
            raise IOError(
                'Could not find input file:\n' +
                '  {:s}\n'.format(input_path) +
                '  Run pytest with "--force-preprocess-data" to get it')

        return input_data
    return _get_input_ad


@pytest.fixture(scope='module')
def reduce_data(output_path):
    """
    Creates a recipe that prepares the data for the tests. The recipe contains
    most of the early primitives of the `reduce` recipe for a science object.

    Parameters
    ----------
    output_path : pytest.fixture
        Fixture containing a custom context manager used to enter and leave the
        output folder easily.

    Returns
    -------
    function
        A recipe used to prepare the data for the tests.
    """
    def _reduce_data(ad, arc):
        with output_path():
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
            p.extract1DSpectra()
            ad = p.writeOutputs()[0]
        return ad
    return _reduce_data
