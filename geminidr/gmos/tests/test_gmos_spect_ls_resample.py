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


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_and_linearize_with_w1_w2(input_ad_list, caplog):
    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
    p.resampleToCommonFrame(dw=0.15, w1=700, w2=850)
    _check_params(caplog.records, 'w1=700.000 w2=850.000 dw=0.150 npix=1001')


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_and_linearize_with_npix(input_ad_list, caplog):
    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
    p.resampleToCommonFrame(dw=0.15, w1=700, npix=1001)
    _check_params(caplog.records, 'w1=700.000 w2=850.000 dw=0.150 npix=1001')


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_error_with_all(input_ad_list, caplog):
    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
    expected_error = "Maximum 3 of w1, w2, dw, npix must be specified"
    with pytest.raises(ValueError, match=expected_error):
        p.resampleToCommonFrame(dw=0.15, w1=700, w2=850, npix=1001)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_linearize_trim_and_stack(input_ad_list, caplog):
    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
    ads = p.resampleToCommonFrame(dw=0.15, trim_data=True)

    assert len(ads) == len(test_datasets)
    assert {ad[0].shape[0] for ad in ads} == {2429}
    _check_params(caplog.records, 'w1=614.666 w2=978.802 dw=0.150 npix=2429')

    adout = p.stackFrames()
    assert len(adout) == 1
    assert len(adout[0]) == 1
    assert adout[0][0].shape[0] == 2429


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_only(input_ad_list, caplog):
    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
    p.resampleToCommonFrame()
    _check_params(caplog.records, 'w1=508.198 w2=1088.323 dw=0.151 npix=3841')

    caplog.clear()
    adout = p.resampleToCommonFrame(dw=0.15)
    assert 'ALIGN' in adout[0].phu
    _check_params(caplog.records, 'w1=508.198 w2=1088.232 dw=0.150 npix=3868')


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resample_only_and_trim(input_ad_list, caplog):
    p = primitives_gmos_spect.GMOSSpect(input_ad_list)
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


@pytest.fixture(scope='function')
def input_ad_list(cache_file_from_archive, path_to_inputs, reduce_arc,
                  reduce_data, request):
    """
    Reads the input data or cache-and-process it in a temporary folder.

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
    request : pytest.fixture
        Pytest's built-in fixture containing information about parent function.

    Returns
    -------
    list
        Containing the input data as a list of AstroData objects.
    """
    should_preprocess = request.config.getoption("--force-preprocess-data")
    ad_list = []

    for basename in test_datasets:
        input_fname = basename.replace('.fits', '_extracted.fits')
        input_path = os.path.join(path_to_inputs, input_fname)

        if os.path.exists(input_path):
            input_data = astrodata.open(input_path)

        elif should_preprocess:
            print(" Caching input file: {:s}".format(basename))
            filename = cache_file_from_archive(basename)
            ad = astrodata.open(filename)
            cals = testing.get_associated_calibrations(basename)

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

        ad_list.append(input_data)
    return ad_list


@pytest.fixture(scope='module')
def reduce_data(change_working_dir):
    """
    Creates a recipe that prepares the data for the tests. The recipe contains
    most of the early primitives of the `reduce` recipe for a science object.

    Parameters
    ----------
    change_working_dir : pytest.fixture
        Fixture containing a custom context manager used to enter and leave the
        output folder easily.

    Returns
    -------
    function
        A recipe used to prepare the data for the tests.
    """
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
            p.traceApertures()
            p.extract1DSpectra()
            ad = p.writeOutputs()[0]
        return ad
    return _reduce_data
