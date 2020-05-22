"""
Regression tests for GMOS LS `resampleToCommonFrame`.
"""

import os
import pytest

import astrodata
import gemini_instruments

from geminidr.gmos import primitives_gmos_spect

# Test parameters -------------------------------------------------------------
test_datasets = [
    "S20190808S0048_extracted.fits",  # R400 at 0.740 um
    "S20190808S0049_extracted.fits",  # R400 at 0.760
    # "S20190808S0052_extracted.fits",  # R400 : 0.650  # Can't find aperture
    "S20190808S0053_extracted.fits",  # R400 at 0.850
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
def input_ad_list(path_to_inputs):
    """
    Reads the inputs data from disk as AstroData objects.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.

    Returns
    -------
    list of AstroData objects.
    """
    _input_ad_list = []

    for input_fname in test_datasets:
        input_path = os.path.join(path_to_inputs, input_fname)

        if os.path.exists(input_path):
            ad = astrodata.open(input_path)
        else:
            raise FileNotFoundError(input_path)

        _input_ad_list.append(ad)

    return _input_ad_list


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

    associated_calibrations = {
        "S20190808S0048.fits": {"arcs": ["S20190808S0167.fits"]},
        "S20190808S0049.fits": {"arcs": ["S20190808S0168.fits"]},
        # "S20190808S0052.fits": {"arcs": ["S20190808S0165.fits"]}, # Can't find aperture
        "S20190808S0053.fits": {"arcs": ["S20190808S0169.fits"]},
    }

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/test_gmos_spect_ls_resample/"
    path = os.path.join(root_path, module_path)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("./inputs", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename, cals in associated_calibrations.items():

        print('Downloading files...')
        sci_path = download_from_archive(filename)
        arc_path = [download_from_archive(f) for f in cals['arcs']]

        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing ARC for {:s}'.format(data_label))
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        arc_reduce = Reduce()
        arc_reduce.files.extend(arc_path)
        arc_reduce.runr()
        arc = arc_reduce.output_filenames.pop()

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
        p.traceApertures()
        p.extract1DSpectra()

        os.chdir("inputs/")
        _ = p.writeOutputs().pop()
        os.chdir("../")


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
