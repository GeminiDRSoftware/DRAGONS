"""
Regression tests for GMOS LS `resampleToCommonFrame`.
"""

import numpy as np
import os
import pytest

import astrodata
import gemini_instruments
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
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
@pytest.mark.parametrize("offset", (5.5, 10))
def test_simple_correlation_test(path_to_inputs, offset):
    """A simple correlation test that uses a single image, shifted, to avoid
    difficulties in centroiding. Placed here because it uses datasets and
    functions in this module"""
    adinputs = [astrodata.from_file(os.path.join(path_to_inputs, test_datasets[0]))
                for i in (0, 1, 2)]
    add_fake_offset(adinputs, offset=offset)
    p = GMOSLongslit(adinputs)
    p.findApertures(max_apertures=1)
    center = p.streams['main'][0][0].APERTURE['c0']
    p.adjustWCSToReference(method='offsets')
    p.resampleToCommonFrame(dw=0.15)
    p.findApertures(max_apertures=1)
    for ad in p.streams['main']:
        assert abs(ad[0].APERTURE['c0'] - center) < 0.1


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("offset", (5.5, 10))
def test_resampling(adinputs, caplog, offset):
    add_fake_offset(adinputs, offset=offset)
    p = GMOSLongslit(adinputs)
    adout = p.adjustWCSToReference(method='offsets')

    assert abs(adout[1].phu['SLITOFF'] + offset) < 0.001
    assert abs(adout[2].phu['SLITOFF'] + 2 * offset) < 0.001

    p.resampleToCommonFrame(dw=0.15)
    _check_params(caplog.records, 'w1=508.343 w2=1088.393 dw=0.150 npix=3868')
    assert all(ad[0].shape == (int(512 - 2*offset), 3868) for ad in p.streams['main'])


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("offset", (5.5, 10))
def test_resampling_and_trim(adinputs, caplog, offset):
    add_fake_offset(adinputs, offset=offset)
    p = GMOSLongslit(adinputs)
    adout = p.adjustWCSToReference(method='offsets')

    assert abs(adout[1].phu['SLITOFF'] + offset) < 0.001
    assert abs(adout[2].phu['SLITOFF'] + 2 * offset) < 0.001

    p.resampleToCommonFrame(dw=0.15, trim_spectral=True)
    _check_params(caplog.records, 'w1=614.812 w2=978.862 dw=0.150 npix=2428')
    for ad in p.streams['main']:
        assert ad[0].shape == (int(512 - 2*offset), 2428)

    # Location of apertures is not important for this test
    #p.findApertures(max_apertures=1)
    #np.testing.assert_allclose([ad[0].APERTURE['c0']
    #                            for ad in p.streams['main']], 260.4, atol=0.25)

    ad = p.stackFrames(reject_method="sigclip")[0]
    #assert ad[0].shape == (492, 3139)  # checked above

    caplog.clear()
    ad = p.findApertures(max_apertures=1)[0]
    assert len(ad[0].APERTURE) == 1
    #np.testing.assert_allclose(ad[0].APERTURE['c0'], 260.4, atol=0.25)

    ad = p.extractSpectra()[0]
    assert ad[0].shape == (2428,)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resampling_and_w1_w2(adinputs, caplog):
    add_fake_offset(adinputs, offset=10)
    p = GMOSLongslit(adinputs)
    adout = p.adjustWCSToReference(method='offsets')

    assert adout[1].phu['SLITOFF'] == -10
    assert adout[2].phu['SLITOFF'] == -20

    p.resampleToCommonFrame(dw=0.15, w1=700, w2=850)
    _check_params(caplog.records, 'w1=700.000 w2=850.000 dw=0.150 npix=1001')

    adstack = p.stackFrames()
    assert adstack[0][0].shape == (492, 1001)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_resampling_non_linearize(adinputs, caplog):
    add_fake_offset(adinputs, offset=10)
    p = GMOSLongslit(adinputs)
    adout = p.adjustWCSToReference(method='offsets')

    assert adout[1].phu['SLITOFF'] == -10
    assert adout[2].phu['SLITOFF'] == -20

    p.resampleToCommonFrame(output_wave_scale="reference", trim_spectral=True)
    _check_params(caplog.records, 'w1=614.870 w2=978.802 dw=0.151 npix=2407')
    caplog.clear()
    adout = p.resampleToCommonFrame(dw=0.15)
    assert 'ALIGN' in adout[0].phu
    _check_params(caplog.records, 'w1=614.870 w2=978.920 dw=0.150 npix=2428')

    adstack = p.stackFrames()
    assert adstack[0][0].shape == (492, 2428)


# Local Fixtures and Helper Functions -----------------------------------------
def _check_params(records, expected):
    assert len(records) > 0  # make sure caplog is capturing something!
    for record in records:
        if record.message.startswith('Resampling and linearizing'):
            assert expected in record.message


def add_fake_offset(adinputs, offset=10):
    # introduce fake offsets
    # CJS hack so this can handle fractional offsets by linear interpolation
    for i, ad in enumerate(adinputs[1:], start=1):
        int_offset =int(np.floor(i * offset))
        frac_offset = i * offset - int_offset
        for attr in ('data', 'mask', 'variance'):
            setattr(ad[0], attr, np.roll(getattr(ad[0], attr), int_offset, axis=0))
            if frac_offset > 0:
                plus_one = np.roll(getattr(ad[0], attr), 1, axis=0)
                if attr == 'mask':
                    setattr(ad[0], attr, getattr(ad[0], attr) | plus_one)
                else:
                    setattr(ad[0], attr, (1 - frac_offset) * getattr(ad[0], attr) +
                            frac_offset * plus_one)
        ad.phu['QOFFSET'] += offset * i * ad.pixel_scale()


@pytest.fixture(scope='function')
def adinputs(path_to_inputs):
    return [astrodata.from_file(os.path.join(path_to_inputs, f))
            for f in test_datasets]


@pytest.fixture(scope='function')
def adinputs2(path_to_inputs):
    return [astrodata.from_file(os.path.join(path_to_inputs, f))
            for f in test_datasets2]


@pytest.fixture(scope="module", autouse=True)
def setup_log(change_working_dir):
    with change_working_dir():
        logutils.config(file_name='test_gmos_spect_ls_resample_2d.log')


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
    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS
    from gempy.utils import logutils

    associated_calibrations = {
        "S20190808S0048.fits": 'S20190808S0167.fits',
        "S20190808S0049.fits": 'S20190808S0168.fits',
        "S20190808S0053.fits": 'S20190808S0169.fits',
        "N20180106S0025.fits": 'N20180115S0264.fits',
        "N20180106S0026.fits": 'N20180115S0264.fits',
        "N20180106S0028.fits": 'N20180115S0264.fits',
        "N20180106S0029.fits": 'N20180115S0264.fits',
    }

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("./inputs", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for fname, arc_fname in associated_calibrations.items():

        sci_path = download_from_archive(fname)
        arc_path = download_from_archive(arc_fname)

        sci_ad = astrodata.from_file(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing ARC for {:s}'.format(data_label))
        logutils.config(file_name='log_arc_{}.txt'.format(data_label))

        if os.path.exists(arc_fname.replace('.fits', '_distortionDetermined.fits')):
            arc = astrodata.from_file(arc_fname.replace('.fits', '_distortionDetermined.fits'))
        else:
            p = GMOSLongslit([astrodata.from_file(arc_path)])
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
        p = GMOSLongslit([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.mosaicDetectors()
        p.attachWavelengthSolution(arc=arc)
        p.distortionCorrect()
        p.findApertures(max_apertures=1)
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
