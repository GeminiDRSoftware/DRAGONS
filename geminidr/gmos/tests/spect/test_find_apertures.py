#!/usr/bin/env python
"""
Tests for GMOS Spect findApertures.
"""

import os
import numpy as np
import pytest

import astrodata
import gemini_instruments

from astropy.modeling.models import Gaussian1D
from geminidr.gmos.primitives_gmos_spect import GMOSSpect


test_data = [
    ("N20180109S0287_varAdded.fits", 256)  # GN-2017B-FT-20-13-001 B600 0.505um
    # "N20190302S0089_varAdded.fits",  # GN-2019A-Q-203-7-001 B600 0.550um
    # "N20190313S0114_varAdded.fits",  # GN-2019A-Q-325-13-001 B600 0.482um
    # "N20190427S0123_varAdded.fits",  # GN-2019A-FT-206-7-001 R400 0.525um
    # "N20190427S0126_varAdded.fits",  # GN-2019A-FT-206-7-004 R400 0.625um
    # "N20190910S0028_varAdded.fits",  # GN-2019B-Q-313-5-001 B600 0.550um
    # "S20180919S0139_varAdded.fits",  # GS-2018B-Q-209-13-003 B600 0.45um
    # "S20191005S0051_varAdded.fits",  # GS-2019B-Q-132-35-001 R400 0.73um
]


@pytest.mark.gmosls
@pytest.mark.parametrize("seeing", [1.0])  # [0.5, 0.75, 1.0, 1.5, 2.0])
def test_find_apertures_with_fake_data(seeing):
    """
    Creates a fake AD object with a gaussian profile in spacial direction with a
    fwhm defined by the seeing variable in arcsec. Then add some noise, and
    test if p.findAperture can find its position.
    """
    astrofaker = pytest.importorskip("astrofaker")

    gmos_fake_noise = 4  # adu
    gmos_plate_scale = 0.0807  # arcsec . px-1
    fwhm_to_stddev = 2 * np.sqrt(2 * np.log(2))
    
    ad = astrofaker.create('GMOS-S', mode='SPECT')
    ad.init_default_extensions()

    y0 = np.random.randint(low=100, high=ad[0].shape[0] - 100)
    fwhm = seeing / gmos_plate_scale
    stddev = fwhm / fwhm_to_stddev

    model = Gaussian1D(mean=y0, stddev=stddev, amplitude=50)
    rows, cols = np.mgrid[:ad.shape[0][0], :ad.shape[0][1]]
    print(rows.shape)

    for ext in ad:
        ext.data = model(rows)
        ext.data += np.random.poisson(ext.data)
        ext.data += (np.random.random(size=ext.data.shape) - 0.5) * gmos_fake_noise
        
        ext.mask = np.zeros_like(ext.data, dtype=np.uint)
        ext.variance = np.sqrt(ext.data)

    p = GMOSSpect([ad])
    _ad = p.findSourceApertures()[0]

    print(_ad.info)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad_and_center", test_data, indirect=True)
def test_find_apertures_using_standard_star(ad_and_center):
    """
    Test that p.findApertures can find apertures in Standard Star (which are
    normally bright) observations.
    """
    ad, expected_center = ad_and_center
    p = GMOSSpect(ad)
    _ad = p.findSourceApertures(max_apertures=1)[0]

    assert hasattr(ad[0], 'APERTURE')
    assert len(ad[0].APERTURE) == 1
    assert np.testing.assert_almost_equal(
        ad[0].APERTURE['center'], expected_center, 1)


# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='function')
def ad_and_center(path_to_inputs, request):
    """
    Returns the pre-processed spectrum file.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the `applyQECorrection`.
    """
    filename, center = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad, center


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
    from gempy.utils import logutils

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/spect/test_find_apertures/inputs/"
    path = os.path.join(root_path, module_path)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    for filename in test_data:

        filename = "{:s}.fits".format(filename.split("_")[0])
        sci_path = download_from_archive(filename)

        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = GMOSSpect([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.writeOutputs()


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
