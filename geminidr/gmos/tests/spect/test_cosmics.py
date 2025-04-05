import os
import pathlib
import sys

import astrodata
import gemini_instruments  # noqa
import numpy as np
import pytest
from astrodata.testing import download_from_archive
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.primitives_gmos_spect import GMOSSpect
from gempy.utils import logutils
from scipy.ndimage import binary_dilation

TESFILE1 = "S20190808S0048_mosaic.fits"  # R400 at 0.740 um
TESFILE2 = "S20190808S0048_varAdded.fits"  # R400 at 0.740 um


# Tests Definitions -----------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('bkgmodel', ['both', 'object', 'skyline', 'none'])
def test_cosmics_on_mosaiced_data(path_to_inputs, caplog, bkgmodel):
    ad = astrodata.from_file(os.path.join(path_to_inputs, TESFILE1))
    ext = ad[0]

    # add some additional fake cosmics
    size = 50
    np.random.seed(42)
    cr_x = np.random.randint(low=5, high=ext.shape[0] - 5, size=size)
    cr_y = np.random.randint(low=5, high=ext.shape[1] - 5, size=size)

    # Don't add cosmics in masked regions
    mask = binary_dilation(ext.mask > 0, iterations=3)
    sel = ~mask[cr_x, cr_y]
    cr_x = cr_x[sel]
    cr_y = cr_y[sel]
    cr_brightnesses = np.random.uniform(low=1000, high=5000, size=len(cr_x))
    ext.data[cr_x, cr_y] += cr_brightnesses

    # Store mask of CR to help debugging
    crmask = np.zeros(ext.shape, dtype=np.uint8)
    crmask[cr_x, cr_y] = 1
    ext.CRMASK = crmask

    debug = os.getenv('DEBUG') is not None
    p = GMOSSpect([ad])
    adout = p.flagCosmicRays(spatial_order=3, bkgfit_niter=5, debug=debug,
                             bkgmodel=bkgmodel)[0]
    if debug:
        p.writeOutputs()
    mask = adout[0].mask
    # check some pixels with real cosmics
    for pix in [(496, 519), (138, 219), (420, 633), (297, 1871)]:
        assert (mask[pix] & DQ.cosmic_ray) == DQ.cosmic_ray

    # And check our fake cosmics.
    assert np.all(mask[np.where(ext.CRMASK)] & DQ.cosmic_ray == DQ.cosmic_ray)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('bkgmodel', ['both', 'object', 'skyline', 'none'])
def test_cosmics(path_to_inputs, caplog, bkgmodel):
    ad = astrodata.from_file(os.path.join(path_to_inputs, TESFILE2))

    for ext in ad:
        # add some additional fake cosmics
        size = 5
        np.random.seed(42)
        cr_x = np.random.randint(low=5, high=ext.shape[0] - 5, size=size)
        cr_y = np.random.randint(low=5, high=ext.shape[1] - 5, size=size)

        # Don't add cosmics in masked regions
        mask = binary_dilation(ext.mask > 0, iterations=3)
        sel = ~mask[cr_x, cr_y]
        cr_x = cr_x[sel]
        cr_y = cr_y[sel]
        cr_brightnesses = np.random.uniform(low=1000, high=5000,
                                            size=len(cr_x))
        ext.data[cr_x, cr_y] += cr_brightnesses

        # Store mask of CR to help debugging
        crmask = np.zeros(ext.shape, dtype=np.uint8)
        crmask[cr_x, cr_y] = 1
        ext.CRMASK = crmask

    debug = os.getenv('DEBUG') is not None
    p = GMOSSpect([ad])
    adout = p.flagCosmicRays(spatial_order=3, bkgfit_niter=5, debug=debug,
                             bkgmodel=bkgmodel)[0]
    if debug:
        p.writeOutputs()

    mask = adout[0].mask
    # # check some pixels with real cosmics
    # for pix in [(496, 519), (138, 219), (420, 633), (297, 1871)]:
    #     assert (mask[pix] & DQ.cosmic_ray) == DQ.cosmic_ray

    # And check our fake cosmics.
    for ext in ad:
        assert np.all(
            mask[np.where(ext.CRMASK)] & DQ.cosmic_ray == DQ.cosmic_ray)


# -- Recipe to create pre-processed data --------------------------------------
def create_inputs_recipe():
    """
    Creates input data for tests using pre-processed standard star and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """

    fnames = ["S20190808S0048.fits"]

    path = pathlib.Path('dragons_test_inputs')
    path = path / "geminidr" / "gmos" / "spect" / "test_cosmics" / "inputs"
    path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)
    print('Current working directory:\n    {!s}'.format(path.cwd()))

    for fname in fnames:
        sci_ad = astrodata.from_file(download_from_archive(fname))
        data_label = sci_ad.data_label()

        print('===== Reducing pre-processed data =====')
        logutils.config(file_name=f'log_{data_label}.txt')
        p = GMOSSpect([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.writeOutputs()
        p.mosaicDetectors()
        p.writeOutputs()


if __name__ == '__main__':
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main([__file__])
