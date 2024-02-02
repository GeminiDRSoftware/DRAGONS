import pytest
import os

import numpy as np

import astrodata, gemini_instruments

from geminidr.ghost.primitives_ghost_spect import GHOSTSpect, make_wavelength_table


FILENAMES = ["S20230130S0104_2x8_blue001_wavelengthSolutionAttached.fits",
             "S20230130S0104_2x8_red001_wavelengthSolutionAttached.fits"]


@pytest.mark.ghostspect
@pytest.mark.parametrize("filename", FILENAMES)
def test_combine_orders_single_file(change_working_dir, path_to_inputs, filename):
    """Check that combineOrders() works on a single file"""
    ad = astrodata.open(os.path.join(path_to_inputs, filename))
    orig_wavl = make_wavelength_table(ad[0])
    p = GHOSTSpect([ad])
    ad_out = p.combineOrders().pop()
    pixels = np.arange(ad_out[0].data.size)
    x = ad_out[0].wcs(pixels[1:]) / ad_out[0].wcs(pixels[:-1])
    assert x.std() < 1e-12  # all wavelength ratios are the same
    assert ad_out[0].wcs(0) == pytest.approx(orig_wavl.min())

    # Check that we can write and read back the _ordersCombined file
    with change_working_dir():
        ad_out.write("test.fits", overwrite=True)
        ad2 = astrodata.open("test.fits")
        np.testing.assert_allclose(ad_out[0].wcs(pixels), ad2[0].wcs(pixels))


@pytest.mark.ghostspect
def test_combine_orders_red_and_blue_no_stacking(path_to_inputs):
    """Check that combineOrders() works on two files without stacking"""
    adinputs = [astrodata.open(os.path.join(path_to_inputs, filename))
                for filename in FILENAMES]
    p = GHOSTSpect(adinputs)
    adoutputs = p.combineOrders(stacking_mode="none")
    assert len(adoutputs) == len(adinputs)  # no stacking

    # Check that the wavelength solutions haven't merged in some way
    for adin, adout in zip(adinputs, adoutputs):
        orig_wavl = make_wavelength_table(adin[0])
        assert adout[0].wcs(0) == pytest.approx(orig_wavl.min())


@pytest.mark.ghostspect
@pytest.mark.parametrize("stacking_mode", ("scaled", "unscaled"))
def test_combine_orders_red_and_blue_stacking(path_to_inputs, stacking_mode):
    """Check that combineOrders() works on two files without stacking"""
    adinputs = [astrodata.open(os.path.join(path_to_inputs, filename))
                for filename in FILENAMES]
    orig_wavl = np.array([make_wavelength_table(ad[0]) for ad in adinputs])
    p = GHOSTSpect(adinputs)
    adoutputs = p.combineOrders(stacking_mode=stacking_mode)
    assert len(adoutputs) == 1  # stacking
    ad_out = adoutputs[0]

    # Check that this combined arc makes sense
    pixels = np.arange(ad_out[0].data.size)
    x = ad_out[0].wcs(pixels[1:]) / ad_out[0].wcs(pixels[:-1])
    assert x.std() < 1e-12  # all wavelength ratios are the same
    assert ad_out[0].wcs(0) == pytest.approx(orig_wavl.min())
