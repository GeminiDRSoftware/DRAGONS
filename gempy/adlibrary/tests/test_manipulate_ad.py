import pytest
from copy import deepcopy
from itertools import cycle

import numpy as np

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from gempy.adlibrary import manipulate_ad as manip

astrofaker = pytest.importorskip('astrofaker')


@pytest.fixture
def ad(astrofaker):
    ad = astrofaker.create('F2', 'IMAGE')
    ad.init_default_extensions()
    ad = manip.remove_single_length_dimension(ad)
    ad.phu['PREPARED'] = 1  # so it will respect DATASEC, etc.
    Y, X = np.mgrid[:ad[0].shape[0], :ad[0].shape[1]]
    ad[0].add(X+Y)
    return ad


def test_reassemble_ad_no_overlap(ad):
    """Split the image into 4 segments and reassemble"""
    adcut = astrodata.create(ad.phu)
    edges = np.linspace(0, ad[0].data.shape[0], 5, dtype=int)
    for y1, y2 in zip(edges[:-1], edges[1:]):
        adcut.append(ad[0].nddata[y1:y2])
        adcut[-1].hdr['ARRAYSEC'] = f"[1:{adcut[-1].shape[1]},{y1+1}:{y2}]"

    adout = manip.reassemble_ad(adcut)
    print(adcut.hdr['ARRAYSEC'])
    np.testing.assert_equal(adout[0].data, ad[0].data)
    assert adout[0].mask.sum() == 0


def test_reassemble_ad_non_contiguous(ad):
    """Split the image into 4 non-contiguous segments and reassemble"""
    adcut = astrodata.create(ad.phu)
    edges = np.linspace(0, ad[0].data.shape[0], 9, dtype=int)
    for y1, y2 in zip(edges[:-1:2], edges[1::2]):
        adcut.append(ad[0].nddata[y1:y2])
        adcut[-1].hdr['ARRAYSEC'] = f"[1:{adcut[-1].shape[1]},{y1+1}:{y2}]"

    adout = manip.reassemble_ad(adcut, shape=ad[0].data.shape)
    for masked, y1, y2 in zip(cycle([False, True]), edges[:-1], edges[1:]):
        if masked:
            assert adout[0].data[y1:y2].sum() == 0
            assert np.unique(adout[0].mask[y1:y2]) == [64]
        else:
            np.testing.assert_equal(adout[0].data[y1:y2], ad[0].data[y1:y2])


def test_reassemble_ad_overlap(ad):
    """
    Split the image into 3 overlapping segments and reassemble.

    The 3 segments are each masked on the bottom and top quarters.
    So the unmasked regions are [256:768], [768:1280], and [1280:1792].
    """
    adcut = astrodata.create(ad.phu)
    edges = np.linspace(0, ad[0].data.shape[0], 5, dtype=int)
    for y1, y2 in zip(edges[:-1], edges[2:]):
        adcut.append(ad[0].nddata[y1:y2])
        adcut[-1].hdr['ARRAYSEC'] = f"[1:{adcut[-1].shape[1]},{y1+1}:{y2}]"
        adcut[-1].mask = np.zeros(adcut[-1].shape, dtype=np.uint16)
        one_quarter = (y2 - y1) // 4
        adcut[-1].mask[:one_quarter] = 64
        adcut[-1].mask[-one_quarter:] = 64

    adout = manip.reassemble_ad(adcut, shape=ad[0].data.shape)

    # Check the masking
    assert adout[0].mask[:256].mean() == 64
    assert adout[0].mask[1792:].mean() == 64
    assert adout[0].mask[256:1792].sum() == 0

    # Check the data
    np.testing.assert_equal(adout[0].data[256:1792], ad[0].data[256:1792])
    assert adout[0].data[:256].sum() == 0
    assert adout[0].data[1792:].sum() == 0


def test_rebin_data(ad):
    """Basic rebinning test"""
    adout = manip.rebin_data(ad, xbin=2, ybin=2)

    assert adout[0].shape == (1024, 1024)

    yy, xx = np.mgrid[:ad[0].shape[0], :ad[0].shape[1]]
    np.testing.assert_allclose(adout[0].data, 8*xx+8*yy+4)

    assert (adout.data_section()[0].x1, adout.data_section()[0].x2) == (0, 1024)


def test_rebin_twice(ad):
    """Check that rebinning to 4x4 via 2x2 doesn't change the result"""
    adcopy = deepcopy(ad)
    ad4 = manip.rebin_data(adcopy, xbin=4, ybin=4)
    ad22 = manip.rebin_data(manip.rebin_data(ad, xbin=2, ybin=2),
                            xbin=4, ybin=4)
    np.testing.assert_allclose(ad4[0].data, ad22[0].data)


def test_rebin_not_multiple(ad):
    """Test that an exception is raised if we can't rebin the data because
    the binning factors are not multiples of the data shape."""
    with pytest.raises(ValueError):
        manip.rebin_data(ad, xbin=2, ybin=3)


def test_remove_single_length_dimension():
    # Any F2 file
    ad = astrodata.open(download_from_archive("S20260711S0242.fits"))
    assert len(ad[0].shape) == 3
    ad = manip.remove_single_length_dimension(ad)
    assert ad[0].shape == (2048, 2048)