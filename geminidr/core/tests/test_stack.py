"""
Tests for primitives_stack.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from numpy.testing import assert_array_equal

import astrodata
import gemini_instruments
from geminidr.niri.primitives_niri_image import NIRIImage
from geminidr.f2.primitives_f2_image import F2Image


@pytest.fixture
def f2_adinputs():
    phu = fits.PrimaryHDU()
    phu.header.update(OBSERVAT='Gemini-North', INSTRUME='F2',
                      ORIGNAME='S20010101S0001.fits')
    data = np.ones((2, 2))
    adinputs = []
    for i in range(2):
        ad = astrodata.create(phu)
        ad.append(data + i)
        adinputs.append(ad)
    return adinputs


@pytest.fixture
def niri_adinputs():
    phu = fits.PrimaryHDU()
    phu.header.update(OBSERVAT='Gemini-North', INSTRUME='NIRI',
                      ORIGNAME='N20010101S0001.fits')
    data = np.ones((2, 2))
    adinputs = []
    for i in range(2):
        ad = astrodata.create(phu)
        ad.append(data + i)
        adinputs.append(ad)
    return adinputs


@pytest.mark.skip("bquint - Investigate this test")
def test_error_only_one_file(niri_adinputs, caplog):
    # With only one file
    p = NIRIImage(niri_adinputs[1:])
    p.stackFrames()
    assert caplog.records[2].message == (
        'No stacking will be performed, since at least two input AstroData '
        'objects are required for stackFrames')


def test_error_extension_number(niri_adinputs, caplog):
    p = NIRIImage(niri_adinputs)
    p.prepare()
    niri_adinputs[1].append(np.zeros((2, 2)))
    match = "Not all inputs have the same number of extensions"
    with pytest.raises(IOError, match=match):
        p.stackFrames()


def test_error_extension_shape(niri_adinputs, caplog):
    niri_adinputs[1][0].data = np.zeros((3, 3))
    p = NIRIImage(niri_adinputs)
    p.prepare()
    match = "Not all inputs images have the same shape"
    with pytest.raises(IOError, match=match):
        p.stackFrames()


def test_stackframes_refcat_propagation(niri_adinputs):
    refcat = Table([[1, 2], [0.0, 1.0], [40.0, 40.0]],
                   names=('Id', 'RAJ2000', 'DEJ2000'))
    for i, ad in enumerate(niri_adinputs):
        if i > 0:
            refcat['RAJ2000'] = [1.0, 2.0]
        ad.REFCAT = refcat

    p = NIRIImage(niri_adinputs)
    p.prepare()
    adout = p.stackFrames()[0]

    # The merged REFCAT should contain 3 sources as follows
    assert len(adout.REFCAT) == 3
    np.testing.assert_equal(adout.REFCAT['Id'], np.arange(1, 4))
    assert all(adout.REFCAT['RAJ2000'] == [0.0, 1.0, 2.0])


def test_rejmap(niri_adinputs):
    for i in (2, 3, 4):
        niri_adinputs.append(niri_adinputs[0] + i)

    p = NIRIImage(niri_adinputs)
    p.prepare()
    adout = p.stackFrames(reject_method='minmax', nlow=1, nhigh=1,
                          save_rejection_map=True)[0]
    assert_array_equal(adout[0].REJMAP, 2)  # rejected 2 values for each pixel


def test_stacking_without_gain_or_readnoise(f2_adinputs):
    """We use F2 since our fake data return None for gain and read_noise,
    due to the absence of the LNRS keyword"""
    p = F2Image(f2_adinputs)
    assert f2_adinputs[0].gain() == [None]
    assert f2_adinputs[0].read_noise() == [None]
    ad = p.stackFrames(operation='mean', reject_method='none')[0]
    assert ad[0].data.mean() == 1.5
    assert ad[0].data.std() == 0
