"""
Tests for primitives_stack.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from numpy.testing import assert_array_equal, assert_allclose

import astrodata, gemini_instruments
from geminidr.niri.primitives_niri_image import NIRIImage
from geminidr.f2.primitives_f2_image import F2Image


# -- Fixtures -----------------------------------------------------------------

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


@pytest.fixture
def niri_image(astrofaker):
    """Create a fake NIRI image.

    Optional
    --------
    keywords : dict
        A dictionary with keys equal to FITS header keywords, whose values
        will be propogated to the new image.

    """

    def _niri_image(filename='N20010101S0001.fits', keywords={}):

        ad = astrofaker.create('NIRI', 'IMAGE',
                                extra_keywords=keywords,
                                filename=filename)
        ad.init_default_extensions()
        return ad

    return _niri_image

# -- Tests --------------------------------------------------------------------

def test_error_only_one_file(niri_adinputs, caplog):
    caplog.set_level(20)  # INFO
    # With only one file
    p = NIRIImage(niri_adinputs[1:])
    p.stackFrames()
    assert caplog.records[2].message == (
        'No stacking will be performed, since at least two input AstroData '
        'objects are required for stackFrames')


def test_error_extension_number(niri_adinputs):
    p = NIRIImage(niri_adinputs)
    niri_adinputs[1].append(np.zeros((2, 2)))
    match = "Not all inputs have the same number of extensions"
    with pytest.raises(IOError, match=match):
        p.stackFrames()


def test_error_extension_shape(niri_adinputs, caplog):
    niri_adinputs[1][0].data = np.zeros((3, 3))
    p = NIRIImage(niri_adinputs)
    match = "Not all the matching extensions have the same shape."
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
    adout = p.stackFrames()[0]

    # The merged REFCAT should contain 3 sources as follows
    assert len(adout.REFCAT) == 3
    np.testing.assert_equal(adout.REFCAT['Id'], np.arange(1, 4))
    assert all(adout.REFCAT['RAJ2000'] == [0.0, 1.0, 2.0])


def test_rejmap(niri_adinputs):
    for i in (2, 3, 4):
        niri_adinputs.append(niri_adinputs[0] + i)

    p = NIRIImage(niri_adinputs)
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


def test_stacking_gain_read_noise_propagation(f2_adinputs, caplog):
    f2_adinputs[0].phu["LNRS"] = 1
    f2_adinputs[1].phu["LNRS"] = 8
    p = F2Image(f2_adinputs)
    ad = p.stackFrames(operation='mean', reject_method='none').pop()
    assert "Not all inputs have the same gain" in caplog.records[0].message
    # Input gains are 4.44 and 4.44/8
    assert_allclose(ad.hdr["GAIN"], 8.88/9)
    # Input read_noises are 11.7 and 5
    assert_allclose(ad.hdr["RDNOISE"], 8.996944, atol=0.001)


@pytest.mark.parametrize("rejection_method, expected",
                         [('varclip', 2.),
                          ('sigclip', 1.6),
                          ('minmax', 1.666666)])
def test_stack_biases(rejection_method, expected, niri_image):
    """Try different rejection methods to make sure we get the expected
    results"""

    adinputs = []
    for i in (0, 1, 2, 2, 3):
        ad = niri_image()
        data = np.ones((2, 2)) * i
        ad[0].data = data
        ad.tags = ad.tags.union({'BIAS'})
        for ext in ad:
            ext.variance = np.where(ext.data > 0,
                                    ext.data, 0).astype(np.float32)
        adinputs.append(ad)

    p = NIRIImage(adinputs)
    p.addVAR()
    if rejection_method == 'minmax':
        ad_out = p.stackBiases(adinputs, reject_method=rejection_method,
                               nlow=1, nhigh=1)
    else:
        ad_out = p.stackBiases(adinputs, reject_method=rejection_method)

    assert len(ad_out) == 1
    assert len(p.streams["main"]) == len(adinputs)
    assert pytest.approx(ad_out[0].data[0]) == expected

    # Check that removing a BIAS tag raises an error.
    adinputs[0].tags = adinputs[0].tags.difference({'BIAS'})

    with pytest.raises(ValueError, match='Not all inputs have BIAS tag'):
        p.stackBiases()


def test_stack_flats(niri_image):
    adinputs = [niri_image(f'N20010101S{i:04d}.fits',
                           keywords={'EXPTIME': 10.})
                for i in range(0, 4)]

    p = NIRIImage(adinputs)
    ad_out = p.stackFlats(adinputs)

    assert len(ad_out) == 1
    assert len(p.streams["main"]) == len(adinputs)


def test_stack_darks(niri_image):
    adinputs = [niri_image(f'N20010101S{i:04d}.fits',
                           keywords={'EXPTIME': 1.})
                for i in range(0, 4)]
    for ad in adinputs:
        ad.tags = ad.tags.union({'DARK'})

    p = NIRIImage(adinputs)
    p.stackDarks(adinputs)

    assert len(p.streams["main"]) == len(adinputs)

    adinputs[0].phu['EXPTIME'] = 100.

    # Check that a file with a different exposure time raises an error.
    with pytest.raises(ValueError):
        p.stackDarks(adinputs)

    # Check that removing a DARK tag raises an error.
    adinputs[0].phu['EXPTIME'] =  1.
    adinputs[0].tags = adinputs[0].tags.difference({'DARK'})

    with pytest.raises(ValueError):
        p.stackDarks(adinputs)
