"""
Tests for primitives_stack.
"""
import logging
import os
import tracemalloc

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
def niri_adinputs_with_noise():
    rng = np.random.RandomState(42)
    phu = fits.PrimaryHDU()
    phu.header.update(OBSERVAT='Gemini-North', INSTRUME='NIRI',
                      ORIGNAME='N20010101S0001.fits')
    data = np.ones((1000, 1000))
    adinputs = []
    for i in range(6):
        ad = astrodata.create(phu)
        ad.append(data + i + rng.normal(scale=0.1, size=data.shape))
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
    with pytest.raises(ValueError, match=match):
        p.stackFrames()


def test_error_extension_shape(niri_adinputs, caplog):
    niri_adinputs[1][0].data = np.zeros((3, 3))
    p = NIRIImage(niri_adinputs)
    match = "Not all inputs images have the same shape"
    with pytest.raises(ValueError, match=match):
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
    caplog.set_level(logging.WARNING)
    f2_adinputs[0].phu["LNRS"] = 1
    f2_adinputs[1].phu["LNRS"] = 8
    p = F2Image(f2_adinputs)
    ad = p.stackFrames(operation='mean', reject_method='none').pop()
    assert "Not all inputs have the same gain" in caplog.records[0].message
    # Input gains are 4.44 and 4.44/8
    assert_allclose(ad.hdr["GAIN"], 8.88/9)
    # Input read_noises are 11.7 and 5
    assert_allclose(ad.hdr["RDNOISE"], 8.996944, atol=0.001)


@pytest.mark.parametrize("old_norm", (True, False))
@pytest.mark.parametrize("statsec", (None, "10:1000,10:1000"))
def test_stacking_with_scaling(niri_adinputs_with_noise, statsec, old_norm):
    """Simple test to scale images and confirm new and old methods"""
    p = NIRIImage(niri_adinputs_with_noise)
    ad = p.stackFrames(operation='mean', reject_method='none', scale=True,
                       statsec=statsec,
                       debug_old_normalization=old_norm).pop()
    assert ad[0].data.mean() == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("old_norm", (True, False))
@pytest.mark.parametrize("statsec", (None, "10:1000,10:1000"))
def test_stacking_with_offsetting(niri_adinputs_with_noise, statsec, old_norm):
    """Simple test to offset images and confirm new and old methods"""
    p = NIRIImage(niri_adinputs_with_noise)
    ad = p.stackFrames(operation='mean', reject_method='none', zero=True,
                       statsec=statsec,
                       debug_old_normalization=old_norm).pop()
    assert ad[0].data.mean() == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("old_norm", (False, True))
@pytest.mark.parametrize("separate_ext", (True, False))
def test_stacking_with_scaling_separate_ext(niri_adinputs_with_noise, old_norm, separate_ext):
    adinputs = []
    # Munge into 2 images with 3 extensions each
    for i, ad in enumerate(niri_adinputs_with_noise):
        if i < 2:
            adinputs.append(ad)
        else:
            adinputs[i % 2].append(ad[0])

    for i, ad in enumerate(adinputs):
        assert ad[0].data.mean() == pytest.approx(i+1, rel=0.01)
        assert ad[1].data.mean() == pytest.approx(i+3, rel=0.01)
        assert ad[2].data.mean() == pytest.approx(i+5, rel=0.01)

    p = NIRIImage(adinputs)
    ad = p.stackFrames(operation='mean', reject_method='none', scale=True,
                       separate_ext=separate_ext, debug_old_normalization=old_norm).pop()

    if separate_ext:
        assert ad[0].data.mean() == pytest.approx(1.0, rel=1e-3)
        assert ad[1].data.mean() == pytest.approx(3.0, rel=1e-3)
        assert ad[2].data.mean() == pytest.approx(5.0, rel=1e-3)
    else:
        # overall scaling will be 0.75, so second image has means (1.5, 3, 4.5)
        assert ad[0].data.mean() == pytest.approx(1.25, rel=1e-3)
        assert ad[1].data.mean() == pytest.approx(3.0, rel=1e-3)
        assert ad[2].data.mean() == pytest.approx(4.75, rel=1e-3)


def test_stacking_with_masked_region(niri_adinputs_with_noise, caplog):
    """Mask the entire 'statsec' region in one image"""
    niri_adinputs_with_noise[-1][0].mask = np.zeros_like(
        niri_adinputs_with_noise[-1][0].data, dtype=np.uint16)
    # Note the difference between the slice and 'statsec' below
    niri_adinputs_with_noise[-1][0].mask[599:700, 499:700] = 1
    p = NIRIImage(niri_adinputs_with_noise)
    ad = p.stackFrames(operation='mean', reject_method='none', scale=True,
                       statsec="500:700,600:700").pop()

    # Confirm 5 warnings about masked statsec (images 0-4 vs image 5)
    nwarnings = sum('No overlapping unmasked pixels' in r.message
                    for r in caplog.records)
    assert nwarnings == 5


@pytest.mark.niri
@pytest.mark.parametrize("scale,zero", [(False, False), (True, False), (False, True)])
def test_memory_control_during_stacking(path_to_inputs, scale, zero):
    """
    Test that the memory control during stacking works as expected.
    """
    memory = 1  # GB
    tracemalloc.start()
    adinputs = [astrodata.open(os.path.join(path_to_inputs, f"stacktest{i:03d}.fits"))
                for i in range(1, 6)]
    p = NIRIImage(adinputs)
    _, start = tracemalloc.get_traced_memory()
    ad = p.stackFrames(operation='mean', reject_method='none', scale=scale,
                       zero=zero, memory=memory).pop()
    adsize = ad[0].nddata.size * 10  # SCI+VAR+DQ
    current, peak = tracemalloc.get_traced_memory()
    assert current - start < 1.5 * adsize
    assert peak - start < 1.5 * adsize + 2e9 * memory



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
