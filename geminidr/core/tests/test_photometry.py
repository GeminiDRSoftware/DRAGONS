# pytest suite
"""
Tests for primitives_photometry.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): pytest -v --capture=no
"""
import pytest
import logging

from astropy.coordinates import SkyCoord
from astropy.table import Table
import numpy as np

from .. import primitives_photometry as prims
from ...gemini.lookups.timestamp_keywords import timestamp_keys

from geminidr.niri.primitives_niri_image import NIRIImage

STAR_POSITIONS = [(200., 200.), (300.5, 800.5)]
LOGFILE = 'test_photometry.log'


@pytest.fixture
def niri_image(astrofaker):
    ad = astrofaker.create('NIRI', 'IMAGE')
    ad.init_default_extensions()

    # SExtractor struggles if the background is noiseless
    ad.add_read_noise()

    for x, y in STAR_POSITIONS:
        ad[0].add_star(amplitude=500, x=x, y=y)

    return NIRIImage([ad])


@pytest.mark.dragons_remote_data
def test_addReferenceCatalog(niri_image, caplog):
    caplog.set_level(logging.WARNING, logger="geminidr")

    adinputs = niri_image.addReferenceCatalog()
    assert len(adinputs) == 1

    ad = adinputs[0]

    assert timestamp_keys["addReferenceCatalog"] in ad.phu

    # Handle problem with catalogue server being down
    try:
        assert hasattr(ad, 'REFCAT')
    except AssertionError:
        for record in caplog.records:
            if (record.levelname == 'WARNING' and
                    "appears to be down" in record.message):
                pytest.skip(record.message)
        raise

    # Check all objects in REFCAT are within prescribed radius
    search_radius = niri_image.params['addReferenceCatalog'].radius
    base_coord = SkyCoord(ra=ad.wcs_ra(), dec=ad.wcs_dec(), unit='deg')

    for ra, dec in ad.REFCAT['RAJ2000', 'DEJ2000']:
        assert SkyCoord(ra=ra, dec=dec, unit='deg').separation(base_coord).value < search_radius


def test_detectSources(niri_image):
    def catsort(cat):
        # Sort catalogue by first ordinate
        return sorted(cat, key=lambda x: x[0])

    adinputs = niri_image.detectSources()
    assert len(adinputs) == 1

    ad = adinputs[0]

    assert timestamp_keys["detectSources"] in ad.phu

    # Hard to do much more than check OBJMASK exists
    assert hasattr(ad[0], 'OBJMASK')

    # We can do more checking on the OBJCAT, however
    assert len(ad[0].OBJCAT) == len(STAR_POSITIONS)

    cat_positions = ad[0].OBJCAT['X_IMAGE', 'Y_IMAGE']

    for (realx, realy), (catx, caty) in zip(catsort(STAR_POSITIONS),
                                            catsort(cat_positions)):
        # 0-index vs 1-index
        assert abs(realx + 1 - catx) < 0.5 and abs(realy + 1 - caty) < 0.5  # NUMPY_2: OK


def test_calculate_magnitudes():
    # Set up a simple REFCAT
    refcat = Table()
    refcat['NUMBER'] = np.arange(5) + 1

    for i, filt in enumerate('jhk'):
        refcat['{}mag'.format(filt)] = np.array(float(i) + refcat['NUMBER'], dtype='f4')
        refcat['{}mag_err'.format(filt)] = np.array([0.1] * len(refcat), dtype='f4')

    # Simple
    prims._calculate_magnitudes(refcat, ['h'])

    assert all(refcat['filtermag'].data == refcat['hmag'].data)
    assert all(refcat['filtermag_err'].data == refcat['hmag_err'].data)

    refcat.remove_columns(['filtermag', 'filtermag_err'])

    # Constant offset
    prims._calculate_magnitudes(refcat, ['h', (0.1, 0.1)])

    assert all(refcat['filtermag'].data == refcat['hmag'].data + 0.1)
    assert all(abs(refcat['filtermag_err'].data - np.sqrt(0.01 + refcat['hmag_err'].data ** 2)) < 0.001)

    refcat.remove_columns(['filtermag', 'filtermag_err'])

    # Colour term
    prims._calculate_magnitudes(refcat, ['h', (1.0, 0.1, 'j-h')])

    assert all(refcat['filtermag'].data == refcat['jmag'].data)

    # This holds because J-H=-1 for all rows
    assert all(abs(refcat['filtermag_err'].data - 0.2) < 0.001)


def test_clean_objcat(niri_image):
    ad = niri_image.detectSources()[0]
    total_objects = len(ad[0].OBJCAT)
    total_object_pixels = np.sum(ad[0].OBJMASK)

    ad[0].OBJCAT['B_IMAGE'][0] = 1.0

    prims.clean_objcat(ad[0])

    # Check one object has gone and "some" OBJMASK pixels have gone
    assert len(ad[0].OBJCAT) == total_objects - 1
    assert np.sum(ad[0].OBJMASK) < total_object_pixels


def test_estimate_seeing(niri_image):
    """This is primarily a test that SExractor is producing sensible values
    since we don't care *precisely* how the FWHM is derived from a long list,
    only that the value is representative."""
    ad = niri_image.detectSources()[0]
    seeing = prims._estimate_seeing(ad[0].OBJCAT)

    assert abs(seeing - ad.seeing) / ad.seeing < 0.05  # NUMPY_2: OK


def test_estimate_seeing_stats():
    # Set up an OBJCAT that's all good
    default_values = {'ISOAREA_IMAGE': (50, 15),
                      'B_IMAGE': (5.0, 1.0),
                      'ELLIPTICITY': (0.05, 0.7),
                      'CLASS_STAR': (0.95, 0.1),
                      'FLUX_AUTO': (10000., 10),
                      'FLUXERR_AUTO': (10., 1000.),
                      'FLAGS': (0, 16),
                      'FWHM_WORLD': (0.00025, -1.),
                      'NIMAFLAGS_ISO': (0, 20)}

    objcat = Table()
    objcat['NUMBER'] = np.arange(10) + 1

    # Need some scatter or else sigma_clip clips any single different value
    # Spread values around 0.00025 degrees = 0.9 arcseconds
    objcat['FWHM_WORLD'] = 0.000241 + 0.000002 * np.arange(10)

    for k, v in default_values.items():
        objcat[k] = [v[0]] * len(objcat)

    assert abs(prims._estimate_seeing(objcat) - 0.9) < 0.0001

    for k, v in default_values.items():
        # Check rejection of bad object, OK value
        objcat[k][0] = v[1]
        objcat['FWHM_WORLD'][0] = 0.00025

        assert abs(prims._estimate_seeing(objcat) - 0.9) < 0.0001

        objcat[k][0] = v[0]
        objcat['FWHM_WORLD'][0] = default_values['FWHM_WORLD'][0] * 10

    # Check rejection of good object, bad value
    objcat['FWHM_WORLD'][0] = 0.001
    assert abs(prims._estimate_seeing(objcat) - 0.9) < 0.0001


def test_profile_sources(niri_image):
    """
    We give slightly more leeway here than for _estimate_seeing since that works
    better for true Gaussians (our fake sources) but this works better for real
    sources.
    """
    ad = niri_image.detectSources()[0]
    ad = prims._profile_sources(ad)

    assert 'PROFILE_FWHM' in ad[0].OBJCAT.colnames
    assert 'PROFILE_EE50' in ad[0].OBJCAT.colnames

    pixscale = ad.pixel_scale()

    for value in ad[0].OBJCAT['PROFILE_FWHM']:
        assert abs(value * pixscale - ad.seeing) / ad.seeing < 0.1

    for value in ad[0].OBJCAT['PROFILE_EE50']:
        assert abs(value * pixscale - ad.seeing) / ad.seeing < 0.1
