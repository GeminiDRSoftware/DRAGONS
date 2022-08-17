"""
Tests for sky stacking and subtraction for GNIRS longslit spectroscopy.
"""

import pytest

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
import pytest

# ---- Fixtures ---------------------------------------------------------------
@pytest.fixture
def gnirs_abba():
    return [astrodata.open(download_from_archive(f)) for f in
            ('N20141119S0331.fits', 'N20141119S0332.fits',
             'N20141119S0333.fits', 'N20141119S0334.fits')]

# ---- Tests ------------------------------------------------------------------
@pytest.mark.gnirsls
def test_associate_sky_abba(gnirs_abba):

    p = GNIRSLongslit(gnirs_abba)
    # The perpendicular offsets are reported as 0.14" for some frames but the
    # slit width is only 0.1", so the WCS needs to be reset.
    p.prepare(bad_wcs="new")
    p.separateSky()
    p.associateSky()

    a1, b1, b2, a2 = p.showList()

    a_frames = {'N20141119S0331_skyAssociated.fits',
                'N20141119S0334_skyAssociated.fits'}
    b_frames = {'N20141119S0332_skyAssociated.fits',
                'N20141119S0333_skyAssociated.fits'}

    # check that the A frames get the B frames as skies, and vice versa
    for ad in (a1, a2):
        assert set(ad.SKYTABLE['SKYNAME']) == b_frames
    for ad in (b1, b2):
        assert set(ad.SKYTABLE['SKYNAME']) == a_frames

@pytest.mark.gnirsls
def test_associate_sky_pass_skies(gnirs_abba):

    in_sky_names = set([ad.filename for ad in gnirs_abba[1:3]])

    p = GNIRSLongslit(gnirs_abba)
    # Don't run separate sky to simulate resuming work with known skies.
    p.associateSky(sky=gnirs_abba[1:3])

    out_sky_names = set([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

    assert in_sky_names == out_sky_names

@pytest.mark.gnirsls
def test_associate_sky_use_all(gnirs_abba):

    in_sky_names = set([ad.filename for ad in gnirs_abba])

    p = GNIRSLongslit(gnirs_abba)
    p.prepare()
    p.separateSky()
    p.associateSky(distance=0, use_all=True)

    for ad in p.showList():
        skies = set([s.replace('_skyAssociated', '')
                     for s in ad.SKYTABLE['SKYNAME']])

        # Check that each AD has all the other frames as skies, but not itself.
        assert skies == in_sky_names - set([ad.phu['ORIGNAME']])

@pytest.mark.gnirsls
def test_associate_sky_exclude_all(gnirs_abba):
    p = GNIRSLongslit(gnirs_abba)
    p.prepare()
    p.separateSky()
    # Offset is 5" so this will exclude skies if 'use_all' is False.
    p.associateSky(distance=10)

    for ad in p.showList():
        with pytest.raises(AttributeError):
            ad.SKYTABLE
