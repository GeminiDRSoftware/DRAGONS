"""
Tests for sky stacking and subtraction for GNIRS longslit spectroscopy.
"""

import pytest

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit

# ---- Fixtures ---------------------------------------------------------------
@pytest.fixture
def gnirs_objects():
    files = ('N20141119S0331.fits', 'N20141119S0334.fits')
    return [astrodata.open(download_from_archive(f)) for f in files]

@pytest.fixture
def gnirs_skies():
    files = ('N20141119S0332.fits', 'N20141119S0333.fits')
    return [astrodata.open(download_from_archive(f)) for f in files]


# ---- Tests ------------------------------------------------------------------
def test_associate_sky_abba(gnirs_objects, gnirs_skies):

    p = GNIRSLongslit(gnirs_objects + gnirs_skies)
    p.prepare()
    p.separateSky()
    p.associateSky()

    a1, a2 = p.streams['main']
    b1, b2 = p.streams['sky']

    b_frames = {'N20141119S0332_skyAssociated.fits',
                'N20141119S0333_skyAssociated.fits'}

    # check that the A frames get the B frames as skies
    for ad in (a1, a2):
        assert set(ad.SKYTABLE['SKYNAME']) == b_frames

def test_associate_sky_pass_skies(gnirs_objects, gnirs_skies):

    in_sky_names = set([ad.filename for ad in gnirs_skies])

    p = GNIRSLongslit(gnirs_objects)
    # Don't run separate sky to simulate resuming work with known skies.
    p.associateSky(sky=gnirs_skies)

    out_sky_names = set([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

    assert in_sky_names == out_sky_names

@pytest.mark.parametrize('use_all', [True, False])
def test_associate_sky_use_all(use_all, gnirs_objects, gnirs_skies):

    in_sky_names = set([ad.filename for ad in gnirs_skies])

    p = GNIRSLongslit(gnirs_objects + gnirs_skies)
    p.prepare()
    p.separateSky()
    # Offset is only 5" so this will exclude skies if 'use_all' is False.
    p.associateSky(distance=10, use_all=use_all)

    if use_all:
        out_sky_names = set([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

        assert in_sky_names == out_sky_names
    else:
        for ad in p.showList():
            with pytest.raises(AttributeError):
                ad.SKYTABLE
