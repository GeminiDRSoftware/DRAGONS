#!/usr/bin/env python3
"""
Tests for sky stacking and subtraction for F2.
"""

import pytest

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.f2.primitives_f2_longslit import F2Longslit

# ---- Fixtures ---------------------------------------------------------------
@pytest.fixture
def f2_objects():
    files = ('S20200301S0071.fits', 'S20200301S0074.fits')
    return [astrodata.open(download_from_archive(f)) for f in files]

@pytest.fixture
def f2_skies():
    files = ('S20200301S0072.fits', 'S20200301S0073.fits')
    return [astrodata.open(download_from_archive(f)) for f in files]


# ---- Tests ------------------------------------------------------------------
def test_associate_sky_abba(f2_objects, f2_skies):

    p = F2Longslit(f2_objects + f2_skies)
    p.prepare()
    p.separateSky()
    p.associateSky()

    a1, a2, b1, b2 = p.showList()

    a_frames = {'S20200301S0071_skyAssociated.fits',
                'S20200301S0074_skyAssociated.fits'}
    b_frames = {'S20200301S0072_skyAssociated.fits',
                'S20200301S0073_skyAssociated.fits'}

    for ad in (a1, a2):
        assert set(ad.SKYTABLE['SKYNAME']) == b_frames
    for ad in (b1, b2):
        assert set(ad.SKYTABLE['SKYNAME']) == a_frames

@pytest.mark.parametrize('use_all', [True, False])
def test_associate_sky_use_all(use_all, f2_objects, f2_skies):

    in_sky_names = set([ad.filename for ad in f2_skies])

    p = F2Longslit(f2_objects + f2_skies)
    p.prepare()
    p.separateSky()
    # Offset is 40" so this will exclude skies if 'use_all' is False.
    p.associateSky(distance=50, use_all=use_all)

    if use_all:
        out_sky_names = set([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

        assert in_sky_names == out_sky_names
    else:
        for ad in p.showList():
            with pytest.raises(AttributeError):
                ad.SKYTABLE


def test_associate_sky_quasi_abcde(caplog):

    files = ['S20210515S0196.fits', 'S20210515S0197.fits',
             'S20210515S0201.fits', 'S20210515S0202.fits',
             'S20210515S0203.fits', 'S20210515S0206.fits',
             'S20210515S0208.fits']

    data = [astrodata.open(download_from_archive(f)) for f in files]

    p = F2Longslit(data)
    p.prepare()
    p.separateSky()
    p.associateSky()

    assert set(p.showList()[0].SKYTABLE['SKYNAME']) == set([
                                        'S20210515S0197_skyAssociated.fits',
                                        'S20210515S0201_skyAssociated.fits',
                                        'S20210515S0202_skyAssociated.fits'])

    for ad in p.showList()[1:-1]:
        assert set(ad.SKYTABLE['SKYNAME']) == set([
                                        'S20210515S0196_skyAssociated.fits',
                                        'S20210515S0208_skyAssociated.fits'])

    assert set(p.showList()[-1].SKYTABLE['SKYNAME']) == set([
                                        'S20210515S0202_skyAssociated.fits',
                                        'S20210515S0203_skyAssociated.fits',
                                        'S20210515S0206_skyAssociated.fits'])
