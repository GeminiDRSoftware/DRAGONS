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
def f2_abba():
    return [astrodata.from_file(download_from_archive(f)) for f in
            ('S20200301S0071.fits', 'S20200301S0072.fits',
             'S20200301S0073.fits', 'S20200301S0074.fits')]

# ---- Tests ------------------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.f2ls
def test_associate_sky_abba(f2_abba):

    p = F2Longslit(f2_abba)
    p.prepare()
    p.separateSky()
    p.associateSky()

    a1, b1, b2, a2 = p.showList()

    a_frames = {'S20200301S0071_skyAssociated.fits',
                'S20200301S0074_skyAssociated.fits'}
    b_frames = {'S20200301S0072_skyAssociated.fits',
                'S20200301S0073_skyAssociated.fits'}

    # check that the A frames get the B frames as skies, and vice versa
    for ad in (a1, a2):
        assert set(ad.SKYTABLE['SKYNAME']) == b_frames
    for ad in (b1, b2):
        assert set(ad.SKYTABLE['SKYNAME']) == a_frames

@pytest.mark.dragons_remote_data
@pytest.mark.f2ls
def test_associate_sky_pass_skies(f2_abba):

    in_sky_names = set([ad.filename for ad in f2_abba[1:3]])

    p = F2Longslit(f2_abba)
    # Don't run separate sky to simulate resuming work with known skies.
    p.associateSky(sky=f2_abba[1:3])

    out_sky_names = set([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

    assert in_sky_names == out_sky_names

@pytest.mark.dragons_remote_data
@pytest.mark.f2ls
def test_associate_sky_use_all(f2_abba):

    in_sky_names = set([ad.filename for ad in f2_abba])

    p = F2Longslit(f2_abba)
    p.prepare()
    p.separateSky()
    p.associateSky(distance=0, use_all=True)

    for ad in p.showList():
        skies = set([s.replace('_skyAssociated', '')
                     for s in ad.SKYTABLE['SKYNAME']])

        # Check that each AD has all the other frames as skies, but not itself.
        assert skies == in_sky_names - set([ad.phu['ORIGNAME']])

@pytest.mark.dragons_remote_data
@pytest.mark.f2ls
def test_associate_sky_exclude_all(f2_abba):
    p = F2Longslit(f2_abba)
    p.prepare()
    p.separateSky()
    # Offset is 40" so this will exclude skies if 'use_all' is False.
    p.associateSky(distance=50)

@pytest.mark.dragons_remote_data
@pytest.mark.f2ls
def test_associate_sky_quasi_abcde():

    files = ['S20210515S0196.fits', 'S20210515S0197.fits',
             'S20210515S0201.fits', 'S20210515S0202.fits',
             'S20210515S0203.fits', 'S20210515S0206.fits',
             'S20210515S0208.fits']

    data = [astrodata.from_file(download_from_archive(f)) for f in files]

    p = F2Longslit(data)
    p.prepare()
    p.separateSky()
    p.associateSky(min_skies=3)

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
