"""
Tests for sky stacking and subtraction for NIRI longslit spectroscopy.
"""

import numpy as np
import pytest

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.niri.primitives_niri_longslit import NIRILongslit

# ---- Fixtures ---------------------------------------------------------------
@pytest.fixture
def niri_abba():
    return [astrodata.from_file(download_from_archive(f)) for f in
            ('N20100602S0437.fits', 'N20100602S0438.fits',
             'N20100602S0439.fits', 'N20100602S0440.fits')]

@pytest.fixture
def niri_abcde():
    return [astrodata.from_file(download_from_archive(f)) for f in
            ('N20070204S0098.fits',
             'N20070204S0099.fits',
             'N20070204S0100.fits',
             'N20070204S0101.fits',
             'N20070204S0102.fits',
             'N20070204S0103.fits',
             'N20070204S0104.fits')]


# ---- Tests ------------------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.nirils
def test_associate_sky_abba(niri_abba):

    p = NIRILongslit(niri_abba)
    # The perpendicular offsets are reported as 0.14" for some frames but the
    # slit width is only 0.1", so the WCS needs to be reset.
    p.prepare()
    p.separateSky()
    p.associateSky()

    a1, b1, b2, a2 = p.showList()

    a_frames = {'N20100602S0437_skyAssociated.fits',
                'N20100602S0440_skyAssociated.fits'}
    b_frames = {'N20100602S0438_skyAssociated.fits',
                'N20100602S0439_skyAssociated.fits'}

    # check that the A frames get the B frames as skies, and vice versa
    for ad in (a1, a2):
        assert set(ad.SKYTABLE['SKYNAME']) == b_frames
    for ad in (b1, b2):
        assert set(ad.SKYTABLE['SKYNAME']) == a_frames

@pytest.mark.dragons_remote_data
@pytest.mark.nirils
def test_associate_sky_pass_skies(niri_abba):

    in_sky_names = set([ad.filename for ad in niri_abba[1:3]])

    p = NIRILongslit(niri_abba)
    # Don't run separate sky to simulate resuming work with known skies.
    p.associateSky(sky=niri_abba[1:3])

    out_sky_names = set([ad.phu['ORIGNAME'] for ad in p.streams['sky']])

    assert in_sky_names == out_sky_names

@pytest.mark.dragons_remote_data
@pytest.mark.nirils
def test_associate_sky_use_all(niri_abba):

    in_sky_names = set([ad.filename for ad in niri_abba])

    p = NIRILongslit(niri_abba)
    p.prepare()
    p.separateSky()
    p.associateSky(distance=0, use_all=True)

    for ad in p.showList():
        skies = set([s.replace('_skyAssociated', '')
                     for s in ad.SKYTABLE['SKYNAME']])

        # Check that each AD has all the other frames as skies, but not itself.
        assert skies == in_sky_names - set([ad.phu['ORIGNAME']])

@pytest.mark.dragons_remote_data
@pytest.mark.nirils
def test_associate_sky_exclude_all(niri_abba):
    p = NIRILongslit(niri_abba)
    p.prepare()
    p.separateSky()
    # Offset is ~10" so this will exclude skies if 'use_all' is False.
    p.associateSky(distance=15)

    for ad in p.showList():
        with pytest.raises(AttributeError):
            ad.SKYTABLE

@pytest.mark.dragons_remote_data
@pytest.mark.nirils
def test_associate_sky_abcde(niri_abcde):
    results = {'N20070204S0098.fits': [1, 2, 3],
               'N20070204S0099.fits': [0, 2, 3, 4],
               'N20070204S0100.fits': [0, 1, 3, 4, 5],
               'N20070204S0101.fits': [0, 1, 2, 4, 5, 6],
               'N20070204S0102.fits': [1, 2, 3, 5, 6],
               'N20070204S0103.fits': [2, 3, 4, 6],
               'N20070204S0104.fits': [3, 4, 5]}

    # data = [astrodata.from_file(download_from_archive(f)) for f in niri_abcde]

    p = NIRILongslit(niri_abcde)
    # Some frames have bad WCS information, so use 'fix' to take care of it.
    p.prepare(bad_wcs='fix')
    p.separateSky()
    p.associateSky()

    for ad in p.showList():
        skies = set([s for s in ad.SKYTABLE['SKYNAME']])

        assert skies == set([niri_abcde[i].filename
                             for i in results[ad.phu['ORIGNAME']]])

@pytest.mark.dragons_remote_data
@pytest.mark.nirils
def test_associate_sky_abcde_exclude_some(niri_abcde):
    # data = [astrodata.from_file(download_from_archive(f)) for f in niri_abcde]

    p = NIRILongslit(niri_abcde)
    p.prepare(bad_wcs='fix')
    p.separateSky()
    # This should eliminate all but two of the frames from being matched with
    # sky (each other, in this case)
    p.associateSky(distance=21.)

    assert p.showList()[0].SKYTABLE['SKYNAME'][0] == 'N20070204S0104_skyAssociated.fits'
    assert p.showList()[1].SKYTABLE['SKYNAME'][0] == 'N20070204S0099_skyAssociated.fits'
    assert len(p.streams['no_skies']) == 5
