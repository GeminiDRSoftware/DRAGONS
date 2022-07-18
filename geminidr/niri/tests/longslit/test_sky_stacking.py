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
def abcde_files():
    return np.array(['N20070204S0098.fits',
                     'N20070204S0099.fits',
                     'N20070204S0100.fits',
                     'N20070204S0101.fits',
                     'N20070204S0102.fits',
                     'N20070204S0103.fits',
                     'N20070204S0104.fits'])


# ---- Tests ------------------------------------------------------------------
def test_associate_sky_abcde(abcde_files):

    results = {'N20070204S0098.fits': [1, 2, 3],
               'N20070204S0099.fits': [0, 2, 3, 4],
               'N20070204S0100.fits': [0, 1, 3, 4, 5],
               'N20070204S0101.fits': [0, 1, 2, 4, 5, 6],
               'N20070204S0102.fits': [1, 2, 3, 5, 6],
               'N20070204S0103.fits': [2, 3, 4, 6],
               'N20070204S0104.fits': [3, 4, 5]}

    data = [astrodata.open(download_from_archive(f)) for f in abcde_files]

    p = NIRILongslit(data)
    p.prepare(bad_wcs='fix')
    p.separateSky()
    p.associateSky()

    for ad in p.showList():
        skies = set([s.replace('_skyAssociated', '')
                     for s in ad.SKYTABLE['SKYNAME']])
        assert skies == set(abcde_files[results[ad.phu['ORIGNAME']]])

def test_associate_sky_abcde_distance(abcde_files):
    data = [astrodata.open(download_from_archive(f)) for f in abcde_files]

    p = NIRILongslit(data)
    p.prepare(bad_wcs='fix')
    p.separateSky()
    # This should eliminate all but two of the frames from being matched with
    # sky (each other, in this case)
    p.associateSky(distance=21.)

    for ad in p.showList():
        print(ad.filename)
    assert p.showList()[0].SKYTABLE['SKYNAME'][0] == 'N20070204S0104_skyAssociated.fits'
    assert p.showList()[1].SKYTABLE['SKYNAME'][0] == 'N20070204S0099_skyAssociated.fits'
    assert len(p.streams['no_skies']) == 5
