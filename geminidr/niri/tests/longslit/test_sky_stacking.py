"""
Tests for sky stacking and subtraction for NIRI longslit spectroscopy.
"""

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.niri.primitives_niri_longslit import NIRILongslit

# ---- Tests ------------------------------------------------------------------
def test_associate_sky_abba():

    files = ['N20070204S0097.fits', 'N20070204S0098.fits',
             'N20070204S0099.fits', 'N20070204S0100.fits']

    data = [astrodata.open(download_from_archive(f)) for f in files]

    p = NIRILongslit(data)
    p.prepare(bad_wcs='fix')
    p.separateSky()
    p.associateSky()

    a1, a2 = p.streams['main']
    b1, b2 = p.streams['sky']

    b_frames = {'N20070204S0098_skyAssociated.fits',
                'N20070204S0100_skyAssociated.fits'}

    # check that the A frames get the B frames as skies
    for ad in (a1, a2):
        assert set(ad.SKYTABLE['SKYNAME']) == b_frames
