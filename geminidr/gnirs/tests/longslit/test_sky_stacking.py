"""
Tests for sky stacking and subtraction for GNIRS longslit spectroscopy.
"""

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit


# ---- Tests ------------------------------------------------------------------
def test_associate_sky_abba():

    files = ['N20141119S0331.fits', 'N20141119S0332.fits',
             'N20141119S0333.fits', 'N20141119S0334.fits']

    data = [astrodata.open(download_from_archive(f)) for f in files]

    p = GNIRSLongslit(data)
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
