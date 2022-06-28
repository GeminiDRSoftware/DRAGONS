#!/usr/bin/env python3
"""
Tests for sky stacking and subtraction for F2.
"""

import pytest

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.f2.primitives_f2_longslit import F2Longslit


# ---- Tests ------------------------------------------------------------------
def test_associate_sky_abba():

    files = ['S20200301S0071.fits', 'S20200301S0072.fits',
             'S20200301S0073.fits', 'S20200301S0074.fits']

    data = [astrodata.open(download_from_archive(f)) for f in files]

    p = F2Longslit(data)
    p.prepare()
    p.separateSky()
    p.associateSky()

    A1, B1, B2, A2 = p.showList()
    A_frames = {'S20200301S0071_skyAssociated.fits',
                'S20200301S0074_skyAssociated.fits'}
    B_frames = {'S20200301S0072_skyAssociated.fits',
                'S20200301S0073_skyAssociated.fits'}

    for ad in (A1, A2):
        assert set(ad.SKYTABLE['SKYNAME']) == B_frames
    for ad in (B1, B2):
        assert set(ad.SKYTABLE['SKYNAME']) == A_frames

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
