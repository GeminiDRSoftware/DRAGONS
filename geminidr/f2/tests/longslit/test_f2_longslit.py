#!/usr/bin/env python3
"""
Tests for F2Longslit.
"""

import pytest

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.f2.primitives_f2_longslit import F2Longslit

# ---- Tests ------------------------------------------------------------------
@pytest.mark.f2ls
@pytest.mark.dragons_remote_data
def test_addMDF():

    p = F2Longslit([astrodata.from_file(
            download_from_archive('S20140605S0101.fits'))])
    ad = p.prepare()[0]  # Includes addMDF() as a step.

    assert hasattr(ad, 'MDF')
    assert ad.MDF['x_ccd']
    assert ad.MDF['slitlength_arcsec']
    assert ad.MDF['slitlength_pixels']
