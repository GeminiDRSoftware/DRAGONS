#!/usr/bin/env python3
"""
Tests for NIRILongslit.
"""

import pytest

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.niri.primitives_niri_longslit import NIRILongslit

# ---- Tests ------------------------------------------------------------------
@pytest.mark.nirils
@pytest.mark.dragons_remote_data
def test_addMDF():

    p = NIRILongslit([astrodata.open(
            download_from_archive('N20100620S0116.fits'))])
    ad = p.prepare()[0]  # Includes addMDF() as a step.

    assert hasattr(ad, 'MDF')
    assert ad.MDF['y_ccd']
    assert ad.MDF['slitlength_arcsec']
    assert ad.MDF['slitlength_pixels']
