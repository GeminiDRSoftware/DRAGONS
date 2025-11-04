#!/usr/bin/env python3
"""
Tests for GNIRSLongslit.
"""

import pytest

import astrodata
from astrodata.testing import download_from_archive
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit

# ---- Tests ------------------------------------------------------------------
@pytest.mark.gnirsls
@pytest.mark.dragons_remote_data
def test_addMDF():

    p = GNIRSLongslit([astrodata.open(
            download_from_archive('N20100915S0138.fits'))])
    ad = p.prepare()[0]  # Includes addMDF() as a step.

    assert hasattr(ad, 'MDF')
    assert ad.MDF['x_ccd']
    assert ad.MDF['slitlength_arcsec']
    assert ad.MDF['slitlength_pixels']
