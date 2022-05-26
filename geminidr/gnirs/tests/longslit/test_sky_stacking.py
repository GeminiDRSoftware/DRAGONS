#!/usr/bin/env python3
"""
Tests for sky stacking and subtraction for GNIRS.
"""

import astrodata
from astrodata.test import download_from_archive
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
import pytest


# ---- Tests ------------------------------------------------------------------
# These files form a GNIRS ABBA sequence:
# N20141119S0331.fits
# N20141119S0332.fits
# N20141119S0333.fits
# N20141119S0334.fits
