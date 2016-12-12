import pytest
import os

import numpy as np

import astrodata
import gemini_instruments

THIS_DIR = os.path.dirname(__file__)

# Tests to perform:

# Opening a FITS file
# Length
# Slicing
# Slicing to single

# Regression:

def test_do_arith_and_retain_features():
    ad = astrodata.open(os.path.join(THIS_DIR, 'test_data/NIFS/N20160727S0077.fits'))
    ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
    ad2 = ad * 5
    ad[0].NEW_FEATURE == ad2[0].NEW_FEATURE
