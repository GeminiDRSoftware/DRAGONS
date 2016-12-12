import pytest
import os

import numpy as np

import astrodata
import gemini_instruments

THIS_DIR = os.path.dirname(__file__)
CHARA = '/net/chara/data2/pub'

def from_test_data(fname):
    return astrodata.open(os.path.join(THIS_DIR, 'test_data', fname))

def from_chara(fname):
    return astrodata.open(os.path.join(CHARA, fname))

# Tests to perform:

# Opening a FITS file
# Length
# Slicing
# Slicing to single

# Regression:

# Make sure that references to associated extension objects are copied across
def test_do_arith_and_retain_features():
    ad = from_test_data('NIFS/N20160727S0077.fits')
    ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
    ad2 = ad * 5
    ad[0].NEW_FEATURE == ad2[0].NEW_FEATURE

# Trying to access a missing attribute in the data provider should raise an
# AttributeError
def test_raise_attribute_error_when_accessing_missing_extenions():
    ad = from_chara('N20131215S0202_refcatAdded.fits')
    with pytest.raises(AttributeError) as excinfo:
        ad.ABC
