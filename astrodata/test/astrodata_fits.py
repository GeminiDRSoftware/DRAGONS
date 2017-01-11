import pytest
import tempfile
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

# Slicing and iterating
def test_for_length():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    assert len(ad) == 3

def test_iterate_over_extensions():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))
    for ext, md in zip(ad, metadata):
        assert (ext.hdr.EXTNAME, ext.hdr.EXTVER) == md

def test_slice_multiple():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    metadata = ('SCI', 2), ('SCI', 3)
    slc = ad[1:]
    assert len(slc) == 2
    for ext, md in zip(slc, metadata):
        assert (ext.hdr.EXTNAME, ext.hdr.EXTVER) == md

def test_slice_single():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    metadata = ('SCI', 2)
    ext = ad[1]
    assert ext.is_single
    assert (ext.hdr.EXTNAME, ext.hdr.EXTVER) == metadata

def test_iterate_over_single_slice():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    metadata = ('SCI', 1)
    for ext in ad[0]:
        assert (ext.hdr.EXTNAME, ext.hdr.EXTVER) == metadata

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

# Some times, internal changes break the writing capability. Make sure that
# this is the case, always
def test_write_without_exceptions():
    # Use an image that we know contains complex structure
    ad = from_chara('N20131215S0202_refcatAdded.fits')
    with tempfile.TemporaryFile() as tf:
        ad.write(tf)
