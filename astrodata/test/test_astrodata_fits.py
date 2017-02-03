import pytest
import tempfile
import os

import numpy as np

import astrodata
import gemini_instruments

from .common_astrodata_test import from_test_data, from_chara

# Object construction
def test_for_length():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    assert len(ad) == 3

# Slicing and iterating
def test_iterate_over_extensions():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))
    for ext, md in zip(ad, metadata):
        assert (ext.hdr.EXTNAME, ext.hdr.EXTVER) == md

def test_slice_range():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    metadata = ('SCI', 2), ('SCI', 3)
    slc = ad[1:]
    assert len(slc) == 2
    for ext, md in zip(slc, metadata):
        assert (ext.hdr.EXTNAME, ext.hdr.EXTVER) == md

def test_slice_multiple():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    metadata = ('SCI', 2), ('SCI', 3)
    slc = ad[1, 2]
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

def test_slice_negative():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    assert ad.data[-1] is ad[-1].data

# Access to headers

def test_read_a_keyword_from_phu():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    assert ad.phu['DETECTOR'] == 'GMOS + Red1'

def test_read_a_keyword_from_hdr():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    assert ad.hdr['CCDNAME'] == ['EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03']

def test_set_a_keyword_on_phu():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    ad.phu['DETECTOR'] = 'FooBar'
    ad.phu['ARBTRARY'] = 'BarBaz'
    assert ad.phu['DETECTOR'] == 'FooBar'
    assert ad.phu['ARBTRARY'] == 'BarBaz'

def test_remove_a_keyword_from_phu():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    del ad.phu['DETECTOR']
    assert 'DETECTOR' not in ad.phu

# Access to headers: DEPRECATED METHODS
# These should fail at some point

def test_read_a_keyword_from_phu_deprecated():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    assert ad.phu.DETECTOR == 'GMOS + Red1'

def test_read_a_keyword_from_hdr_deprecated():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    assert ad.hdr.CCDNAME == ['EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03']

def test_set_a_keyword_on_phu_deprecated():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    ad.phu.DETECTOR = 'FooBar'
    ad.phu.ARBTRARY = 'BarBaz'
    assert ad.phu.DETECTOR == 'FooBar'
    assert ad.phu.ARBTRARY == 'BarBaz'

def test_remove_a_keyword_from_phu_deprecated():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    del ad.phu.DETECTOR
    assert 'DETECTOR' not in ad.phu

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
