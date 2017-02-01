import sys
import pytest
import tempfile
import os

import numpy as np

import astrodata
import gemini_instruments
from astropy.nddata import NDData
from astropy.io.fits import ImageHDU
from astropy.table import Table

from common_astrodata_test import from_test_data, from_chara

# Object construction
def test_for_length():
    ad = from_test_data('GMOS/N20110826S0336.fits')
    assert len(ad) == 3

# Append basic elements

def test_append_array_to_root_no_name():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    ones = np.ones((100, 100))
    ad.append(ones)
    assert len(ad) == (lbefore + 1)
    assert ad[-1].data is ones
    assert ad.header[-1]['EXTNAME'] == 'SCI'
    assert ad.header[-1]['EXTVER'] == len(ad)

def test_append_array_to_root_with_name_sci():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    ones = np.ones((100, 100))
    ad.append(ones, name='SCI')
    assert len(ad) == (lbefore + 1)
    assert ad[-1].data is ones
    assert ad.header[-1]['EXTNAME'] == 'SCI'
    assert ad.header[-1]['EXTVER'] == len(ad)

def test_append_array_to_root_with_arbitrary_name():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    ones = np.ones((100, 100))
    with pytest.raises(ValueError) as excinfo:
        ad.append(ones, name='ARBITRARY')

def test_append_array_to_extension_with_name_sci():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    ones = np.ones((100, 100))
    with pytest.raises(ValueError) as excinfo:
        ad[0].append(ones, name='SCI')

def test_append_array_to_extension_with_arbitrary_name():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    ones = np.ones((100, 100))
    ad[0].append(ones, name='ARBITRARY')
    assert len(ad) == lbefore
    assert ad[0].ARBITRARY is ones

def test_append_nddata_to_root_no_name():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    ones = np.ones((100, 100))
    hdu = ImageHDU(ones)
    nd = NDData(hdu.data)
    nd.meta['header'] = hdu.header
    ad.append(nd)
    assert len(ad) == (lbefore + 1)

def test_append_nddata_to_root_with_arbitrary_name():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    ones = np.ones((100, 100))
    hdu = ImageHDU(ones)
    nd = NDData(hdu.data)
    nd.meta['header'] = hdu.header
    hdu.header['EXTNAME'] = 'ARBITRARY'
    with pytest.raises(ValueError) as excinfo:
        ad.append(nd)

def test_append_table_to_root():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    table = Table(([1, 2, 3], [4, 5, 6], [7, 8, 9]),
                  names=('a', 'b', 'c'))
    ad.append(table, 'MYTABLE')
    assert (ad.MYTABLE == table).all()

def test_append_table_to_root_without_name():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    table = Table(([1, 2, 3], [4, 5, 6], [7, 8, 9]),
                  names=('a', 'b', 'c'))
    with pytest.raises(ValueError) as excinfo:
        ad.append(table)

def test_append_table_to_extension():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    lbefore = len(ad)
    table = Table(([1, 2, 3], [4, 5, 6], [7, 8, 9]),
                  names=('a', 'b', 'c'))
    ad[0].append(table, 'MYTABLE')
    assert (ad[0].MYTABLE == table).all()

# Append / assign Gemini specific

def test_append_dq_to_root():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    dq = np.zeros(ad[0].data.shape)
    with pytest.raises(ValueError):
        ad.append(dq, 'DQ')

def test_append_dq_to_ext():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    dq = np.zeros(ad[0].data.shape)
    ad[0].append(dq, 'DQ')
    assert dq is ad[0].mask

def test_append_var_to_root():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    var = np.random.random(ad[0].data.shape)
    with pytest.raises(ValueError):
        ad.append(var, 'VAR')

def test_append_var_to_ext():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    var = np.random.random(ad[0].data.shape)
    ad[0].append(var, 'VAR')
    assert np.abs(var - ad[0].variance).mean() < 0.00000001

# Append AstroData slices

def test_append_single_slice():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    ad2 = from_test_data('GMOS/N20110826S0336.fits')
    lbefore = len(ad2)
    last_ever = ad2[-1].nddata.meta['header'].get('EXTVER', -1)
    ad2.append(ad[1])
    assert len(ad2) == (lbefore + 1)
    assert np.all(ad2[-1].data == ad[1].data)
    assert last_ever < ad2[-1].nddata.meta['header'].get('EXTVER', -1)

def test_append_non_single_slice():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    ad2 = from_test_data('GMOS/N20110826S0336.fits')
    with pytest.raises(ValueError):
        ad2.append(ad[1:])

def test_append_whole_instance():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    ad2 = from_test_data('GMOS/N20110826S0336.fits')
    with pytest.raises(ValueError):
        ad2.append(ad)

def test_append_slice_to_extension():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    ad2 = from_test_data('GMOS/N20110826S0336.fits')
    with pytest.raises(ValueError):
        ad2[0].append(ad[0], name="FOOBAR")

# Deletion

def test_delete_named_attribute_at_top_level():
    ad = from_test_data('NIRI/N20131215S0202_refcatAdded.fits')
    assert 'REFCAT' in ad.tables
    del ad.REFCAT
    assert 'REFCAT' not in ad.tables

def test_delete_named_associated_extension():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    table = Table(([1, 2, 3], [4, 5, 6], [7, 8, 9]),
                  names=('a', 'b', 'c'))
    ad[0].append(table, 'MYTABLE')
    assert 'MYTABLE' in ad[0]
    del ad[0].MYTABLE
    assert 'MYTABLE' not in ad[0]

def test_delete_arbitrary_attribute_from_ad():
    ad = from_test_data('GMOS/N20160524S0119.fits')
    with pytest.raises(AttributeError):
        ad.arbitrary
    ad.arbitrary = 15
    assert ad.arbitrary == 15
    del ad.arbitrary
    with pytest.raises(AttributeError):
        ad.arbitrary
