
import os
import numpy as np
import pytest
import tempfile

import astrodata


@pytest.mark.ad_local_data
def test_can_read_data(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    test_data_full_name = os.path.join(test_path, test_data_name)

    assert os.path.exists(test_data_full_name)
    ad = astrodata.open(test_data_full_name)


@pytest.mark.ad_local_data
def test_can_return_ad_length(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    assert len(ad) == 3


@pytest.mark.ad_local_data
def test_iterate_over_extensions(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))
    for ext, md in zip(ad, metadata):
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md


@pytest.mark.ad_local_data
def test_slice_range(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    metadata = ('SCI', 2), ('SCI', 3)
    slc = ad[1:]
    
    assert len(slc) == 2
    
    for ext, md in zip(slc, metadata):
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md


@pytest.mark.ad_local_data
def test_slice_multiple(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))
    metadata = ('SCI', 2), ('SCI', 3)
    slc = ad[1, 2]
    
    assert len(slc) == 2
    
    for ext, md in zip(slc, metadata):
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md


@pytest.mark.ad_local_data
def test_slice_single(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    metadata = ('SCI', 2)
    ext = ad[1]

    assert ext.is_single
    assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata


@pytest.mark.ad_local_data
def test_iterate_over_single_slice(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    metadata = ('SCI', 1)
    
    for ext in ad[0]:
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata


@pytest.mark.ad_local_data
def test_slice_negative(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    assert ad.data[-1] is ad[-1].data


@pytest.mark.ad_local_data
def test_read_a_keyword_from_phu(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))
    
    assert ad.phu['DETECTOR'] == 'GMOS + Red1'


@pytest.mark.ad_local_data
def test_read_a_keyword_from_hdr(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    assert ad.hdr['CCDNAME'] == ['EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03']


@pytest.mark.ad_local_data
def test_set_a_keyword_on_phu(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    ad.phu['DETECTOR'] = 'FooBar'
    ad.phu['ARBTRARY'] = 'BarBaz'

    assert ad.phu['DETECTOR'] == 'FooBar'
    assert ad.phu['ARBTRARY'] == 'BarBaz'


@pytest.mark.ad_local_data
def test_remove_a_keyword_from_phu(test_path):

    test_data_name = "GMOS/N20110826S0336.fits"
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    del ad.phu['DETECTOR']

    assert 'DETECTOR' not in ad.phu


# Access to headers: DEPRECATED METHODS
# These should fail at some point
@pytest.mark.skip(reason="Deprecated methods")
def test_read_a_keyword_from_phu_deprecated():
    ad = astrodata.open('GMOS/N20110826S0336.fits')
    with pytest.raises(AttributeError):
        assert ad.phu.DETECTOR == 'GMOS + Red1'


@pytest.mark.skip(reason="Deprecated methods")
def test_read_a_keyword_from_hdr_deprecated():
    ad = astrodata.open('GMOS/N20110826S0336.fits')
    with pytest.raises(AttributeError):
        assert ad.hdr.CCDNAME == ['EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03']


@pytest.mark.skip(reason="Deprecated methods")
def test_set_a_keyword_on_phu_deprecated():
    ad = astrodata.open('GMOS/N20110826S0336.fits')
    with pytest.raises(AssertionError):
        ad.phu.DETECTOR = 'FooBar'
        ad.phu.ARBTRARY = 'BarBaz'
        assert ad.phu.DETECTOR == 'FooBar'
        assert ad.phu.ARBTRARY == 'BarBaz'
        assert ad.phu['DETECTOR'] == 'FooBar'


@pytest.mark.skip(reason="Deprecated methods")
def test_remove_a_keyword_from_phu_deprecated():
    ad = astrodata.open('GMOS/N20110826S0336.fits')
    with pytest.raises(AttributeError):
        del ad.phu.DETECTOR
        assert 'DETECTOR' not in ad.phu


# Regression:
# Make sure that references to associated extension objects are copied across
@pytest.mark.ad_local_data
def test_do_arith_and_retain_features(test_path):

    test_data_name = 'NIFS/N20160727S0077.fits'
    ad = astrodata.open(os.path.join(test_path, test_data_name))

    ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
    ad2 = ad * 5

    np.testing.assert_array_almost_equal(ad[0].NEW_FEATURE, ad2[0].NEW_FEATURE)


# Trying to access a missing attribute in the data provider should raise an
# AttributeError
@pytest.mark.skip(reason="uses chara")
def test_raise_attribute_error_when_accessing_missing_extenions():
    ad = from_chara('N20131215S0202_refcatAdded.fits')
    with pytest.raises(AttributeError) as excinfo:
        ad.ABC


# Some times, internal changes break the writing capability. Make sure that
# this is the case, always
@pytest.mark.skip(reason="uses chara")
def test_write_without_exceptions():
    # Use an image that we know contains complex structure
    ad = from_chara('N20131215S0202_refcatAdded.fits')
    with tempfile.TemporaryFile() as tf:
        ad.write(tf)
