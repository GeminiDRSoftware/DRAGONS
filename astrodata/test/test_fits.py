import glob
import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import astrodata


def test_file_exists(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    for _file in list_of_files:
        assert os.path.exists(os.path.join(input_test_path, _file)), \
            "File does not exists: {:s}".format(_file)


def test_can_open_fits_file(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))

    for _file in list_of_files:
        ad = astrodata.open(os.path.join(input_test_path, _file))
        assert isinstance(ad, astrodata.fits.AstroDataFits), \
            "Could not open file: {:s}".format(_file)


def test_basename_is_properly_set(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))

    for _file in list_of_files:
        ad = astrodata.open(_file)
        basename = os.path.basename(_file)
        assert ad.filename == basename, \
            ".filename property does not match input file name for file " \
            "{:s}".format(basename)


def test_can_add_and_del_extension(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))

    for _file in list_of_files:
        ad = astrodata.open(_file)
        original_size = len(ad)

        ourarray = np.array([(1, 2, 3), (11, 12, 13), (21, 22, 23)])
        ad.append(ourarray)

        assert len(ad) == (original_size + 1), \
            "Could not append extension to ad: {:s}".format(_file)

        del ad[original_size]

        assert len(ad) == original_size, \
            "Could not remove extension from ad: {:s}".format(_file)


def test_extension_data_is_an_array(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))

    for _file in list_of_files:
        ad = astrodata.open(_file)
        assert type(ad[0].data) == np.ndarray, \
            "Expected data type {} for {} but found {}".format(
                np.ndarray, _file, type(ad[0].data))


def test_iterate_over_extensions(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))

    for _file in list_of_files:
        ad = astrodata.open(_file)

        for ext, md in zip(ad, metadata):
            assert ext.hdr['EXTNAME'] == md[0], \
                "Mismatching EXTNAME for file {:s}".format(_file)
            assert ext.hdr['EXTVER'] == md[1], \
                "Mismatching EXTVER for file {:s}".format(_file)


def test_slice_multiple(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    metadata = ('SCI', 2), ('SCI', 3)

    for _file in list_of_files:
        ad = astrodata.open(_file)

        try:
            slc = ad[1, 2]

        except IndexError:
            assert len(ad) == 1

        else:
            assert len(slc) == 2
            for ext, md in zip(slc, metadata):
                assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md, \
                    "Test failed for file: {:s}".format(_file)


def test_slice_single(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    for _file in list_of_files:

        ad = astrodata.open(_file)

        try:
            metadata = ('SCI', 2)
            ext = ad[1]

        except IndexError:
            assert len(ad) == 1, \
                "Mismatching number of extensions for file {:s}".format(
                    _file)

        else:
            assert ext.is_single, \
                "Mismatching number of extensions for file {:s}".format(
                    _file)

            assert ext.hdr['EXTNAME'] == metadata[0], \
                "Mismatching EXTNAME for file {:s}".format(_file)

            assert ext.hdr['EXTVER'] == metadata[1], \
                "Mismatching EXTVER for file {:s}".format(_file)


def test_iterate_over_single_slice(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    for _file in list_of_files:

        ad = astrodata.open(_file)

        metadata = ('SCI', 1)

        for ext in ad[0]:
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata, \
                "Assertion failed for file: {}".format(_file)


def test_slice_negative(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    for _file in list_of_files:
        ad = astrodata.open(_file)
        assert ad.data[-1] is ad[-1].data, \
            "Assertion failed for file: {}".format(_file)


def test_set_a_keyword_on_phu(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    for _file in list_of_files:
        ad = astrodata.open(_file)
        ad.phu['DETECTOR'] = 'FooBar'
        ad.phu['ARBTRARY'] = 'BarBaz'

        assert ad.phu['DETECTOR'] == 'FooBar', \
            "Assertion failed for file: {}".format(_file)

        assert ad.phu['ARBTRARY'] == 'BarBaz', \
            "Assertion failed for file: {}".format(_file)


def test_remove_a_keyword_from_phu(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    for _file in list_of_files:
        ad = astrodata.open(_file)

        if ad.instrument().upper() in ['GNIRS', 'NIRI', 'F2']:
            continue

        del ad.phu['DETECTOR']
        assert 'DETECTOR' not in ad.phu, \
            "Assertion failed for file: {}".format(_file)


def test_writes_to_new_fits(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    test_file_location = os.path.join(input_test_path, 'temp.fits')

    for _file in list_of_files:
        ad = astrodata.open(_file)

        if os.path.exists(test_file_location):
            os.remove(test_file_location)

        ad.write(test_file_location)

        assert os.path.exists(test_file_location)

    os.remove(test_file_location)


def test_can_overwrite_existing_file(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    test_file_location = os.path.join(input_test_path, 'temp_overwrite.fits')

    for _file in list_of_files:
        ad = astrodata.open(_file)

        if os.path.exists(test_file_location):
            os.remove(test_file_location)

        ad.write(test_file_location)

        assert os.path.exists(test_file_location)

        adnew = astrodata.open(test_file_location)
        adnew.write(overwrite=True)

        # erasing file for cleanup
        os.remove(test_file_location)


def test_can_make_and_write_ad_object(input_test_path):
    # Creates data and ad object
    phu = fits.PrimaryHDU()
    pixel_data = np.random.rand(100, 100)

    hdu = fits.ImageHDU()
    hdu.data = pixel_data

    ad = astrodata.create(phu)
    ad.append(hdu, name='SCI')

    # Write file and test it exists properly
    test_file_location = os.path.join(
        input_test_path, 'created_fits_file.fits')

    if os.path.exists(test_file_location):
        os.remove(test_file_location)
    ad.write(test_file_location)

    assert os.path.exists(test_file_location)
    # Opens file again and tests data is same as above

    adnew = astrodata.open(test_file_location)
    assert np.array_equal(adnew[0].data, pixel_data)
    os.remove(test_file_location)


def test_can_append_table_and_access_data():
    my_astropy_table = Table(list(np.random.rand(2, 100)),
                             names=['col1', 'col2'])

    phu = fits.PrimaryHDU()
    ad = astrodata.create(phu)
    astrodata.add_header_to_table(my_astropy_table)

    ad.append(my_astropy_table, name='BOB')

    print(ad.info())


def test_set_a_keyword_on_phu_deprecated(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))
    for _file in list_of_files:
        ad = astrodata.open(_file)

    try:
        with pytest.raises(AssertionError):
            ad.phu.DETECTOR = 'FooBar'
            ad.phu.ARBTRARY = 'BarBaz'

            assert ad.phu.DETECTOR == 'FooBar'
            assert ad.phu.ARBTRARY == 'BarBaz'
            assert ad.phu['DETECTOR'] == 'FooBar'

    except KeyError as e:

        # Some instruments don't have DETECTOR as a keyword
        if e.args[0] == "Keyword 'DETECTOR' not found.":
            pass
        else:
            raise KeyError


# Regression:
# Make sure that references to associated
# extension objects are copied across
def test_do_arith_and_retain_features(input_test_path):
    list_of_files = glob.glob(os.path.join(input_test_path, "*fits"))

    for _file in list_of_files:
        ad = astrodata.open(_file)

        ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
        ad2 = ad * 5

        np.testing.assert_array_almost_equal(
            ad[0].NEW_FEATURE, ad2[0].NEW_FEATURE)


# ============================================================================
# Deprecated tests

@pytest.mark.skip(reason="Deprecated methods")
def test_remove_a_keyword_from_phu_deprecated(self):
    ad = astrodata.open('N20110826S0336.fits')
    with pytest.raises(AttributeError):
        del ad.phu.DETECTOR
        assert 'DETECTOR' not in ad.phu

    # Regression:
    # Make sure that references to associated
    # extension objects are copied across
    # Trying to access a missing attribute in
    # the data provider should raise an
    # AttributeError
    test_data_name = "N20110826S0336.fits"


@pytest.mark.skip(reason="uses chara")
def test_raise_attribute_error_when_accessing_missing_extenions(self):
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


# Access to headers: DEPRECATED METHODS
# These should fail at some point
@pytest.mark.skip(reason="Deprecated methods")
def test_read_a_keyword_from_phu_deprecated():
    ad = astrodata.open('N20110826S0336.fits')

    with pytest.raises(AttributeError):
        assert ad.phu.DETECTOR == 'GMOS + Red1'


@pytest.mark.skip(reason="Deprecated methods")
def test_read_a_keyword_from_hdr_deprecated():
    ad = astrodata.open('N20110826S0336.fits')

    with pytest.raises(AttributeError):
        assert ad.hdr.CCDNAME == [
            'EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03'
        ]
