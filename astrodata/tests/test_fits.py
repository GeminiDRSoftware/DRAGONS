import glob
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import astrodata
from astrodata.testing import download_from_archive

test_files = [
    "N20160727S0077.fits",  # NIFS DARK
    "N20170529S0168.fits",  # GMOS-N SPECT
    "N20190116G0054i.fits",  # GRACES SPECT
    "N20190120S0287.fits",  # NIRI IMAGE
    "N20190206S0279.fits",  # GNIRS SPECT XD
    "S20150609S0023.fits",  # GSAOI DARK
    "S20170103S0032.fits",  # F2 IMAGE
    "S20170505S0031.fits",  # GSAOI FLAT
    "S20170505S0095.fits",  # GSAOI IMAGE
    "S20171116S0078.fits",  # GMOS-S MOS NS
    "S20180223S0229.fits",  # GMOS IFU ACQUISITION
    "S20190213S0084.fits",  # F2 IMAGE
]


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_file_exists(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    assert os.path.exists(os.path.join(path_to_inputs, fname)), \
        "File does not exists: {:s}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_can_open_fits_file(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(os.path.join(path_to_inputs, fname))

    assert isinstance(ad, astrodata.fits.AstroDataFits), \
        "Could not open file: {:s}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_basename_is_properly_set(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)
    basename = os.path.basename(fname)

    assert ad.filename == basename, \
        ".filename property does not match input file name for file " \
        "{:s}".format(basename)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_can_add_and_del_extension(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    original_size = len(ad)

    ourarray = np.array([(1, 2, 3), (11, 12, 13), (21, 22, 23)])
    ad.append(ourarray)

    assert len(ad) == (original_size + 1), \
        "Could not append extension to ad: {:s}".format(fname)

    del ad[original_size]

    assert len(ad) == original_size, \
        "Could not remove extension from ad: {:s}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_extension_data_is_an_array(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    assert type(ad[0].data) == np.ndarray, \
        "Expected data type {} for {} but found {}".format(
            np.ndarray, fname, type(ad[0].data))


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_iterate_over_extensions(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))

    for ext, md in zip(ad, metadata):
        assert ext.hdr['EXTNAME'] == md[0], \
            "Mismatching EXTNAME for file {:s}".format(fname)
        assert ext.hdr['EXTVER'] == md[1], \
            "Mismatching EXTVER for file {:s}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_slice_multiple(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    metadata = ('SCI', 2), ('SCI', 3)

    try:
        slc = ad[1, 2]

    except IndexError:
        assert len(ad) == 1

    else:
        assert len(slc) == 2
        for ext, md in zip(slc, metadata):
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md, \
                "Test failed for file: {:s}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_slice_single(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    try:
        metadata = ('SCI', 2)
        ext = ad[1]

    except IndexError:
        assert len(ad) == 1, \
            "Mismatching number of extensions for file {:s}".format(
                fname)

    else:
        assert ext.is_single, \
            "Mismatching number of extensions for file {:s}".format(
                fname)

        assert ext.hdr['EXTNAME'] == metadata[0], \
            "Mismatching EXTNAME for file {:s}".format(fname)

        assert ext.hdr['EXTVER'] == metadata[1], \
            "Mismatching EXTVER for file {:s}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_iterate_over_single_slice(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    metadata = ('SCI', 1)

    for ext in ad[0]:
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata, \
            "Assertion failed for file: {}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_slice_negative(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    assert ad.data[-1] is ad[-1].data, \
        "Assertion failed for file: {}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_set_a_keyword_on_phu(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    ad.phu['DETECTOR'] = 'FooBar'
    ad.phu['ARBTRARY'] = 'BarBaz'

    assert ad.phu['DETECTOR'] == 'FooBar', \
        "Assertion failed for file: {}".format(fname)

    assert ad.phu['ARBTRARY'] == 'BarBaz', \
        "Assertion failed for file: {}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_remove_a_keyword_from_phu(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    if ad.instrument().upper() not in ['GNIRS', 'NIRI', 'F2']:
        del ad.phu['DETECTOR']
        assert 'DETECTOR' not in ad.phu, \
            "Assertion failed for file: {}".format(fname)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_writes_to_new_fits(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    test_file_location = os.path.join(path_to_inputs, 'temp.fits')

    if os.path.exists(test_file_location):
        os.remove(test_file_location)

    ad.write(test_file_location)
    assert os.path.exists(test_file_location)

    os.remove(test_file_location)


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_can_overwrite_existing_file(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    test_file_location = os.path.join(path_to_inputs, 'temp_overwrite.fits')

    if os.path.exists(test_file_location):
        os.remove(test_file_location)

    ad.write(test_file_location)

    assert os.path.exists(test_file_location)

    adnew = astrodata.open(test_file_location)
    adnew.write(overwrite=True)

    # erasing file for cleanup
    os.remove(test_file_location)


def test_can_make_and_write_ad_object(path_to_inputs):
    # Creates data and ad object
    phu = fits.PrimaryHDU()
    pixel_data = np.random.rand(100, 100)

    hdu = fits.ImageHDU()
    hdu.data = pixel_data

    ad = astrodata.create(phu)
    ad.append(hdu, name='SCI')

    # Write file and test it exists properly
    test_file_location = os.path.join(
        path_to_inputs, 'created_fits_file.fits')

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


@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_set_a_keyword_on_phu_deprecated(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

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
@pytest.mark.remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_do_arith_and_retain_features(input_file, path_to_inputs):
    fname = download_from_archive(input_file, path=path_to_inputs)
    ad = astrodata.open(fname)

    ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
    ad2 = ad * 5

    np.testing.assert_array_almost_equal(
        ad[0].NEW_FEATURE, ad2[0].NEW_FEATURE)


def test_update_filename():
    phu = fits.PrimaryHDU()
    ad = astrodata.create(phu)
    ad.filename = 'myfile.fits'

    # This will also set ORIGNAME='myfile.fits'
    ad.update_filename(suffix='_suffix1')
    assert ad.filename == 'myfile_suffix1.fits'

    ad.update_filename(suffix='_suffix2', strip=True)
    assert ad.filename == 'myfile_suffix2.fits'

    ad.update_filename(suffix='_suffix1', strip=False)
    assert ad.filename == 'myfile_suffix2_suffix1.fits'

    ad.filename = 'myfile.fits'
    ad.update_filename(prefix='prefix_', strip=True)
    assert ad.filename == 'prefix_myfile.fits'

    ad.update_filename(suffix='_suffix', strip=True)
    assert ad.filename == 'prefix_myfile_suffix.fits'

    ad.update_filename(prefix='', suffix='_suffix2', strip=True)
    assert ad.filename == 'myfile_suffix2.fits'

    # Now check that updates are based on existing filename
    # (so "myfile" shouldn't appear)
    ad.filename = 'file_suffix1.fits'
    ad.update_filename(suffix='_suffix2')
    assert ad.filename == 'file_suffix1_suffix2.fits'

    # A suffix shouldn't have an underscore, so should assume that
    # "file_suffix1" is the root
    ad.update_filename(suffix='_suffix3', strip=True)
    assert ad.filename == 'file_suffix1_suffix3.fits'


@pytest.mark.remote_data
def test_read_a_keyword_from_phu_deprecated():
    "Test deprecated methods to access headers"
    ad = astrodata.open(
        download_from_archive('N20110826S0336.fits', path='GMOS'))

    with pytest.raises(AttributeError):
        assert ad.phu.DETECTOR == 'GMOS + Red1'

    with pytest.raises(AttributeError):
        assert ad.hdr.CCDNAME == [
            'EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03'
        ]

    # and when accessing missing extension
    with pytest.raises(AttributeError):
        ad.ABC


if __name__ == '__main__':
    pytest.main()
