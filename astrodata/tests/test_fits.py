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


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_ad_basics(input_file):
    fname = download_from_archive(input_file)
    ad = astrodata.open(fname)

    assert isinstance(ad, astrodata.AstroDataFits)
    assert ad.filename == os.path.basename(fname)
    assert type(ad[0].data) == np.ndarray

    metadata = (('SCI', 1), ('SCI', 2), ('SCI', 3))

    for ext, md in zip(ad, metadata):
        assert ext.hdr['EXTNAME'] == md[0]
        assert ext.hdr['EXTVER'] == md[1]


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_can_add_and_del_extension(input_file):
    fname = download_from_archive(input_file)
    ad = astrodata.open(fname)
    original_size = len(ad)

    ourarray = np.array([(1, 2, 3), (11, 12, 13), (21, 22, 23)])
    ad.append(ourarray)
    assert len(ad) == original_size + 1

    del ad[original_size]
    assert len(ad) == original_size


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_slice(input_file):
    fname = download_from_archive(input_file)
    ad = astrodata.open(fname)
    assert ad.is_sliced is False

    # single
    try:
        metadata = ('SCI', 2)
        ext = ad[1]
    except IndexError:
        assert len(ad) == 1  # some files does not have enough extensions
    else:
        assert ext.is_single is True
        assert ext.is_sliced is True
        assert ext.hdr['EXTNAME'] == metadata[0]
        assert ext.hdr['EXTVER'] == metadata[1]

    # multiple
    metadata = ('SCI', 2), ('SCI', 3)
    try:
        slc = ad[1, 2]
    except IndexError:
        assert len(ad) == 1  # some files does not have enough extensions
    else:
        assert len(slc) == 2
        assert ext.is_sliced is True
        for ext, md in zip(slc, metadata):
            assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md

    # iterate over single slice
    metadata = ('SCI', 1)
    for ext in ad[0]:
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata

    # slice negative
    assert ad.data[-1] is ad[-1].data


@pytest.mark.dragons_remote_data
def test_phu():
    fname = download_from_archive(test_files[0])
    ad = astrodata.open(fname)

    # The result of this depends if gemini_instruments was imported or not
    # assert ad.descriptors == ('instrument', 'object', 'telescope')
    # assert ad.tags == set()

    assert ad.instrument() == 'NIFS'
    assert ad.object() == 'Dark'
    assert ad.telescope() == 'Gemini-North'

    ad.phu['DETECTOR'] = 'FooBar'
    ad.phu['ARBTRARY'] = 'BarBaz'

    assert ad.phu['DETECTOR'] == 'FooBar'
    assert ad.phu['ARBTRARY'] == 'BarBaz'

    if ad.instrument().upper() not in ['GNIRS', 'NIRI', 'F2']:
        del ad.phu['DETECTOR']
        assert 'DETECTOR' not in ad.phu


@pytest.mark.dragons_remote_data
def test_writes_to_new_fits(path_to_outputs):
    fname = download_from_archive(test_files[0])
    ad = astrodata.open(fname)

    testfile = os.path.join(path_to_outputs, 'temp.fits')
    ad.write(testfile)
    assert os.path.exists(testfile)

    # overwriting is forbidden by default
    with pytest.raises(OSError):
        ad.write(testfile)

    ad.write(testfile, overwrite=True)
    assert os.path.exists(testfile)


@pytest.mark.dragons_remote_data
def test_from_hdulist():
    fname = download_from_archive(test_files[0])

    with fits.open(fname) as hdul:
        ad = astrodata.open(hdul)
        assert ad.instrument() == 'NIFS'
        assert ad.object() == 'Dark'
        assert ad.telescope() == 'Gemini-North'
        assert len(ad) == 1
        assert ad[0].shape == (2048, 2048)


def test_can_make_and_write_ad_object(tmpdir):
    # Creates data and ad object
    phu = fits.PrimaryHDU()
    pixel_data = np.random.rand(100, 100)

    hdu = fits.ImageHDU()
    hdu.data = pixel_data

    ad = astrodata.create(phu)
    ad.append(hdu, name='SCI')

    # Write file and test it exists properly
    testfile = str(tmpdir.join('created_fits_file.fits'))
    ad.write(testfile)

    # Opens file again and tests data is same as above
    adnew = astrodata.open(testfile)
    assert np.array_equal(adnew[0].data, pixel_data)


def test_can_append_table_and_access_data(capsys, tmpdir):
    tbl = Table([np.zeros(10), np.ones(10)], names=['col1', 'col2'])
    phu = fits.PrimaryHDU()
    ad = astrodata.create(phu)
    astrodata.add_header_to_table(tbl)
    ad.append(tbl, name='BOB')
    assert ad.exposed == {'BOB'}

    ad.info()
    captured = capsys.readouterr()
    assert '.BOB           Table       (10, 2)' in captured.out

    # Write file and test it exists properly
    testfile = str(tmpdir.join('created_fits_file.fits'))
    ad.write(testfile)
    adnew = astrodata.open(testfile)
    assert adnew.exposed == {'BOB'}
    assert len(adnew.BOB) == 10


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_set_a_keyword_on_phu_deprecated(input_file):
    fname = download_from_archive(input_file)
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
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("input_file", test_files)
def test_do_arith_and_retain_features(input_file):
    fname = download_from_archive(input_file)
    ad = astrodata.open(fname)

    ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
    ad2 = ad * 5
    np.testing.assert_array_almost_equal(ad[0].NEW_FEATURE, ad2[0].NEW_FEATURE)


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


@pytest.mark.dragons_remote_data
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


def test_invalid_file(tmpdir, caplog):
    testfile = str(tmpdir.join('test.fits'))
    with open(testfile, 'w'):
        # create empty file
        pass

    with pytest.raises(astrodata.AstroDataError):
        astrodata.open(testfile)

    assert caplog.records[0].message.endswith('is zero size')


if __name__ == '__main__':
    pytest.main()
