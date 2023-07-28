import copy
import os
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import astrodata
from astrodata.utils import AstroDataDeprecationWarning
from astrodata.nddata import ADVarianceUncertainty, NDAstroData
from astrodata.testing import download_from_archive, compare_models
import astropy
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models

# test_files = [
#     "N20160727S0077.fits",  # NIFS DARK
#     "N20170529S0168.fits",  # GMOS-N SPECT
#     "N20190116G0054i.fits",  # GRACES SPECT
#     "N20190120S0287.fits",  # NIRI IMAGE
#     "N20190206S0279.fits",  # GNIRS SPECT XD
#     "S20150609S0023.fits",  # GSAOI DARK
#     "S20170103S0032.fits",  # F2 IMAGE
#     "S20170505S0031.fits",  # GSAOI FLAT
#     "S20170505S0095.fits",  # GSAOI IMAGE
#     "S20171116S0078.fits",  # GMOS-S MOS NS
#     "S20180223S0229.fits",  # GMOS IFU ACQUISITION
#     "S20190213S0084.fits",  # F2 IMAGE
# ]


@pytest.fixture(scope='module')
def NIFS_DARK():
    """
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU     144   ()
      1                1 ImageHDU       108   (2048, 2048)   float32
    """
    return download_from_archive("N20160727S0077.fits")


@pytest.fixture(scope='module')
def GMOSN_SPECT():
    """
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU     180   ()
      1               -1 ImageHDU       108   (288, 512)   int16 (rescales to uint16)
      2               -1 ImageHDU       108   (288, 512)   int16 (rescales to uint16)
      3               -1 ImageHDU       108   (288, 512)   int16 (rescales to uint16)
      4               -1 ImageHDU       108   (288, 512)   int16 (rescales to uint16)
      5               -1 ImageHDU        72   (288, 512)   int16 (rescales to uint16)
      6               -1 ImageHDU        72   (288, 512)   int16 (rescales to uint16)
      7               -1 ImageHDU        72   (288, 512)   int16 (rescales to uint16)
      8               -1 ImageHDU        72   (288, 512)   int16 (rescales to uint16)
      9               -1 ImageHDU        38   (288, 512)   int16 (rescales to uint16)
     10               -1 ImageHDU        38   (288, 512)   int16 (rescales to uint16)
     11               -1 ImageHDU        38   (288, 512)   int16 (rescales to uint16)
     12               -1 ImageHDU        38   (288, 512)   int16 (rescales to uint16)
    """
    return download_from_archive("N20170529S0168.fits")


@pytest.fixture(scope='module')
def GSAOI_DARK():
    """
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU     289   ()
      1                1 ImageHDU       144   (2048, 2048)   float32
      2                2 ImageHDU       144   (2048, 2048)   float32
      3                3 ImageHDU       144   (2048, 2048)   float32
      4                4 ImageHDU       144   (2048, 2048)   float32
    """
    return download_from_archive("S20150609S0023.fits")


@pytest.fixture(scope='module')
def GRACES_SPECT():
    """
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU     183   (190747, 28)   float32
    """
    return download_from_archive("N20190116G0054i.fits")


def test_extver(tmp_path):
    """Test that EXTVER is written sequentially for new extensions,
    and preserved with slicing.
    """
    testfile = tmp_path / 'test.fits'

    ad = astrodata.create({})
    for _ in range(10):
        ad.append(np.zeros((4, 5)))
    ad.write(testfile)

    ad = astrodata.open(testfile)
    ext = ad[2]
    assert ext.hdr['EXTNAME'] == 'SCI'
    assert ext.hdr['EXTVER'] == 3

    ext = ad[4]
    assert ext.hdr['EXTNAME'] == 'SCI'
    assert ext.hdr['EXTVER'] == 5

    ext = ad[:8][4]
    assert ext.hdr['EXTNAME'] == 'SCI'
    assert ext.hdr['EXTVER'] == 5


def test_extver2(tmp_path):
    """Test renumbering of EXTVER."""
    testfile = tmp_path / 'test.fits'

    ad = astrodata.create(fits.PrimaryHDU())
    data = np.arange(5)
    ad.append(fits.ImageHDU(data=data, header=fits.Header({'EXTVER': 2})))
    ad.append(fits.ImageHDU(data=data + 2, header=fits.Header({'EXTVER': 5})))
    ad.append(fits.ImageHDU(data=data + 5))
    ad.append(fits.ImageHDU(data=data + 7, header=fits.Header({'EXTVER': 3})))
    ad.write(testfile)

    ad = astrodata.open(testfile)
    assert [hdr['EXTVER'] for hdr in ad.hdr] == [1, 2, 3, 4]


def test_extver3(tmp_path, GSAOI_DARK):
    """Test that original EXTVER are preserved and extensions added
    from another object are renumbered.
    """
    testfile = tmp_path / 'test.fits'

    ad1 = astrodata.open(GSAOI_DARK)
    ad2 = astrodata.open(GSAOI_DARK)

    del ad1[2]
    ad1.append(ad2[2])
    ad1.append(np.zeros(10))

    ad1.write(testfile)

    ad = astrodata.open(testfile)
    assert [hdr['EXTVER'] for hdr in ad.hdr] == [1, 2, 4, 5, 6]


@pytest.mark.dragons_remote_data
def test_can_add_and_del_extension(GMOSN_SPECT):
    ad = astrodata.open(GMOSN_SPECT)
    original_size = len(ad)

    ourarray = np.array([(1, 2, 3), (11, 12, 13), (21, 22, 23)])
    ad.append(ourarray)
    assert len(ad) == original_size + 1

    del ad[original_size]
    assert len(ad) == original_size


@pytest.mark.dragons_remote_data
def test_slice(GMOSN_SPECT):
    ad = astrodata.open(GMOSN_SPECT)
    assert ad.is_sliced is False

    n_ext = len(ad)
    with pytest.raises(IndexError, match="Index out of range"):
        ad[n_ext + 1]

    with pytest.raises(ValueError, match='Invalid index: FOO'):
        ad['FOO']

    # single
    metadata = ('SCI', 2)
    ext = ad[1]
    assert ext.id == 2
    assert ext.is_single is True
    assert ext.is_sliced is True
    assert ext.hdr['EXTNAME'] == metadata[0]
    assert ext.hdr['EXTVER'] == metadata[1]
    assert not ext.is_settable('filename')
    assert ext.data[0, 0] == 387

    # when astrofaker is imported this will be recognized as AstroFakerGmos
    # instead of AstroData
    match = r"'Astro.*' object has no attribute 'FOO'"
    with pytest.raises(AttributeError, match=match):
        ext.FOO

    # setting uppercase attr adds to the extension:
    ext.FOO = 1
    assert ext.FOO == 1
    assert ext.exposed == {'FOO'}
    assert ext.nddata.meta['other']['FOO'] == 1
    del ext.FOO

    with pytest.raises(AttributeError):
        del ext.BAR

    match = "Can't append objects to slices, use 'ext.NAME = obj' instead"
    with pytest.raises(TypeError, match=match):
        ext.append(np.zeros(5))

    # but lowercase just adds a normal attribute to the object
    ext.bar = 1
    assert ext.bar == 1
    assert 'bar' not in ext.nddata.meta['other']
    del ext.bar

    with pytest.raises(TypeError, match="Can't slice a single slice!"):
        ext[1]


@pytest.mark.dragons_remote_data
def test_slice_single_element(GMOSN_SPECT):
    ad = astrodata.open(GMOSN_SPECT)
    assert ad.is_sliced is False

    metadata = ('SCI', 2)

    ext = ad[1:2]
    assert ext.is_single is False
    assert ext.is_sliced is True
    assert ext.indices == [1]
    assert isinstance(ext.data, list) and len(ext.data) == 1

    ext = ext[0]
    assert ext.id == 2
    assert ext.is_single is True
    assert ext.is_sliced is True
    assert ext.hdr['EXTNAME'] == metadata[0]
    assert ext.hdr['EXTVER'] == metadata[1]


@pytest.mark.dragons_remote_data
def test_slice_multiple(GMOSN_SPECT):
    ad = astrodata.open(GMOSN_SPECT)

    metadata = ('SCI', 2), ('SCI', 3)
    slc = ad[1, 2]
    assert len(slc) == 2
    assert slc.is_sliced is True
    assert len(slc.data) == 2
    assert slc.data[0][0, 0] == 387
    assert slc.data[1][0, 0] == 383
    assert slc.shape == [(512, 288), (512, 288)]

    for ext, md in zip(slc, metadata):
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == md

    with pytest.raises(ValueError, match="Cannot return id"):
        slc.id

    assert slc[0].id == 2
    assert slc[1].id == 3

    match = "Can't remove items from a sliced object"
    with pytest.raises(TypeError, match=match):
        del slc[0]

    match = "Can't append objects to slices, use 'ext.NAME = obj' instead"
    with pytest.raises(TypeError, match=match):
        slc.append(np.zeros(5), name='ARR')

    match = "This attribute can only be assigned to a single-slice object"
    with pytest.raises(TypeError, match=match):
        slc.ARR = np.zeros(5)

    # iterate over single slice
    metadata = ('SCI', 1)
    for ext in ad[0]:
        assert (ext.hdr['EXTNAME'], ext.hdr['EXTVER']) == metadata

    # slice negative
    assert ad.data[-1] is ad[-1].data

    match = "This attribute can only be assigned to a single-slice object"
    with pytest.raises(TypeError, match=match):
        slc.FOO = 1

    with pytest.raises(TypeError,
                       match="Can't delete attributes on non-single slices"):
        del slc.FOO

    ext.bar = 1
    assert ext.bar == 1
    del ext.bar


@pytest.mark.dragons_remote_data
def test_slice_data(GMOSN_SPECT):
    ad = astrodata.open(GMOSN_SPECT)

    slc = ad[1, 2]
    match = "Trying to assign to an AstroData object that is not a single slice"
    with pytest.raises(ValueError, match=match):
        slc.data = 1
    with pytest.raises(ValueError, match=match):
        slc.uncertainty = 1
    with pytest.raises(ValueError, match=match):
        slc.mask = 1
    with pytest.raises(ValueError, match=match):
        slc.variance = 1

    assert slc.uncertainty == [None, None]
    assert slc.mask == [None, None]

    ext = ad[1]
    match = "Trying to assign data to be something with no shape"
    with pytest.raises(AttributeError, match=match):
        ext.data = 1

    # set/get on single slice
    ext.data = np.ones(10)
    assert_array_equal(ext.data, 1)

    ext.variance = np.ones(10)
    assert_array_equal(ext.variance, 1)
    ext.variance = None
    assert ext.variance is None

    ext.uncertainty = ADVarianceUncertainty(np.ones(10))
    assert_array_equal(ext.variance, 1)
    assert_array_equal(slc.variance[0], 1)

    ext.mask = np.zeros(10)
    assert_array_equal(ext.mask, 0)
    assert_array_equal(slc.mask[0], 0)

    assert slc.nddata[0].data is ext.data
    assert slc.nddata[0].uncertainty is ext.uncertainty
    assert slc.nddata[0].mask is ext.mask


@pytest.mark.dragons_remote_data
def test_phu(NIFS_DARK):
    ad = astrodata.open(NIFS_DARK)

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
def test_paths(tmpdir, NIFS_DARK):
    ad = astrodata.open(NIFS_DARK)
    assert ad.orig_filename == 'N20160727S0077.fits'

    srcdir = os.path.dirname(NIFS_DARK)
    assert ad.filename == 'N20160727S0077.fits'
    assert ad.path == os.path.join(srcdir, 'N20160727S0077.fits')

    ad.filename = 'newfile.fits'
    assert ad.filename == 'newfile.fits'
    assert ad.path == os.path.join(srcdir, 'newfile.fits')

    testfile = os.path.join(str(tmpdir), 'temp.fits')
    ad.path = testfile
    assert ad.filename == 'temp.fits'
    assert ad.path == testfile
    assert ad.orig_filename == 'N20160727S0077.fits'
    ad.write()
    assert os.path.exists(testfile)
    os.remove(testfile)

    testfile = os.path.join(str(tmpdir), 'temp2.fits')
    ad.write(testfile)
    assert os.path.exists(testfile)

    # overwriting is forbidden by default
    with pytest.raises(OSError):
        ad.write(testfile)

    ad.write(testfile, overwrite=True)
    assert os.path.exists(testfile)
    os.remove(testfile)

    ad.path = None
    assert ad.filename is None
    with pytest.raises(ValueError):
        ad.write()


@pytest.mark.dragons_remote_data
def test_from_hdulist(NIFS_DARK):
    with fits.open(NIFS_DARK) as hdul:
        assert 'ORIGNAME' not in hdul[0].header
        ad = astrodata.open(hdul)
        assert ad.path is None
        assert ad.instrument() == 'NIFS'
        assert ad.object() == 'Dark'
        assert ad.telescope() == 'Gemini-North'
        assert len(ad) == 1
        assert ad[0].shape == (2048, 2048)

    with fits.open(NIFS_DARK) as hdul:
        # Make sure that when ORIGNAME is set, astrodata use it
        hdul[0].header['ORIGNAME'] = 'N20160727S0077.fits'
        ad = astrodata.open(hdul)
        assert ad.path == 'N20160727S0077.fits'


def test_from_hdulist2():
    tablehdu = fits.table_to_hdu(Table([[1]]))
    tablehdu.name = 'REFCAT'

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({'INSTRUME': 'FISH'})),
        fits.ImageHDU(data=np.zeros(10), name='SCI', ver=1),
        fits.ImageHDU(data=np.ones(10), name='VAR', ver=1),
        fits.ImageHDU(data=np.zeros(10, dtype='uint16'), name='DQ', ver=1),
        tablehdu,
        fits.BinTableHDU.from_columns(
            [fits.Column(array=['a', 'b'], format='A', name='col')], ver=1,
        ),  # This HDU will be skipped because it has no EXTNAME
    ])

    with pytest.warns(UserWarning,
                      match='Skip HDU .* because it has no EXTNAME'):
        ad = astrodata.open(hdul)

    assert len(ad) == 1
    assert ad.phu['INSTRUME'] == 'FISH'
    assert_array_equal(ad[0].data, 0)
    assert_array_equal(ad[0].variance, 1)
    assert_array_equal(ad[0].mask, 0)
    assert len(ad.REFCAT) == 1
    assert ad.exposed == {'REFCAT'}
    assert ad[0].exposed == {'REFCAT'}


def test_from_hdulist3():
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.zeros(10), name='SCI', ver=1),
        fits.TableHDU.from_columns(
            [fits.Column(array=['a', 'b'], format='A', name='col')],
            name='ASCIITAB',
        ),
    ])

    ad = astrodata.open(hdul)

    assert hasattr(ad, 'ASCIITAB')
    assert len(ad.ASCIITAB) == 2


def test_can_make_and_write_ad_object(tmpdir):
    # Creates data and ad object
    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=np.arange(10))
    ad = astrodata.create(phu)
    ad.append(hdu, name='SCI')

    hdr = fits.Header({'EXTNAME': 'SCI', 'EXTVER': 1, 'FOO': 'BAR'})
    ad.append(hdu, header=hdr)
    assert ad[1].hdr['FOO'] == 'BAR'

    # Write file and test it exists properly
    testfile = str(tmpdir.join('created_fits_file.fits'))
    ad.write(testfile)

    # Opens file again and tests data is same as above
    adnew = astrodata.open(testfile)
    assert np.array_equal(adnew[0].data, np.arange(10))


def test_can_append_table_and_access_data(capsys, tmpdir):
    tbl = Table([np.zeros(10), np.ones(10)], names=['col1', 'col2'])
    phu = fits.PrimaryHDU()
    ad = astrodata.create(phu)

    with pytest.raises(ValueError,
                       match='Tables should be set directly as attribute'):
        ad.append(tbl, name='BOB')

    ad.BOB = tbl
    assert ad.exposed == {'BOB'}

    assert ad.tables == {'BOB'}
    assert np.all(ad.table()['BOB'] == tbl)

    ad.info()
    captured = capsys.readouterr()
    assert '.BOB           Table       (10, 2)' in captured.out

    # Write file and test it exists properly
    testfile = str(tmpdir.join('created_fits_file.fits'))
    ad.write(testfile)
    adnew = astrodata.open(testfile)
    assert adnew.exposed == {'BOB'}
    assert len(adnew.BOB) == 10

    del ad.BOB
    assert ad.tables == set()
    with pytest.raises(AttributeError):
        del ad.BOB


@pytest.mark.dragons_remote_data
def test_attributes(GSAOI_DARK):
    ad = astrodata.open(GSAOI_DARK)
    assert ad.shape == [(2048, 2048)] * 4
    assert [arr.shape for arr in ad.data] == [(2048, 2048)] * 4
    assert [arr.dtype for arr in ad.data] == ['f'] * 4
    assert ad.uncertainty == [None] * 4
    assert ad.variance == [None] * 4
    assert ad.mask == [None] * 4

    ad[0].variance = np.ones(ad[0].shape)
    assert isinstance(ad[0].uncertainty, ADVarianceUncertainty)
    assert_array_equal(ad[0].uncertainty.array, 1)
    assert_array_equal(ad[0].variance, 1)
    assert_array_equal(ad.variance[0], 1)

    assert all(isinstance(nd, NDAstroData) for nd in ad.nddata)
    assert [nd.shape for nd in ad.nddata] == [(2048, 2048)] * 4

    match = "Trying to assign to an AstroData object that is not a single slice"
    with pytest.raises(ValueError, match=match):
        ad.data = 1
    with pytest.raises(ValueError, match=match):
        ad.variance = 1
    with pytest.raises(ValueError, match=match):
        ad.uncertainty = 1
    with pytest.raises(ValueError, match=match):
        ad.mask = 1


@pytest.mark.dragons_remote_data
def test_set_a_keyword_on_phu_deprecated(NIFS_DARK):
    ad = astrodata.open(NIFS_DARK)
    # Test that setting DETECTOR as an attribute doesn't modify the header
    ad.phu.DETECTOR = 'FooBar'
    assert ad.phu.DETECTOR == 'FooBar'
    assert ad.phu['DETECTOR'] == 'NIFS'


# Regression:
# Make sure that references to associated
# extension objects are copied across
@pytest.mark.dragons_remote_data
def test_do_arith_and_retain_features(NIFS_DARK):
    ad = astrodata.open(NIFS_DARK)
    ad[0].NEW_FEATURE = np.array([1, 2, 3, 4, 5])
    ad2 = ad * 5
    assert_array_equal(ad[0].NEW_FEATURE, ad2[0].NEW_FEATURE)


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


def test_update_filename2():
    phu = fits.PrimaryHDU()
    ad = astrodata.create(phu)

    with pytest.raises(ValueError):
        # Not possible when ad.filename is None
        ad.update_filename(suffix='_suffix1')

    # filename is taken from ORIGNAME by default
    ad.phu['ORIGNAME'] = 'origfile.fits'
    ad.update_filename(suffix='_suffix')
    assert ad.filename == 'origfile_suffix.fits'

    ad.phu['ORIGNAME'] = 'temp.fits'
    ad.filename = 'origfile.fits'
    ad.update_filename(suffix='_bar', strip=True)
    assert ad.filename == 'origfile_bar.fits'


@pytest.mark.dragons_remote_data
def test_read_a_keyword_from_phu_deprecated():
    """Test deprecated methods to access headers"""
    ad = astrodata.open(download_from_archive('N20110826S0336.fits'))

    with pytest.raises(AttributeError):
        assert ad.phu.DETECTOR == 'GMOS + Red1'

    with pytest.raises(AttributeError):
        assert ad.hdr.CCDNAME == [
            'EEV 9273-16-03', 'EEV 9273-20-04', 'EEV 9273-20-03'
        ]

    # and when accessing missing extension
    with pytest.raises(AttributeError):
        ad.ABC


def test_read_invalid_file(tmpdir, caplog):
    testfile = str(tmpdir.join('test.fits'))
    with open(testfile, 'w'):
        # create empty file
        pass

    with pytest.raises(astrodata.AstroDataError):
        astrodata.open(testfile)

    assert caplog.records[0].message.endswith('is zero size')


def test_read_empty_file(tmpdir):
    testfile = str(tmpdir.join('test.fits'))
    hdr = fits.Header({'INSTRUME': 'darkimager', 'OBJECT': 'M42'})
    fits.PrimaryHDU(header=hdr).writeto(testfile)
    ad = astrodata.open(testfile)
    assert len(ad) == 0
    assert ad.object() == 'M42'
    assert ad.instrument() == 'darkimager'


def test_read_file(tmpdir):
    testfile = str(tmpdir.join('test.fits'))
    hdr = fits.Header({'INSTRUME': 'darkimager', 'OBJECT': 'M42'})
    fits.PrimaryHDU(header=hdr).writeto(testfile)
    ad = astrodata.open(testfile)
    assert len(ad) == 0
    assert ad.object() == 'M42'
    assert ad.instrument() == 'darkimager'


@pytest.mark.dragons_remote_data
def test_header_collection(GMOSN_SPECT):
    ad = astrodata.create({})
    assert ad.hdr is None

    ad = astrodata.open(GMOSN_SPECT)
    assert len(ad) == 12
    assert len([hdr for hdr in ad.hdr]) == 12

    # get
    assert 'FRAMEID' in ad.hdr
    assert 'FOO' not in ad.hdr
    assert ad.hdr.get('FRAMEID') == [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'
    ]
    with pytest.raises(KeyError):
        ad.hdr['FOO']
    assert ad.hdr.get('FOO') == [None] * 12
    assert ad.hdr.get('FOO', default='BAR') == ['BAR'] * 12

    # del/remove
    assert ad.hdr['GAIN'] == [1.0] * 12
    del ad.hdr['GAIN']
    with pytest.raises(KeyError):
        ad.hdr['GAIN']
    with pytest.raises(KeyError):
        del ad.hdr['GAIN']

    # set
    assert ad.hdr['RDNOISE'] == [1.0] * 12
    ad.hdr['RDNOISE'] = 2.0
    assert ad.hdr['RDNOISE'] == [2.0] * 12

    # comment
    assert ad.hdr.get_comment('DATATYPE') == ['Type of Data'] * 12
    ad.hdr.set_comment('DATATYPE', 'Hello!')
    assert ad.hdr.get_comment('DATATYPE') == ['Hello!'] * 12
    ad.hdr['RDNOISE'] = (2.0, 'New comment')
    assert ad.hdr.get_comment('RDNOISE') == ['New comment'] * 12
    with pytest.raises(KeyError,
                       match="Keyword 'FOO' not available at header 0"):
        ad.hdr.set_comment('FOO', 'A comment')

    ad = astrodata.open(GMOSN_SPECT)
    hdr = ad.hdr
    assert len(list(hdr)) == 12
    hdr._insert(1, fits.Header({'INSTRUME': 'darkimager', 'OBJECT': 'M42'}))
    assert len(list(hdr)) == 13


@pytest.mark.dragons_remote_data
def test_header_deprecated(GMOSN_SPECT):
    ad = astrodata.open(GMOSN_SPECT)
    with pytest.warns(AstroDataDeprecationWarning):
        warnings.simplefilter('always', AstroDataDeprecationWarning)
        header = ad.header
    assert header[0]['ORIGNAME'] == 'N20170529S0168.fits'
    assert header[1]['EXTNAME'] == 'SCI'
    assert header[1]['EXTVER'] == 1

    with pytest.warns(AstroDataDeprecationWarning):
        warnings.simplefilter('always', AstroDataDeprecationWarning)
        header = ad[0].header
    assert header[0]['ORIGNAME'] == 'N20170529S0168.fits'


@pytest.mark.dragons_remote_data
def test_read_no_extensions(GRACES_SPECT):
    ad = astrodata.open(GRACES_SPECT)
    assert len(ad) == 1
    # header is duplicated for .phu and extension's header
    assert len(ad.phu) == 181
    assert len(ad[0].hdr) == 185
    assert ad[0].hdr['EXTNAME'] == 'SCI'
    assert ad[0].hdr['EXTVER'] == 1


def test_add_var_and_dq():
    shape = (3, 4)
    fakedata = np.arange(np.prod(shape)).reshape(shape)

    ad = astrodata.create({'OBJECT': 'M42'})
    ad.append(fakedata)
    assert ad[0].hdr['EXTNAME'] == 'SCI'  # default value for EXTNAME

    with pytest.raises(ValueError, match="Only one Primary HDU allowed"):
        ad.append(fits.PrimaryHDU(data=fakedata), name='FOO')

    with pytest.raises(ValueError,
                       match="Arbitrary image extensions can "
                       "only be added in association to a 'SCI'"):
        ad.append(np.zeros(shape), name='FOO')

    with pytest.raises(ValueError,
                       match="'VAR' need to be associated to a 'SCI' one"):
        ad.append(np.ones(shape), name='VAR')

    with pytest.raises(AttributeError,
                       match="SCI extensions should be appended with .append"):
        ad[0].SCI = np.ones(shape)


def test_add_table():
    shape = (3, 4)
    fakedata = np.arange(np.prod(shape)).reshape(shape)

    ad = astrodata.create({'OBJECT': 'M42'})
    ad.append(fakedata)

    ad.TABLE1 = Table([['a', 'b', 'c'], [1, 2, 3]])
    assert ad.tables == {'TABLE1'}

    ad.TABLE2 = Table([['a', 'b', 'c'], [1, 2, 3]])
    assert ad.tables == {'TABLE1', 'TABLE2'}

    ad.MYTABLE = Table([['a', 'b', 'c'], [1, 2, 3]])
    assert ad.tables == {'TABLE1', 'TABLE2', 'MYTABLE'}

    ad[0].TABLE3 = Table([['aa', 'bb', 'cc'], [1, 2, 3]])
    ad[0].TABLE4 = Table([['aa', 'bb', 'cc'], [1, 2, 3]])
    ad[0].OTHERTABLE = Table([['aa', 'bb', 'cc'], [1, 2, 3]])

    assert list(ad[0].OTHERTABLE['col0']) == ['aa', 'bb', 'cc']

    assert ad.tables == {'TABLE1', 'TABLE2', 'MYTABLE'}
    assert ad[0].tables == {'TABLE1', 'TABLE2', 'MYTABLE'}
    assert ad[0].ext_tables == {'OTHERTABLE', 'TABLE3', 'TABLE4'}
    assert ad[0].exposed == {'MYTABLE', 'OTHERTABLE', 'TABLE1', 'TABLE2',
                             'TABLE3', 'TABLE4'}

    with pytest.raises(AttributeError):
        ad.ext_tables

    assert set(ad[0].nddata.meta['other'].keys()) == {'OTHERTABLE',
                                                      'TABLE3', 'TABLE4'}
    assert_array_equal(ad[0].TABLE3['col0'], ['aa', 'bb', 'cc'])
    assert_array_equal(ad[0].TABLE4['col0'], ['aa', 'bb', 'cc'])
    assert_array_equal(ad[0].OTHERTABLE['col0'], ['aa', 'bb', 'cc'])


@pytest.mark.dragons_remote_data
def test_copy(GSAOI_DARK, capsys):
    ad = astrodata.open(GSAOI_DARK)
    ad.TABLE = Table([['a', 'b', 'c'], [1, 2, 3]])
    ad[0].MYTABLE = Table([['aa', 'bb', 'cc'], [1, 2, 3]])

    ad.info()
    captured = capsys.readouterr()

    ad2 = copy.deepcopy(ad)
    ad2.info()
    captured2 = capsys.readouterr()

    # Compare that objects have the same attributes etc. with their
    # .info representation
    assert captured.out == captured2.out

    ext = ad[0]
    ext.info()
    captured = capsys.readouterr()

    ext2 = copy.deepcopy(ext)
    ext2.info()
    captured2 = capsys.readouterr()

    # Same for extension, except that first line is different (no
    # filename in the copied ext)
    assert captured.out.splitlines()[1:] == captured2.out.splitlines()[1:]


@pytest.mark.dragons_remote_data
def test_crop(GSAOI_DARK):
    ad = astrodata.open(GSAOI_DARK)
    assert set(ad.shape) == {(2048, 2048)}

    ad.crop(0, 0, 5, 10)
    assert len(ad.nddata) == 4
    assert set(ad.shape) == {(11, 6)}


@pytest.mark.dragons_remote_data
def test_crop_ext(GSAOI_DARK):
    ad = astrodata.open(GSAOI_DARK)
    ext = ad[0]
    ext.uncertainty = ADVarianceUncertainty(np.ones(ext.shape))
    ext.mask = np.ones(ext.shape, dtype=np.uint8)

    # FIXME: cannot test cropping attached array because that does not work
    # with numpy arrays, and can only attach NDData instance at the top level
    # ext.append(np.zeros(ext.shape, dtype=int), name='FOO')

    ext.BAR = 1

    ext.crop(0, 0, 5, 10)
    assert ext.shape == (11, 6)
    assert_allclose(ext.data[0], [-1.75, -0.75, -4.75, 2.375, -0.25, 1.375])
    assert_array_equal(ext.uncertainty.array, 1)
    assert_array_equal(ext.mask, 1)

    # assert ext.FOO.shape == (11, 6)
    # assert_array_equal(ext.FOO, 0)
    assert ext.BAR == 1


@pytest.mark.xfail(not astropy.utils.minversion(astropy, '4.0.1'),
                   reason='requires astropy >=4.0.1 for correct serialization')
def test_round_trip_gwcs(tmpdir):
    """
    Add a 2-step gWCS instance to NDAstroData, save to disk, reload & compare.
    """

    from gwcs import coordinate_frames as cf
    from gwcs import WCS

    arr = np.zeros((10, 10), dtype=np.float32)
    ad1 = astrodata.create(fits.PrimaryHDU(), [fits.ImageHDU(arr, name='SCI')])

    # Transformation from detector pixels to pixels in some reference row,
    # removing relative distortions in wavelength:
    det_frame = cf.Frame2D(name='det_mosaic', axes_names=('x', 'y'),
                           unit=(u.pix, u.pix))
    dref_frame = cf.Frame2D(name='dist_ref_row', axes_names=('xref', 'y'),
                            unit=(u.pix, u.pix))

    # A made-up example model that looks vaguely like some real distortions:
    fdist = models.Chebyshev2D(2, 2,
                               c0_0=4.81125, c1_0=5.43375, c0_1=-0.135,
                               c1_1=-0.405, c0_2=0.30375, c1_2=0.91125,
                               x_domain=[0., 9.], y_domain=[0., 9.])

    # This is not an accurate inverse, but will do for this test:
    idist = models.Chebyshev2D(2, 2,
                               c0_0=4.89062675, c1_0=5.68581232,
                               c2_0=-0.00590263, c0_1=0.11755526,
                               c1_1=0.35652358, c2_1=-0.01193828,
                               c0_2=-0.29996306, c1_2=-0.91823397,
                               c2_2=0.02390594,
                               x_domain=[-1.5, 12.], y_domain=[0., 9.])

    # The resulting 2D co-ordinate mapping from detector to ref row pixels:
    distrans = models.Mapping((0, 1, 1)) | (fdist & models.Identity(1))
    distrans.inverse = models.Mapping((0, 1, 1)) | (idist & models.Identity(1))

    # Transformation from reference row pixels to linear, row-stacked spectra:
    spec_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm,
                                  axes_names='lambda', name='wavelength')
    row_frame = cf.CoordinateFrame(1, 'SPATIAL', axes_order=(1,), unit=u.pix,
                                   axes_names='y', name='row')
    rss_frame = cf.CompositeFrame([spec_frame, row_frame])

    # Toy wavelength model & approximate inverse:
    fwcal = models.Chebyshev1D(2, c0=500.075, c1=0.05, c2=0.001, domain=[0, 9])
    iwcal = models.Chebyshev1D(2, c0=4.59006292, c1=4.49601817, c2=-0.08989608,
                               domain=[500.026, 500.126])

    # The resulting 2D co-ordinate mapping from ref pixels to wavelength:
    wavtrans = fwcal & models.Identity(1)
    wavtrans.inverse = iwcal & models.Identity(1)

    # The complete WCS chain for these 2 transformation steps:
    ad1[0].nddata.wcs = WCS([(det_frame, distrans),
                             (dref_frame, wavtrans),
                             (rss_frame, None)
                             ])

    # Save & re-load the AstroData instance with its new WCS attribute:
    testfile = str(tmpdir.join('round_trip_gwcs.fits'))
    ad1.write(testfile)
    ad2 = astrodata.open(testfile)

    wcs1 = ad1[0].nddata.wcs
    wcs2 = ad2[0].nddata.wcs

    # # Temporary workaround for issue #9809, to ensure the test is correct:
    # wcs2.forward_transform[1].x_domain = (0, 9)
    # wcs2.forward_transform[1].y_domain = (0, 9)
    # wcs2.forward_transform[3].domain = (0, 9)
    # wcs2.backward_transform[0].domain = (500.026, 500.126)
    # wcs2.backward_transform[3].x_domain = (-1.5, 12.)
    # wcs2.backward_transform[3].y_domain = (0, 9)

    # Did we actually get a gWCS instance back?
    assert isinstance(wcs2, WCS)

    # Do the transforms have the same number of submodels, with the same types,
    # degrees, domains & parameters? Here the inverse gets checked redundantly
    # as both backward_transform and forward_transform.inverse, but it would be
    # convoluted to ensure that both are correct otherwise (since the transforms
    # get regenerated as new compound models each time they are accessed).
    compare_models(wcs1.forward_transform, wcs2.forward_transform)
    compare_models(wcs1.backward_transform, wcs2.backward_transform)

    # Do the instances have matching co-ordinate frames?
    for f in wcs1.available_frames:
        assert repr(getattr(wcs1, f)) == repr(getattr(wcs2, f))

    # Also compare a few transformed values, as the "proof of the pudding":
    y, x = np.mgrid[0:9:2, 0:9:2]
    np.testing.assert_allclose(wcs1(x, y), wcs2(x, y), rtol=1e-7, atol=0.)

    y, w = np.mgrid[0:9:2, 500.025:500.12:0.0225]
    np.testing.assert_allclose(wcs1.invert(w, y), wcs2.invert(w, y),
                               rtol=1e-7, atol=0.)


@pytest.mark.parametrize('dtype', ['int8', 'uint8', 'int16', 'uint16',
                                   'int32', 'uint32', 'int64', 'uint64'])
def test_uint_data(dtype, tmp_path):
    testfile = tmp_path / 'test.fits'
    data = np.arange(10, dtype=np.int16)
    fits.writeto(testfile, data)

    ad = astrodata.open(str(testfile))
    assert ad[0].data.dtype == data.dtype
    assert_array_equal(ad[0].data, data)


if __name__ == '__main__':
    pytest.main()
