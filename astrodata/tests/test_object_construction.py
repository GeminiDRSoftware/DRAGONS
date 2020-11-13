import astrodata
import numpy as np
import pytest
from astrodata.testing import download_from_archive
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from numpy.testing import assert_array_equal


@pytest.fixture()
def testfile1():
    """
    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDAstroData       (2304, 1056)   uint16
    [ 1]   science                  NDAstroData       (2304, 1056)   uint16
    [ 2]   science                  NDAstroData       (2304, 1056)   uint16
    """
    return download_from_archive("N20110826S0336.fits")


@pytest.fixture
def testfile2():
    """
    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDAstroData       (4608, 1056)   uint16
    [ 1]   science                  NDAstroData       (4608, 1056)   uint16
    [ 2]   science                  NDAstroData       (4608, 1056)   uint16
    [ 3]   science                  NDAstroData       (4608, 1056)   uint16
    [ 4]   science                  NDAstroData       (4608, 1056)   uint16
    [ 5]   science                  NDAstroData       (4608, 1056)   uint16
    """
    return download_from_archive("N20160524S0119.fits")


def test_create_with_no_data():
    for phu in (fits.PrimaryHDU(), fits.Header(), {}):
        ad = astrodata.create(phu)
        assert isinstance(ad, astrodata.AstroData)
        assert len(ad) == 0
        assert ad.instrument() is None
        assert ad.object() is None


def test_create_with_header():
    hdr = fits.Header({'INSTRUME': 'darkimager', 'OBJECT': 'M42'})
    for phu in (hdr, fits.PrimaryHDU(header=hdr), dict(hdr), list(hdr.cards)):
        ad = astrodata.create(phu)
        assert isinstance(ad, astrodata.AstroData)
        assert len(ad) == 0
        assert ad.instrument() == 'darkimager'
        assert ad.object() == 'M42'


def test_create_from_hdu():
    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=np.zeros((4, 5)), name='SCI')
    ad = astrodata.create(phu, [hdu])

    assert isinstance(ad, astrodata.AstroData)
    assert len(ad) == 1
    assert isinstance(ad[0].data, np.ndarray)
    assert ad[0].data is hdu.data


def test_create_invalid():
    with pytest.raises(ValueError):
        astrodata.create('FOOBAR')
    with pytest.raises(ValueError):
        astrodata.create(42)


def test_append_image_hdu():
    ad = astrodata.create(fits.PrimaryHDU())
    ad.append(fits.ImageHDU(data=np.zeros((4, 5))))
    ad.append(fits.ImageHDU(data=np.zeros((4, 5))), name='SCI')

    with pytest.raises(ValueError,
                       match="Arbitrary image extensions can only be added "
                       "in association to a 'SCI'"):
        ad.append(fits.ImageHDU(data=np.zeros((4, 5))), name='SCI2')

    assert len(ad) == 2


def test_append_lowercase_name():
    ad = astrodata.create({})
    with pytest.warns(UserWarning,
                      match="extension name 'sci' should be uppercase"):
        ad.append(NDData(np.zeros((4, 5))), name='sci')


def test_append_arrays(tmp_path):
    testfile = tmp_path / 'test.fits'

    ad = astrodata.create({})
    ad.append(np.zeros(10))
    ad[0].ARR = np.arange(5)

    with pytest.raises(AttributeError):
        ad[0].SCI = np.arange(5)
    with pytest.raises(AttributeError):
        ad[0].VAR = np.arange(5)
    with pytest.raises(AttributeError):
        ad[0].DQ = np.arange(5)

    match = ("Arbitrary image extensions can only be added in association "
             "to a 'SCI'")
    with pytest.raises(ValueError, match=match):
        ad.append(np.zeros(10), name='FOO')

    with pytest.raises(ValueError, match=match):
        ad.append(np.zeros(10), header=fits.Header({'EXTNAME': 'FOO'}))

    ad.write(testfile)

    ad = astrodata.open(testfile)
    assert len(ad) == 1
    assert ad[0].nddata.meta['header']['EXTNAME'] == 'SCI'
    assert_array_equal(ad[0].ARR, np.arange(5))


@pytest.mark.dragons_remote_data
def test_can_read_data(testfile1):
    ad = astrodata.open(testfile1)
    assert len(ad) == 3
    assert ad.shape == [(2304, 1056), (2304, 1056), (2304, 1056)]


@pytest.mark.dragons_remote_data
def test_can_read_write_pathlib(tmp_path):
    testfile = tmp_path / 'test.fits'

    ad = astrodata.create({})
    ad.append(np.zeros((4, 5)))
    ad.write(testfile)

    ad = astrodata.open(testfile)
    assert len(ad) == 1
    assert ad.shape == [(4, 5)]


@pytest.mark.dragons_remote_data
def test_append_array_to_root_no_name(testfile2):
    ad = astrodata.open(testfile2)

    lbefore = len(ad)
    ones = np.ones((10, 10))
    ad.append(ones)
    assert len(ad) == (lbefore + 1)
    assert ad[-1].data is ones
    assert ad[-1].hdr['EXTNAME'] == 'SCI'
    assert ad[-1].hdr['EXTVER'] == len(ad)


@pytest.mark.dragons_remote_data
def test_append_array_to_root_with_name_sci(testfile2):
    ad = astrodata.open(testfile2)

    lbefore = len(ad)
    ones = np.ones((10, 10))
    ad.append(ones, name='SCI')
    assert len(ad) == (lbefore + 1)
    assert ad[-1].data is ones
    assert ad[-1].hdr['EXTNAME'] == 'SCI'
    assert ad[-1].hdr['EXTVER'] == len(ad)


@pytest.mark.dragons_remote_data
def test_append_array_to_root_with_arbitrary_name(testfile2):
    ad = astrodata.open(testfile2)
    assert len(ad) == 6

    ones = np.ones((10, 10))
    with pytest.raises(ValueError):
        ad.append(ones, name='ARBITRARY')


@pytest.mark.dragons_remote_data
def test_append_array_to_extension_with_name_sci(testfile2):
    ad = astrodata.open(testfile2)
    assert len(ad) == 6

    ones = np.ones((10, 10))
    with pytest.raises(TypeError):
        ad[0].append(ones, name='SCI')


@pytest.mark.dragons_remote_data
def test_append_array_to_extension_with_arbitrary_name(testfile2):
    ad = astrodata.open(testfile2)

    lbefore = len(ad)
    ones = np.ones((10, 10))
    ad[0].ARBITRARY = ones

    assert len(ad) == lbefore
    assert ad[0].ARBITRARY is ones


@pytest.mark.dragons_remote_data
def test_append_nddata_to_root_no_name(testfile2):
    ad = astrodata.open(testfile2)

    lbefore = len(ad)
    ones = np.ones((10, 10))
    hdu = fits.ImageHDU(ones)
    nd = NDData(hdu.data)
    nd.meta['header'] = hdu.header
    ad.append(nd)
    assert len(ad) == (lbefore + 1)


@pytest.mark.dragons_remote_data
def test_append_nddata_to_root_with_arbitrary_name(testfile2):
    ad = astrodata.open(testfile2)
    assert len(ad) == 6

    ones = np.ones((10, 10))
    hdu = fits.ImageHDU(ones)
    nd = NDData(hdu.data)
    nd.meta['header'] = hdu.header
    hdu.header['EXTNAME'] = 'ARBITRARY'
    with pytest.raises(ValueError):
        ad.append(nd)


def test_append_table_to_extensions():
    ad = astrodata.create({})
    ad.append(NDData(np.zeros((4, 5))))
    ad.append(NDData(np.zeros((4, 5))))
    ad.append(NDData(np.zeros((4, 5))))
    ad[0].TABLE1 = Table([[1]])
    ad[0].TABLE2 = Table([[22]])
    ad[1].TABLE2 = Table([[2]])  # extensions can have the same table name
    ad[2].TABLE3 = Table([[3]])

    # Check that slices do not report extension tables
    assert ad.exposed == set()
    assert ad[0].exposed == {'TABLE1', 'TABLE2'}
    assert ad[1].exposed == {'TABLE2'}
    assert ad[2].exposed == {'TABLE3'}
    assert ad[1:].exposed == set()

    match = ("Cannot append table 'TABLE1' because it would hide an "
             "extension table")
    with pytest.raises(ValueError, match=match):
        ad.TABLE1 = Table([[1]])


# Append / assign Gemini specific

@pytest.mark.dragons_remote_data
def test_append_dq_var(testfile2):
    ad = astrodata.open(testfile2)

    dq = np.zeros(ad[0].data.shape)
    with pytest.raises(ValueError):
        ad.append(dq, name='DQ')
    with pytest.raises(AttributeError):
        ad.DQ = dq
    with pytest.raises(AttributeError):
        ad[0].DQ = dq

    var = np.ones(ad[0].data.shape)
    with pytest.raises(ValueError):
        ad.append(var, name='VAR')
    with pytest.raises(AttributeError):
        ad.VAR = var
    with pytest.raises(AttributeError):
        ad[0].VAR = var


# Append AstroData slices

@pytest.mark.dragons_remote_data
def test_append_single_slice(testfile1, testfile2):
    ad = astrodata.open(testfile2)
    ad2 = astrodata.open(testfile1)

    lbefore = len(ad2)
    last_ever = ad2[-1].nddata.meta['header'].get('EXTVER', -1)
    ad2.append(ad[1])

    assert len(ad2) == (lbefore + 1)
    assert np.all(ad2[-1].data == ad[1].data)
    assert last_ever < ad2[-1].nddata.meta['header'].get('EXTVER', -1)

    # With a custom header
    ad2.append(ad[1], header=fits.Header({'FOO': 'BAR'}))
    assert ad2[-1].nddata.meta['header']['FOO'] == 'BAR'


@pytest.mark.dragons_remote_data
def test_append_non_single_slice(testfile1, testfile2):
    ad = astrodata.open(testfile2)
    ad2 = astrodata.open(testfile1)

    with pytest.raises(ValueError):
        ad2.append(ad[1:])


@pytest.mark.dragons_remote_data
def test_append_whole_instance(testfile1, testfile2):
    ad = astrodata.open(testfile2)
    ad2 = astrodata.open(testfile1)

    with pytest.raises(ValueError):
        ad2.append(ad)


@pytest.mark.dragons_remote_data
def test_append_slice_to_extension(testfile1, testfile2):
    ad = astrodata.open(testfile2)
    ad2 = astrodata.open(testfile1)

    with pytest.raises(TypeError):
        ad2[0].append(ad[0], name="FOOBAR")

    match = "Cannot append an AstroData slice to another slice"
    with pytest.raises(ValueError, match=match):
        ad[2].FOO = ad2[1]


@pytest.mark.dragons_remote_data
def test_delete_named_associated_extension(testfile2):
    ad = astrodata.open(testfile2)
    ad[0].MYTABLE = Table(([1, 2, 3], [4, 5, 6], [7, 8, 9]),
                          names=('a', 'b', 'c'))
    assert 'MYTABLE' in ad[0]
    del ad[0].MYTABLE
    assert 'MYTABLE' not in ad[0]


@pytest.mark.dragons_remote_data
def test_delete_arbitrary_attribute_from_ad(testfile2):
    ad = astrodata.open(testfile2)

    with pytest.raises(AttributeError):
        ad.arbitrary

    ad.arbitrary = 15

    assert ad.arbitrary == 15

    del ad.arbitrary

    with pytest.raises(AttributeError):
        ad.arbitrary
