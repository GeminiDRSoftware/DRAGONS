import numpy as np
import pytest
from astropy.io.fits import ImageHDU
from astropy.nddata import NDData
from astropy.table import Table

import astrodata
from astrodata.testing import download_from_archive


@pytest.fixture
def testfile1():
    """
    Pixels Extensions
    Index  Content                  Type              Dimensions     Format
    [ 0]   science                  NDAstroData       (2304, 1056)   uint16
    [ 1]   science                  NDAstroData       (2304, 1056)   uint16
    [ 2]   science                  NDAstroData       (2304, 1056)   uint16
    """
    return download_from_archive("N20110826S0336.fits", path="GMOS")


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
    return download_from_archive("N20160524S0119.fits", path="GMOS")


@pytest.mark.remote_data
def test_can_read_data(testfile1):
    ad = astrodata.open(testfile1)
    assert len(ad) == 3


@pytest.mark.remote_data
def test_append_array_to_root_no_name(testfile2):
    ad = astrodata.open(testfile2)

    lbefore = len(ad)
    ones = np.ones((10, 10))
    ad.append(ones)
    assert len(ad) == (lbefore + 1)
    assert ad[-1].data is ones
    assert ad[-1].hdr['EXTNAME'] == 'SCI'
    assert ad[-1].hdr['EXTVER'] == len(ad)


@pytest.mark.remote_data
def test_append_array_to_root_with_name_sci(testfile2):
    ad = astrodata.open(testfile2)

    lbefore = len(ad)
    ones = np.ones((10, 10))
    ad.append(ones, name='SCI')
    assert len(ad) == (lbefore + 1)
    assert ad[-1].data is ones
    assert ad[-1].hdr['EXTNAME'] == 'SCI'
    assert ad[-1].hdr['EXTVER'] == len(ad)


@pytest.mark.remote_data
def test_append_array_to_root_with_arbitrary_name(testfile2):
    ad = astrodata.open(testfile2)
    assert len(ad) == 6

    ones = np.ones((10, 10))
    with pytest.raises(ValueError):
        ad.append(ones, name='ARBITRARY')


@pytest.mark.remote_data
def test_append_array_to_extension_with_name_sci(testfile2):
    ad = astrodata.open(testfile2)
    assert len(ad) == 6

    ones = np.ones((10, 10))
    with pytest.raises(ValueError):
        ad[0].append(ones, name='SCI')


@pytest.mark.remote_data
def test_append_array_to_extension_with_arbitrary_name(testfile2):
    ad = astrodata.open(testfile2)

    lbefore = len(ad)
    ones = np.ones((10, 10))
    ad[0].append(ones, name='ARBITRARY')

    assert len(ad) == lbefore
    assert ad[0].ARBITRARY is ones


@pytest.mark.remote_data
def test_append_nddata_to_root_no_name(testfile2):
    ad = astrodata.open(testfile2)

    lbefore = len(ad)
    ones = np.ones((10, 10))
    hdu = ImageHDU(ones)
    nd = NDData(hdu.data)
    nd.meta['header'] = hdu.header
    ad.append(nd)
    assert len(ad) == (lbefore + 1)


@pytest.mark.remote_data
def test_append_nddata_to_root_with_arbitrary_name(testfile2):
    ad = astrodata.open(testfile2)
    assert len(ad) == 6

    ones = np.ones((10, 10))
    hdu = ImageHDU(ones)
    nd = NDData(hdu.data)
    nd.meta['header'] = hdu.header
    hdu.header['EXTNAME'] = 'ARBITRARY'
    with pytest.raises(ValueError):
        ad.append(nd)


@pytest.mark.remote_data
def test_append_table_to_root(testfile2):
    ad = astrodata.open(testfile2)
    with pytest.raises(AttributeError):
        ad.MYTABLE

    assert len(ad) == 6
    table = Table(([1, 2, 3], [4, 5, 6], [7, 8, 9]), names=('a', 'b', 'c'))
    ad.append(table, 'MYTABLE')
    assert (ad.MYTABLE == table).all()


@pytest.mark.remote_data
def test_append_table_to_root_without_name(testfile2):
    ad = astrodata.open(testfile2)
    assert len(ad) == 6
    with pytest.raises(AttributeError):
        ad.TABLE1

    table = Table(([1, 2, 3], [4, 5, 6], [7, 8, 9]), names=('a', 'b', 'c'))
    ad.append(table)
    assert len(ad) == 6
    assert isinstance(ad.TABLE1, Table)


@pytest.mark.remote_data
def test_append_table_to_extension(testfile2):
    ad = astrodata.open(testfile2)
    assert len(ad) == 6

    table = Table(([1, 2, 3], [4, 5, 6], [7, 8, 9]), names=('a', 'b', 'c'))
    ad[0].append(table, 'MYTABLE')
    assert (ad[0].MYTABLE == table).all()


# Append / assign Gemini specific

@pytest.mark.remote_data
def test_append_dq_to_root(testfile2):
    ad = astrodata.open(testfile2)

    dq = np.zeros(ad[0].data.shape)
    with pytest.raises(ValueError):
        ad.append(dq, 'DQ')


@pytest.mark.remote_data
def test_append_dq_to_ext(testfile2):
    ad = astrodata.open(testfile2)

    dq = np.zeros(ad[0].data.shape)
    ad[0].append(dq, 'DQ')
    assert dq is ad[0].mask


@pytest.mark.remote_data
def test_append_var_to_root(testfile2):
    ad = astrodata.open(testfile2)

    var = np.random.random(ad[0].data.shape)
    with pytest.raises(ValueError):
        ad.append(var, 'VAR')


@pytest.mark.remote_data
def test_append_var_to_ext(testfile2):
    ad = astrodata.open(testfile2)

    var = np.random.random(ad[0].data.shape)
    ad[0].append(var, 'VAR')
    assert np.abs(var - ad[0].variance).mean() < 0.00000001


# Append AstroData slices

@pytest.mark.remote_data
def test_append_single_slice(testfile1, testfile2):
    ad = astrodata.open(testfile2)
    ad2 = astrodata.open(testfile1)

    lbefore = len(ad2)
    last_ever = ad2[-1].nddata.meta['header'].get('EXTVER', -1)
    ad2.append(ad[1])

    assert len(ad2) == (lbefore + 1)
    assert np.all(ad2[-1].data == ad[1].data)
    assert last_ever < ad2[-1].nddata.meta['header'].get('EXTVER', -1)


@pytest.mark.remote_data
def test_append_non_single_slice(testfile1, testfile2):
    ad = astrodata.open(testfile2)
    ad2 = astrodata.open(testfile1)

    with pytest.raises(ValueError):
        ad2.append(ad[1:])


@pytest.mark.remote_data
def test_append_whole_instance(testfile1, testfile2):
    ad = astrodata.open(testfile2)
    ad2 = astrodata.open(testfile1)

    with pytest.raises(ValueError):
        ad2.append(ad)


@pytest.mark.remote_data
def test_append_slice_to_extension(testfile1, testfile2):
    ad = astrodata.open(testfile2)
    ad2 = astrodata.open(testfile1)

    with pytest.raises(ValueError):
        ad2[0].append(ad[0], name="FOOBAR")


@pytest.mark.remote_data
def test_delete_named_associated_extension(testfile2):
    ad = astrodata.open(testfile2)
    table = Table(([1, 2, 3], [4, 5, 6], [7, 8, 9]), names=('a', 'b', 'c'))
    ad[0].append(table, 'MYTABLE')
    assert 'MYTABLE' in ad[0]
    del ad[0].MYTABLE
    assert 'MYTABLE' not in ad[0]


@pytest.mark.remote_data
def test_delete_arbitrary_attribute_from_ad(testfile2):
    ad = astrodata.open(testfile2)

    with pytest.raises(AttributeError):
        ad.arbitrary

    ad.arbitrary = 15

    assert ad.arbitrary == 15

    del ad.arbitrary

    with pytest.raises(AttributeError):
        ad.arbitrary
