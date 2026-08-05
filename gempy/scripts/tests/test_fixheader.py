import pytest

import os

import astrodata
from gempy.scripts import fixheader


@pytest.fixture(scope='function')
def testfile(astrofaker, change_working_dir):
    with change_working_dir():
        ad = astrofaker.create('GMOS-N', filename="N20010101S0001.fits")
        ad.init_default_extensions()
        ad.write(overwrite=True)
        cwd = os.getcwd()
    return os.path.join(cwd, ad.filename)


def test_edit_phu(testfile):
    fixheader.main([testfile, 'EXPTIME', '30'])
    ad = astrodata.open(testfile)
    assert ad.phu['EXPTIME'] == pytest.approx(30)


def test_add_phu(testfile):
    fixheader.main([testfile, 'NEW', 'value', '-a'])
    ad = astrodata.open(testfile)
    assert ad.phu['NEW'] == 'value'


def test_edit_phu_new_keyword(testfile):
    with pytest.raises(KeyError):
        fixheader.main([testfile, 'NEW', 'value'])


def test_add_phu_dtype(testfile):
    fixheader.main([testfile, 'NEW', '30', '-d', 'float', '-a'])
    ad = astrodata.open(testfile)
    assert ad.phu['NEW'] == pytest.approx(30)


def test_edit_all_hdr(testfile):
    fixheader.main([testfile, 'GAIN', '2'])
    ad = astrodata.open(testfile)
    assert 'GAIN' not in ad.phu
    assert ad.hdr['GAIN'] == [2] * len(ad)


def test_add_all_hdr(testfile):
    fixheader.main([testfile + ":", 'NEW', '2', '-d', 'float', '-a'])
    ad = astrodata.open(testfile)
    assert 'NEW' not in ad.phu
    assert ad.hdr['NEW'] == [2] * len(ad)



