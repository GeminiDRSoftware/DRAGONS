import os
from subprocess import call

from nose.tools import raises, assert_true, ok_

from file_urls import sci123
from astrodata import AstroData
from astrodata import Errors

@raises(Errors.AstroDataError)
def test1():
    '''ASTRODATA-accessmode TEST 1: Fail when try to overwrite readonly'''
    ad = AstroData(sci123)
    ad.write()

def test2():
    '''ASTRODATA-accessmode TEST 2: Overwrite readonly AD using name change'''
    ad = AstroData(sci123)
    outfile = 'python_out/accessmodetest2.fits'
    if os.path.exists(outfile):
        os.remove(outfile)
    ad.filename = outfile
    ad.write()
    assert_true(os.path.isfile(outfile), 'ad.write() FAIL')
    os.remove(outfile)
    ad.close()

def test3():
    '''ASTRODATA-accessmode TEST 3: Clobber unchanged readonly file'''
    ad = AstroData(sci123)
    ad.write(clobber=True)
    assert_true(os.path.isfile(sci123), 'Clobber fail')
    ad.close()

def test4():
    '''ASTRODATA-accessmode TEST 4: Fail when try to update readonly file
    '''
    outfile = 'python_out/accessmodetest3.fits'
    call('cp ' + sci123 + ' ' + outfile, shell=True)
    ad = AstroData(outfile, mode='update')
    ad.hdulist[0].header['INSTRUME'] = 'GMOS-S'
    try:
        ad.write()
    except Errors.AstroDataError,e:
        throwexcept = True
    assert_true(throwexcept, "Updated readonly file")
    ad.close()
    os.remove(outfile)

def test5():
    '''ASTRODATA-accessmode TEST 5: Clobber changed readonly file'''
    outfile = 'python_out/accessmodetest5.fits'
    call('cp ' + sci123 + ' ' + outfile, shell=True)
    ad = AstroData(outfile, mode='update')
    ad.hdulist[0].header['INSTRUME'] = 'GMOS-S'
    ad.write(clobber=True)
    ad.close()
    ad = AstroData(outfile)
    ok_(ad.hdulist[0].header['INSTRUME'] == 'GMOS-S', msg='Keyword is different')
    os.remove(outfile)
    ad.close()

