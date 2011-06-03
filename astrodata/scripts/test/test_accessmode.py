import os
from subprocess import call

from nose.tools import *

import file_urls 
from astrodata import AstroData
from astrodata import Errors

testfile = file_urls.testdatafile_1

def accessmode_test1():
    '''accessmode_test1 -Exception, ad write readonly using same filename  
    '''
    print('\n\t* testfile: %s' % testfile)
    ad = AstroData(testfile)
    print('\tad = AstroData(testfile)')
    print('\tad.write()   #expect exception here')
    throwexcept = False
    try:
        ad.write()
    except Errors.AstroDataError,e:
        throwexcept = True
        print("\t%s" % e)
    assert_true(throwexcept)
    ad.close()
    print('\t#ad closed')

def accessmode_test2():
    '''accessmode_test2 -ad write readonly using different filename
    '''
    print('\n\t* testfile: %s' % testfile)
    ad = AstroData(testfile)
    print('\tad = AstroData(testfile)   #default is readonly')
    ad.filename = 'test2.fits'
    print('\tad.filename = "test2.fits"   #filename changed')
    ad.write()
    print('\tad.write()')
    assert_true(os.path.isfile('./test2.fits'))
    print("\tassert_true(os.path.isfile('./test2.fits'))")
    os.remove('./test2.fits')
    ad.close()
    print('\t#ad closed and test2.fits removed')

def accessmode_test3():
    '''accessmode_test3 -Exception, ad write readonly to same file (clobber)
    '''
    print('\n\t* testfile: %s' % testfile)
    ad = AstroData(testfile)
    print('\tad = AstroData(testfile)')
    print('\tad.write(clobber=True)   #readonly should override clobber')
    throwexcept = False
    try:
        ad.write(clobber=True)
    except Errors.AstroDataError,e:
        throwexcept = True
        print("\t%s" % e)
    assert_true(throwexcept)
    ad.close()
    print('\t#ad closed')

def accessmode_test4():
    '''accessmode_test4 -Exception, ad write to same file (updated, no clobber)
    '''
    print('\n\t* testfile: %s' % testfile)
    call('cp ' + testfile + ' test4.fits', shell=True)
    print('\t#moved testfile into current directory as "test4.fits"')
    ad = AstroData('test4.fits', mode='update')
    print('\tad = AstroData("test4.fits", mode="update")')
    ad.hdulist[0].header['INSTRUME'] = 'GMOS-S'
    print('\t#change keyword INSTRUME from GMOS-N to GMOS-S')
    print("\tad.hdulist[0].header['INSTRUME'] = 'GMOS-S'")
    throwexcept = False
    try:
        print('\tad.write()   #expected failure because clobber=False')
        ad.write()
    except Errors.AstroDataError,e:
        throwexcept = True
        print("\t%s" % e)
    assert_true(throwexcept)
    ad.close()
    os.remove('./test4.fits')
    print('\t#ad closed and test4.fits removed')

def accessmode_test5():
    '''accessmode_test5 -ad write to same file (updated, with clobber)
    '''
    print('\n\t* testfile: %s' % testfile)
    call('cp ' + testfile + ' test5.fits', shell=True)
    print('\t#moved testfile into current directory as "test5.fits"')
    ad = AstroData('test5.fits', mode='update')
    print('\tad = AstroData("test5.fits", mode="update")')
    ad.hdulist[0].header['INSTRUME'] = 'GMOS-S'
    print('\t#change keyword INSTRUME from GMOS-N to GMOS-S')
    print("\tad.hdulist[0].header['INSTRUME'] = 'GMOS-S'")
    ad.write(clobber=True)
    print('\tad.write(clobber=True)')
    ad.close()
    print('\tad.close()')
    ad = AstroData('test5.fits')
    print('\tad = AstroData("test5.fits")   #re-open ad instance')
    ok_(ad.hdulist[0].header['INSTRUME'] == 'GMOS-S', msg='Keyword is different')
    print('\t#assert the "INSTRUME" keyword is still GMOS-S')
    os.remove('./test5.fits')
    ad.close()
    print('\t#ad closed and test5.fits removed')

