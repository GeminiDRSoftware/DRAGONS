from nose.tools import *

from  file_urls import sci123, sci1 
from astrodata import AstroData

def test1():
    '''ASTRODATA-close TEST 1: MEF, Closing AD will cause hdulist to be None
    '''
    ad = AstroData(sci123)
    ad.close()
    eq_(ad.hdulist, None, msg='ad.hdulist is not None')

def test2():
    '''ASTRODATA-close TEST 2: MEF, Closing AD subdata will not affect hdulist  
    '''
    ad = AstroData(sci123)
    ad['SCI',1].close()
    assert_not_equal(ad.hdulist, None, msg='original hdulist is None')
    ad.close() 

def test3():
    '''ASTRODATA-close TEST 3: SEF, Closing AD will cause hdulist to be None
    '''
    ad = AstroData(sci1)
    ad.close()
    eq_(ad.hdulist, None, msg='ad.hdulist is not None')

def test4():
    '''ASTRODATA-close TEST 4: SEF, Closing AD subdata will not affect hdulist  
    '''
    ad = AstroData(sci1)
    ad['SCI', 1].close()
    assert_not_equal(ad.hdulist, None, msg='original hdulist is None')
    ad.close() 

