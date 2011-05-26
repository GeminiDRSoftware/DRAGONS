import sys
import os

from nose.tools import *

import file_urls 
from astrodata import AstroData

testfile = file_urls.testdatafile_1

def ad_close_test1():
    '''AD_CLOSE: test1 -calls hdulist.close() and sets ad.hdulist to None  
    '''
    print('\n\t* testfile: %s' % testfile)
    ad = AstroData(testfile)
    print('\tad = AstroData(testfile)')
    print('\tad.hdulist == %s' % ad.hdulist)
    print('\tad.close()')
    ad.close()
    eq_(ad.hdulist, None, msg='ad.hdulist is not None')
    print('\tad.hdulist == None')

def ad_close_test2():
    '''AD_CLOSE: test2 -subdata will not close the original hdulist  
    '''
    print('\n\t* testfile: %s' % testfile)
    ad = AstroData(testfile)
    print('\tad = AstroData(testfile)')
    sd = ad['SCI,1']
    print("\tsd = ad['SCI,1']")
    sd.close()
    print('\tsd.close()')
    assert_not_equal(ad.hdulist, None, msg='original hdulist is None')
    print('\tad.hdulist != None')
    ad.close() 

