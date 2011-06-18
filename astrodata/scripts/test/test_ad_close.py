import sys
import os

from nose.tools import *

import file_urls 
from astrodata import AstroData

mef_file = file_urls.testdatafile_1
sef_file = file_urls.testdatafile_2

def ad_close_test1():
    '''ad_close_test1 -close MEF (hdulist is None)  
    '''
    print('\n\t* mef_file: %s' % mef_file)
    ad = AstroData(mef_file)
    print('\tad = AstroData(mef_file)')
    print('\tad.hdulist == %s' % ad.hdulist)
    print('\tad.close()')
    ad.close()
    eq_(ad.hdulist, None, msg='ad.hdulist is not None')
    print('\tad.hdulist == None')

def ad_close_test2():
    '''ad_close_test2 -close MEF subdata (adhdulist stays open)  
    '''
    print('\n\t* mef_file: %s' % mef_file)
    ad = AstroData(mef_file)
    print('\tad = AstroData(mef_file)')
    sd = ad['SCI,1']
    print("\tsd = ad['SCI,1']")
    sd.close()
    print('\tsd.close()')
    assert_not_equal(ad.hdulist, None, msg='original hdulist is None')
    print('\tad.hdulist != None')
    ad.close() 

def ad_close_test3():
    '''ad_close_test3 -close single ext fits (hdulist is None)  
    '''
    print('\n\t* sef_file: %s' % sef_file)
    ad = AstroData(sef_file)
    print('\tad = AstroData(sef_file)')
    print('\tad.hdulist == %s' % ad.hdulist)
    print('\tad.close()')
    ad.close()
    eq_(ad.hdulist, None, msg='ad.hdulist is not None')
    print('\tad.hdulist == None')

def ad_close_test4():
    '''ad_close_test4 -close single ext fits subdata (ad.hdulist stays open)  
    '''
    print('\n\t* sef_file: %s' % sef_file)
    ad = AstroData(sef_file)
    print('\tad = AstroData(sef_file)')
    sd = ad['SCI,1']
    print("\tsd = ad['SCI,1']")
    sd.close()
    print('\tsd.close()')
    assert_not_equal(ad.hdulist, None, msg='original hdulist is None')
    print('\tad.hdulist != None')
    ad.close() 

