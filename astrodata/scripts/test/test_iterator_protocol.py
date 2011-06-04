from nose.tools import *

import file_urls 
from astrodata import AstroData

testfile = file_urls.testdatafile_1

def iterator_protocol_test1():
    '''iterator_protocol_test1 -Use __iter__() and next() to iterate over ad
    '''
    print('\n\t#Also tests that AstroData Objects are being iterated and')
    print('\t#that the iterated ad is mapped to the original ad.hdulist')
    print('\t*testfile: %s' % testfile)
    ad = AstroData(testfile)
    print('\tad = AstroData(testfile)')
    aditerImageObjectIdList = []
    hduImageObjectIdList = []
    for a in ad:
        isinstance(a, AstroData)
        aditerImageObjectIdList.append(id(a.hdulist[1]))
    #skipping phu
    for i in range(1,4):
        hduImageObjectIdList.append(id(ad.hdulist[i]))
    print('\t#Created two lists to compare image object ids')
    print('\taditerImageObjectIdList: %s' % str(aditerImageObjectIdList)) 
    print('\thduImageObjectIdList: %s' % str(hduImageObjectIdList))
    ok_(aditerImageObjectIdList == hduImageObjectIdList, \
        msg='Object ids are not the same')
    print('\t#Assert the two lists are the same')
    ad.close()
    print('\tad.close()')

