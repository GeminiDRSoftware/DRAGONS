from nose.tools import *

import file_urls 
from astrodata import AstroData

mef_file = file_urls.testdatafile_1
sef_file = file_urls.testdatafile_1

def iterprotocol(ad):
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

def iterator_protocol_test1():
    '''iterator_protocol_test1 -MEF __iter__() and next()
    '''
    print('\n\t#Also tests that AstroData Objects are being iterated and')
    print('\t#that the iterated ad is mapped to the original ad.hdulist')
    print('\t*mef_file: %s' % mef_file)
    print('\tad = AstroData(mef_file)')
    ad = AstroData(mef_file)
    iterprotocol(ad)

def iterator_protocol_test2():
    '''iterator_protocol_test2 -single ext. __iter__() and next()
    '''
    print('\n\t#Also tests that AstroData Objects are being iterated and')
    print('\t#that the iterated ad is mapped to the original ad.hdulist')
    print('\t*sef_file: %s' % mef_file)
    print('\tad = AstroData(sef_file)')
    ad = AstroData(sef_file)
    iterprotocol(ad)
