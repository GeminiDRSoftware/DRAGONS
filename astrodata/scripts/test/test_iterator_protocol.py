from nose.tools import ok_ 

from file_urls import sci123, sci1 
from astrodata import AstroData

def iterprotocol(ad):
    aditerImageObjectIdList = []
    hduImageObjectIdList = []
    for a in ad:
        aditerImageObjectIdList.append(id(a.hdulist[1]))
    for phu in ad.hdulist[1:]:
        hduImageObjectIdList.append(id(phu))
    ok_(aditerImageObjectIdList == hduImageObjectIdList, \
        msg='Object ids are not the same')
    ad.close()

def test1():
    '''ASTRODATA-iterator-protocol TEST 1: Compare for AD and for HDUList (MEF)'''
    ad = AstroData(sci123)
    iterprotocol(ad)

def test2():
    '''ASTRODATA-iterator-protocol TEST 2: Compare for AD and for HDUList (SEF)'''
    ad = AstroData(sci1)
    iterprotocol(ad)
