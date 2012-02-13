import pyfits
import numpy
from nose.plugins.skip import Skip, SkipTest
from nose.tools import eq_, raises, ok_

import astrodata
from astrodata import AstroData
from file_urls import sci123, sci1


def checkad(ad):
    #check mode
    ok_(ad.mode != None, msg='mode is None')
    
    #check type subdata
    for i in range(len(ad)):
        eq_(type(ad[i]),astrodata.AstroData)
    
    #check AstroData subdata 
    for i in range(len(ad)):
        exn = ad[i].extname()
        exv = ad[i].extver()
        eq_(id(ad[i]), id(ad[exn, exv]), msg='object ids are different')
    
    #check phu type
    eq_(type(ad.phu),pyfits.core.PrimaryHDU)
    
    #check phu propagation
    checkid = id(ad.phu)
    eq_(id(ad.phu),id(ad.hdulist[0]), msg='objects ids are different')
    for i in range(len(ad)):
        eq_(checkid, id(ad[i].hdulist[0]), msg='object ids are different')

    #check phu.header propagation
    checkid = id(ad.phu.header)
    eq_(id(ad.phu.header),id(ad.hdulist[0].header),\
        msg='objects ids are different')
    for i in range(len(ad)):
        eq_(checkid, id(ad[i].hdulist[0].header), \
            msg='object ids are different')
    
    #check imageHdu propagation
    for i in range(len(ad)):
        idhdu1 = id(ad.hdulist[i+1].data)
        idhdu2 = id(ad[i].hdulist[1].data)
        eq_(idhdu1, idhdu2, msg='object ids are different')
    
    ad.close()

def test1():
    '''ASTRODATA-open TEST 1: Pass filename (MEF, SCI123)'''
    ad = AstroData(sci123)
    checkad(ad)

def test2():
    '''ASTRODATA-open TEST 2: Pass filename (SEF, SCI1)'''
    ad = AstroData(sci1)
    checkad(ad)

def test3():
    '''ASTRODATA-open TEST 3: Pass pyfits HDUList (MEF, SCI123)'''
    hdulist_mef = pyfits.open(sci123)
    ad = AstroData(hdulist_mef)
    checkad(ad)
    hdulist_mef.close()

def test4():
    '''ASTRODATA-open TEST 4: Pass pyfits HDUList (SEF, SCI1)'''
    hdulist_sef = pyfits.open(sci1)
    ad = AstroData(hdulist_sef)
    checkad(ad)
    hdulist_sef.close()

def test5():
    '''ASTRODATA-open TEST 5: Pass None (AstroData())'''
    ad = AstroData()
    checkad(ad)

def test6():
    '''ASTRODATA-open TEST 6: Pass param phu=pyfits PHU'''
    hdulist_sef = pyfits.open(sci1)
    ad = AstroData(phu=hdulist_sef[0])
    checkad(ad)
    hdulist_sef.close()

def test7():
    '''ASTRODATA-open TEST 7: Pass param phu=pyfits header'''
    hdulist_sef = pyfits.open(sci1)
    ad = AstroData(phu=hdulist_sef[0].header)
    checkad(ad)
    hdulist_sef.close()

def test8():
    '''ASTRODATA-open TEST 8: Pass params phu, header, and data'''
    hdulist_sef = pyfits.open(sci1)
    phu = hdulist_sef[0]
    header = hdulist_sef[1].header
    data = hdulist_sef[1].data
    ad = AstroData(phu=phu, header=header, data=data)
    checkad(ad)
    hdulist_sef.close()

def test9():
    '''ASTRODATA-open TEST 9: Pass params header and data'''
    hdulist_sef = pyfits.open(sci1)
    header = hdulist_sef[1].header
    data = hdulist_sef[1].data
    ad = AstroData(header=header, data=data)
    checkad(ad)
    hdulist_sef.close()



     
