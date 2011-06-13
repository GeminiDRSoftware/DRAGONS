import pyfits
import numpy
from nose.plugins.skip import Skip, SkipTest
from nose.tools import *

import astrodata
from astrodata import AstroData
import file_urls

mef_file = file_urls.testdatafile_1
sef_file = file_urls.testdatafile_2

def checkad(ad):
    #check mode
    ok_(ad.mode != None, msg='mode is None')
    print('\tad.mode = %s' % ad.mode)
    
    #check type subdata
    for i in range(len(ad)):
        eq_(type(ad[i]),astrodata.AstroData)
        print('\ttype(ad[%s]) = astrodata.AstroData' % str(i))
    
    #check AstroData subdata 
    print('\tcheck AstroData subdata')
    for i in range(len(ad)):
        exn = ad[i].extname()
        exv = ad[i].extver()
        print('\t\tid(ad[%s]) = %s   id(ad[%s, %s]) = %s' % \
            (str(i), id(ad[i]), exn, exv, id(ad[exn, exv])))
        eq_(id(ad[i]), id(ad[exn, exv]), msg='object ids are different')
    print('\t\tAstroData subdata good')
    
    #check phu type
    eq_(type(ad.phu),pyfits.core.PrimaryHDU)
    print('\ttype(ad.phu) = pyfits.core.PrimaryHDU')
    
    #check phu propagation
    print('\tcheck phu propagation')
    checkid = id(ad.phu)
    eq_(id(ad.phu),id(ad.hdulist[0]), msg='objects ids are different')
    print('\t\t%s = id(ad.phu)' % str(checkid))
    print('\t\t%s = id(ad.hdulist[0]' % str(checkid))
    for i in range(len(ad)):
        print('\t\t%s = id(ad[%s].hdulist[0])' % \
            (str(id(ad[i].hdulist[0])), str(i)))
        eq_(checkid, id(ad[i].hdulist[0]), msg='object ids are different')
    print('\t\tphu propagation good')

    #check phu.header propagation
    print('\tcheck phu.header propagation')
    checkid = id(ad.phu.header)
    eq_(id(ad.phu.header),id(ad.hdulist[0].header),\
        msg='objects ids are different')
    print('\t\t%s = id(ad.phu.header)' % str(checkid))
    print('\t\t%s = id(ad.hdulist[0].header' % str(checkid))
    for i in range(len(ad)):
        print('\t\t%s = id(ad[%s].hdulist[0].header)' % \
            (str(id(ad[i].hdulist[0].header)), str(i)))
        eq_(checkid, id(ad[i].hdulist[0].header), \
            msg='object ids are different')
    print('\t\tphu.header propagation good')
    
    #check data type is numpy array
    for i in range(len(ad)):
        typeofdata = str(type(ad.hdulist[i+1].data))
        print('\ttype(ad.hdulist[%s].data) = %s' % ((i+1), typeofdata ))
    
    #check imageHdu propagation
    print('\tcheck data propagation')
    for i in range(len(ad)):
        idhdu1 = id(ad.hdulist[i+1].data)
        idhdu2 = id(ad[i].hdulist[1].data)
        print('\t\t%s = id(ad.hdulist[%s].data)' % (idhdu1, str(i+1)))
        print('\t\t%s = id(ad[%s].hdulist[1].data) ' % (idhdu2, str(i)))
        eq_(idhdu1, idhdu2, msg='object ids are different')
    print('\t\tdata propagation good')
    
    # clean up
    ad.close()

def ad_open_test1():
    '''ad_open_test1 -open astrodata inst. with  3 ext MEF file
    '''
    print('\n\tTest input file: %s' % mef_file)
    ad = AstroData(mef_file)
    print('\tad = AstroData(%s)' % ad.filename)
    checkad(ad)

def ad_open_test2():
    '''ad_open_test2 -open astrodata inst. with single ext. fits file 
    '''
    print('\n\tTest input file: %s' % sef_file)
    ad = AstroData(sef_file)
    print('\tad = AstroData(%s)' % ad.filename)
    checkad(ad)

def ad_open_test3():
    '''ad_open_test3 -open astrodata inst. with pyfits hdulist (3 ext MEF)
    '''
    print('\n\tTest input file: %s' % mef_file)
    hdulist_mef = pyfits.open(mef_file)
    print('\thdulist_mef = pyfits.open(%s)' % mef_file)
    ad = AstroData(hdulist_mef)
    print('\tad = AstroData(hdulist_mef)')
    checkad(ad)
    hdulist_mef.close()

def ad_open_test4():
    '''ad_open_test4 -open ad instance with pyfits hdulist (single ext. fits)
    '''
    print('\n\tTest input file: %s' % sef_file)
    hdulist_sef = pyfits.open(sef_file)
    print('\thdulist_sef = pyfits.open(%s)' % sef_file)
    ad = AstroData(hdulist_sef)
    print('\tad = AstroData(hdulist_sef)')
    checkad(ad)
    hdulist_sef.close()

def ad_open_test5():
    '''ad_open_test5 -open null ad instance  
    '''
    ad = AstroData()
    print('\tad = AstroData()')
    checkad(ad)

def ad_open_test6():
    '''ad_open_test6 -open ad with phu (pyfits PrimaryHDU from single ext fits)
    '''
    print('\n\tTest input file: %s' % sef_file)
    hdulist_sef = pyfits.open(sef_file)
    print('\thdulist_sef = pyfits.open(%s)' % sef_file)
    ad = AstroData(phu=hdulist_sef[0])
    print('\tad = AstroData(phu=hdulist_sef[0])')
    checkad(ad)
    hdulist_sef.close()

def ad_open_test7():
    '''ad_open_test7 -open ad with phu = PrimaryHDU.header (single ext fits)
    '''
    print('\n\tTest input file: %s' % sef_file)
    hdulist_sef = pyfits.open(sef_file)
    print('\thdulist_sef = pyfits.open(%s)' % sef_file)
    ad = AstroData(phu=hdulist_sef[0].header)
    print('\tad = AstroData(phu=hdulist_sef[0].header)')
    checkad(ad)
    hdulist_sef.close()

def ad_open_test8():
    '''ad_open_test8 -open ad with phu, header, and data (single ext fits)
    '''
    print('\n\tTest input file: %s' % sef_file)
    hdulist_sef = pyfits.open(sef_file)
    print('\thdulist_sef = pyfits.open(%s)' % sef_file)
    phu = hdulist_sef[0]
    print('phu = hdulist_sef[0]')
    header = hdulist_sef[1].header
    print('header = hdulist_sef[1].header')
    data = hdulist_sef[1].data
    print('data = hdulist_sef[1].data')
    ad = AstroData(phu=phu, header=header, data=data)
    print('\tad = AstroData(phu=phu, header=header, data=data)')
    checkad(ad)
    hdulist_sef.close()

def ad_open_test9():
    '''ad_open_test9 -open ad with header and data (no phu, single ext fits)
    '''
    print('\n\tTest input file: %s' % sef_file)
    hdulist_sef = pyfits.open(sef_file)
    print('\thdulist_sef = pyfits.open(%s)' % sef_file)
    header = hdulist_sef[1].header
    print('header = hdulist_sef[1].header')
    data = hdulist_sef[1].data
    print('data = hdulist_sef[1].data')
    ad = AstroData(header=header, data=data)
    print('\tad = AstroData(header=header, data=data)')
    checkad(ad)
    hdulist_sef.close()



     
