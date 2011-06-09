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
        eq_(type(ad.hdulist[i+1].data), numpy.ndarray)
        print('\ttype(ad.hdulist[%s].data) = numpy.ndarray' % (i+1))

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
    '''ad_open_test5 -open ad instance with pyfits header and 3 ext MEF
    '''
    raise SkipTest
    print('\n\tTest input file: %s' % mef_file)
    hdulist_mef = pyfits.open(mef_file)
    print('\thdulist_mef = pyfits.open(%s)' % mef_file)
    header = hdulist_mef[0].header
    print('\theader = hdulist_mef[1].header')
    data = hdulist_mef[1].data
    print('\tdata = hdulist_mef[1].data')
    ad = AstroData(header=header, data=data)
    print('\tad = AstroData(header=header, data=data)')
    #ad.append(hdulist_mef[3].data)
    #ad.insert(hdulist_mef[2].data, 1)
    checkad(ad)
    hdulist_mef.close()



     
