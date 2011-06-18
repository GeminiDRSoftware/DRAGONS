import pyfits
from nose.tools import *
from numpy import*

import file_urls 
from astrodata import Errors
from astrodata import AstroData


mef_file = file_urls.testdatafile_1
sef_file = file_urls.testdatafile_2

def single_extfuncs_test1():
    '''single_extfuncs_test1 -get_data()  
    '''
    print('\n\t* single ext fits testfile: %s' % sef_file)
    ad = AstroData(sef_file)
    print('\tad = AstroData(sef_file)')
    data = ad.get_data()
    print('\tdata = ad.get_data()')
    print('\tid(data) = %s' % str(id(data)))
    print('\tid(ad.data) = %s' % str(id(ad.data)))
    eq_(id(data), id(ad.data), msg='objects are different')

def single_extfuncs_test2():
    '''single_extfuncs_test2 -set_data()  
    '''
    print('\n\t* single ext fits testfile: %s' % sef_file)
    ad = AstroData(sef_file)
    print('\tad = AstroData(sef_file)')
    a = array([1,2,3])
    print('\ta = array([1,2,3])')
    ad.set_data(a)
    print('\tad.set_data(a)')
    for i in range(len(ad)):
        eq_(ad.data[i], a[i], msg='array elements are different')

def single_extfuncs_test3():
    '''single_extfuncs_test3 -get_header() (extension)  
    '''
    print('\n\t* single ext fits testfile: %s' % sef_file)
    ad = AstroData(sef_file)
    print('\tad = AstroData(sef_file)')
    header = ad.get_header()
    print('\theader = ad.get_header)')
    print('\tid(header) = %s' % str(id(header)))
    print('\tid(ad[0].header) = %s' % str(id(ad[0].header)))
    eq_(id(header), id(ad[0].header), msg='objects are different')

def single_extfuncs_test4():
    '''single_extfuncs_test4 -set_header()  
    '''
    print('\n\t* single ext fits testfile: %s' % sef_file)
    hdulist = pyfits.open(sef_file)
    print('\thdulist = pyfits.open(sef_file)')
    ad = AstroData()
    print('\tad = AstroData()  # null ad')
    ad.set_header(hdulist[1].header)
    print('\tad.set_header(hdulist[1].header)')
    eq_(ad[0].header, hdulist[1].header, msg='ext. header not set correctly')

def single_extfuncs_test5():
    '''single_extfuncs_test5 -rename_ext()  
    '''
    print('\n\t* single ext fits testfile: %s' % sef_file)
    hdulist = pyfits.open(sef_file)
    print('\thdulist = pyfits.open(sef_file)')
    ad = AstroData()
    print('\tad = AstroData()  # null ad')
    ad.rename_ext('SPAM')
    print('\tad.rename_ext("SPAM")')
    eq_('SPAM', ad.extname(), msg='ext. header not set correctly')

def single_extfuncs_test6():
    '''single_extfuncs_test6 -MEF exceptions  
    '''
    print('\n\t* MEF testfile: %s' % sef_file)
    hdulist = pyfits.open(mef_file)
    print('\thdulist = pyfits.open(mef_file)')
    ad = AstroData(hdulist)
    print('\tad = AstroData(hdulist)  # 3 ext MEF hdulist')
    mes1 = 'This member or method can only be called for '
    mes1 += 'Single HDU AstroData instances'
    try:
        data = ad.get_data()
    except Errors.SingleHDUMemberExcept,s:
        pass
    eq_(str(s), mes1, msg='Exception message different')
    print('\tget_data() exception ok')
    try:
        a = array([1,2,3])
        ad.set_data(a)
    except Errors.SingleHDUMemberExcept,s:
        pass
    eq_(str(s), mes1, msg='Exception message different')
    print('\tset_data() exception ok')
    try:
        header = ad.get_header()
    except Errors.SingleHDUMemberExcept,s:
        pass
    eq_(str(s), mes1, msg='Exception message different')
    print('\tget_header() exception ok')
    try:
        header = ad.set_header(hdulist[2].header)
    except Errors.SingleHDUMemberExcept,s:
        pass
    eq_(str(s), mes1, msg='Exception message different')
    print('\tset_header() exception ok')
    try:
        ad.rename_ext('MDF')
    except Errors.SingleHDUMemberExcept,s:
        pass
    eq_(str(s), mes1, msg='Exception message different')
    print('\trename_ext() exception ok')





    
