import pyfits
from nose.tools import eq_
from numpy import*

from file_urls import sci123, sci1
from astrodata import Errors
from astrodata import AstroData

def test1():
    '''ASTRODATA-single-extfuncs TEST 1: get_data()  
    '''
    print('\n\t* single ext fits testfile: %s' % sci1)
    ad = AstroData(sci1)
    print('\tad = AstroData(sci1)')
    data = ad.get_data()
    print('\tdata = ad.get_data()')
    print('\tid(data) = %s' % str(id(data)))
    print('\tid(ad.data) = %s' % str(id(ad.data)))
    eq_(id(data), id(ad.data), msg='objects are different')

def test2():
    '''ASTRODATA-single-extfuncs TEST 2: set_data()  
    '''
    print('\n\t* single ext fits testfile: %s' % sci1)
    ad = AstroData(sci1)
    print('\tad = AstroData(sci1)')
    a = array([1,2,3])
    print('\ta = array([1,2,3])')
    ad.set_data(a)
    print('\tad.set_data(a)')
    for i in range(len(ad)):
        eq_(ad.data[i], a[i], msg='array elements are different')

def test3():
    '''ASTRODATA-single-extfuncs TEST 3: get_header() (extension)  
    '''
    print('\n\t* single ext fits testfile: %s' % sci1)
    ad = AstroData(sci1)
    print('\tad = AstroData(sci1)')
    header = ad.get_header()
    print('\theader = ad.get_header)')
    print('\tid(header) = %s' % str(id(header)))
    print('\tid(ad[0].header) = %s' % str(id(ad[0].header)))
    eq_(id(header), id(ad[0].header), msg='objects are different')

def test4():
    '''ASTRODATA-single-extfuncs TEST 4: set_header()  
    '''
    print('\n\t* single ext fits testfile: %s' % sci1)
    hdulist = pyfits.open(sci1)
    print('\thdulist = pyfits.open(sci1)')
    ad = AstroData(sci123)
    print('\tad = AstroData(sci123)')
    ad[0].set_header(hdulist[1].header)
    print('\tad[0].set_header(hdulist[1].header)')
    eq_(ad[0].header, hdulist[1].header, msg='ext. header not set correctly')

def test5():
    '''ASTRODATA-single-extfuncs TEST 5: rename_ext name, ver  
    '''
    print('\n\t* MEF testfile: %s' % sci123)
    ad = AstroData(sci123)
    print('\tad = AstroData(sci123)')
    ad[0].rename_ext('SPAM', 2)
    print('\tad[0].rename_ext("SPAM", 2)')
    eq_(2, ad[0].hdulist[1]._extver, msg='extver not 1')
    eq_('SPAM', ad[0].hdulist[1].name, msg='extname not SPAM')

def test6():
    '''ASTRODATA-single-extfuncs TEST 6: rename_ext with tuple  
    '''
    print('\n\t* MEF testfile: %s' % sci123)
    ad = AstroData(sci123)
    print('\tad = AstroData(sci123)')
    ad[0].rename_ext(('SCI', 10))
    print('\tad[0].rename_ext(("SCI", 10))')
    eq_(10, ad[0].hdulist[1]._extver, msg='extver not 1')
    eq_('SCI', ad[0].hdulist[1].name, msg='extname not SCI')

def test7():
    '''ASTRODATA-single-extfuncs TEST 7: MEF exceptions  
    '''
    print('\n\t* MEF testfile: %s' % sci123)
    hdulist = pyfits.open(sci123)
    print('\thdulist = pyfits.open(sci123)')
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





    
