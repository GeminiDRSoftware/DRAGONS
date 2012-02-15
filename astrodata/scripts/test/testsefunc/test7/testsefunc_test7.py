from astrodata.adutils.testutil import Errors, AstroData, eq_, pyfits, sci123
from numpy import array

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





    
