from astrodata.adutils.testutil import sci123, AstroData, eq_

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
