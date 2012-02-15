from astrodata.adutils.testutil import sci123, AstroData, eq_ 

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
