from astrodata.adutils.testutil import AstroData, eq_, sci1, sci123, pyfits

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
