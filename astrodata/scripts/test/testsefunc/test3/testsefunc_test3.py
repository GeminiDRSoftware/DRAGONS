from astrodata.adutils.testutil import AstroData, eq_, sci1

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
