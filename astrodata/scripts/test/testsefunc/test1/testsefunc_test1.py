from astrodata.adutils.testutil import AstroData, eq_, sci1

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
