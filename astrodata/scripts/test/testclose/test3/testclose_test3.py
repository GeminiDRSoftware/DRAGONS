from  astrodata.adutils.testutil import eq_, sci1, AstroData 

def test3():
    '''ASTRODATA-close TEST 3: SEF, Closing AD will cause hdulist to be None
    '''
    ad = AstroData(sci1)
    ad.close()
    eq_(ad.hdulist, None, msg='ad.hdulist is not None')
