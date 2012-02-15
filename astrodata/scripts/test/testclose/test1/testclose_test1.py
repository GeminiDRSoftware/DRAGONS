from  astrodata.adutils.testutil import eq_, sci123, sci1, AstroData 

def test1():
    '''ASTRODATA-close TEST 1: MEF, Closing AD will cause hdulist to be None
    '''
    ad = AstroData(sci123)
    ad.close()
    eq_(ad.hdulist, None, msg='ad.hdulist is not None')
