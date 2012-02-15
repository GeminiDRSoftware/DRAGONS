from  astrodata.adutils.testutil import assert_not_equal, sci1, AstroData

def test4():
    '''ASTRODATA-close TEST 4: SEF, Closing AD subdata will not affect hdulist  
    '''
    ad = AstroData(sci1)
    ad['SCI', 1].close()
    assert_not_equal(ad.hdulist, None, msg='original hdulist is None')
    ad.close() 

