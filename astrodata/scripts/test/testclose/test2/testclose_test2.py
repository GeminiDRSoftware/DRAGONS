from  astrodata.adutils.testutil \
    import AstroData, assert_not_equal, sci123 

def test2():
    '''ASTRODATA-close TEST 2: MEF, Closing AD subdata will not affect hdulist  
    '''
    ad = AstroData(sci123)
    ad['SCI',1].close()
    assert_not_equal(ad.hdulist, None, msg='original hdulist is None')
    ad.close() 
