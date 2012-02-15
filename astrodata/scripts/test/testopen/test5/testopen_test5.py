from astrodata.adutils.testutil import checkad, AstroData 

def test5():
    '''ASTRODATA-open TEST 5: Pass None (AstroData())'''
    ad = AstroData()
    checkad(ad)
