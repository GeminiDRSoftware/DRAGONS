from astrodata.adutils.testutil import sci123, AstroData, eq_  

def test1():
    """ASTRODATA-remove TEST 1: Use index"""
    ad1 = AstroData(sci123) 
    print "\n             >>>>>>>     AD       <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD remove   <<<<<<<<"
    ad1.remove(1)
    print("ad1.remove(1)")
    print "\n             >>>>>>>  AD MODIFIED   <<<<<<<<"
    ad1.info()
    eq_(ad1[0].extname(), "SCI")
    eq_(ad1[0].extver(), 2)
    eq_(ad1[1].extname(), "SCI")
    eq_(ad1[1].extver(), 3)
