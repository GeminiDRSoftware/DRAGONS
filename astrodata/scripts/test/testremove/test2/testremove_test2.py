from astrodata.adutils.testutil import AstroData, eq_, sci123  

def test2():
    """ASTRODATA-remove TEST 2: Use 'XNAM', XVER"""
    ad1 = AstroData(sci123) 
    print "\n             >>>>>>>     AD       <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD remove   <<<<<<<<"
    ad1.remove(("SCI",1))
    print("ad1.remove(('SCI',1))")
    print "\n             >>>>>>>  AD  MODIFIED   <<<<<<<<"
    ad1.info()
    eq_(ad1[0].extname(), "SCI")
    eq_(ad1[0].extver(), 2)
    eq_(ad1[1].extname(), "SCI")
