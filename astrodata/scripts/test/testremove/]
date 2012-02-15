from astrodata.adutils.testutil import Errors, AstroData, sci123, raises  

@raises(Errors.AstroDataError)
def test3():
    """ASTRODATA-remove TEST 3: Fail when given index out of range"""
    ad1 = AstroData(sci123) 
    print "\n             >>>>>>>     AD       <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD remove   <<<<<<<<"
    ad1.remove(3)
    print("ad1.remove(3)")
