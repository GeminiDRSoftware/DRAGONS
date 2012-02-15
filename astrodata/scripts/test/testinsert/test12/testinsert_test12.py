from astrodata.adutils.testutil import AstroData, raises, sci123, scivardq123 

@raises(TypeError)
def test12():
    """ASTRODATA-insert TEST 12: Fail when no index found
    """
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    ad1.insert(moredata=ad4)
