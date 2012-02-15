from astrodata.adutils.testutil \
    import raises, Errors, AstroData, sci123, scivardq123 

@raises(Errors.AstroDataError)
def test14():
    """ASTRODATA-insert TEST 14: Fail when XNAM XVER conflict"""
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    ad4.insert(index=4, moredata=ad1)

