from astrodata.adutils.testutil import raises, \
    AstroData, Errors, sci123, scivardq123

@ raises(Errors.AstroDataError)
def test10():
    """ASTRODATA-append TEST 10: AUTO NUMBER, Fail when data with no header"""
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    adsci = ad4['SCI', 1]
    ad1.append(data=adsci.data, auto_number=True)

