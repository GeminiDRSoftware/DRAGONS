from astrodata.adutils.testutil \
    import raises, AstroData, sci123, scivardq123, Errors 

@raises(Errors.AstroDataError)
def test10():
    """ASTRODATA-insert TEST 10: AUTO NUMBER, Fail when header not provided with data"""
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    adsci = ad4['SCI', 1]
    ad1.insert(index=0, data=adsci.data, auto_number=True)
