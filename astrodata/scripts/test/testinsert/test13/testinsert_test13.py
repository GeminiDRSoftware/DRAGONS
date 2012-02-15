from astrodata.adutils.testutil \
    import AstroData, raises, Errors, sci123, scivardq123 

@raises(Errors.AstroDataError)
def test13():
    """ASTRODATA-insert TEST 13: Fail when matching XNAM XVER found"""
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    ad1.info()
    adsci = ad4['SCI', 1]
    
    # Inserting sci 1 when sci 1 already exists
    ad1.insert(index=0, header=adsci.header, data=adsci.data)
