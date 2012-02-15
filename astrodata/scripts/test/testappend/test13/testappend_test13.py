from astrodata.adutils.testutil import AstroData, Errors, raises, sci123, scivardq123

@raises(Errors.AstroDataError)
def test13():
    """ASTRODATA-append TEST 13: Fail sci1 to sci123 """
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.append(header=adsci.header, data=adsci.data)

