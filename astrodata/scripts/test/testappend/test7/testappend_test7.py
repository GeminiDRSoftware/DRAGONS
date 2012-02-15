from astrodata.adutils.testutil import AstroData, sci123, scivardq123, eq_ 

def test7():
    """ASTRODATA-append TEST 7: AUTO NUMBER, sci1 to sci123"""
    ad1 = AstroData(sci123)
    ad4 = AstroData(scivardq123) 
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.append(header=adsci.header, data=adsci.data, auto_number=True)
    print "ad1.append(header=adsci.header, data=adsci.data, auto_number=True)"
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[3].hdulist[1].name, "SCI")
    eq_(ad1[3].extver(), 4)

