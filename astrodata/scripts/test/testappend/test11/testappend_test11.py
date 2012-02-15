from astrodata.adutils.testutil import AstroData, eq_, sci123, scivardq123

def test11():
    """ASTRODATA-append TEST 11: AUTO NUMBER, Do not alter extver if > existing"""
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.append(extver=10, header=adsci.header, data=adsci.data, auto_number=True)
    print("ad1.append(extver=10, header=adsci.header, data=adsci.data,"
          " auto_number=True)")
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[3].extname(), "SCI")
    eq_(ad1[3].extver(), 10)

