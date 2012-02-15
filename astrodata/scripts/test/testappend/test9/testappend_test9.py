from astrodata.adutils.testutil import AstroData, sci123, scivardq123, eq_ 

def test9():
    """ASTRODATA-append TEST 9: AUTO NUMBER, var3 to sci12"""
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.hdulist.__delitem__(3)
    print("ad1.hdulist.__delitem__(3)")
    ad1.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    adsci = ad4['VAR', 3]
    print("adsci = ad4['VAR', 3]")
    print "ad1.append(header=adsci.header, data=adsci.data, auto_number=True)"
    ad1.append(header=adsci.header, data=adsci.data, auto_number=True)
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[2].extname(), "VAR")
    eq_(ad1[2].extver(), 3)

