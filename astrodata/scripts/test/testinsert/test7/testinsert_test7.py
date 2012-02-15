from astrodata.adutils.testutil import eq_, AstroData, sci123, scivardq123 

def test7():
    """
    ASTRODATA-insert TEST 7: AUTO NUMBER, Incremnt XVER & latter indx if XNAME exists
    """
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    print "\n             >>>>>>>     AD    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.insert(index=1, header=adsci.header, data=adsci.data, auto_number=True)
    print "ad1.insert(index=1, header=adsci.header, data=adsci.data, auto_number=True)"
    print "\n             >>>>>>>  MODIFIED AD   <<<<<<<<"
    ad1.info()
    eq_(ad1[1].hdulist[1].name, "SCI")
    eq_(ad1[1].extver(), 4)
