from astrodata.adutils.testutil \
    import AstroData, eq_, sci123, scivardq123 

def test8():
    """ASTRODATA-insert TEST 8: Increment XVER if XNAME unknown
    """
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    print "\n             >>>>>>>     AD     <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['VAR', 2]
    print("adsci = ad4['VAR', 2]")
    ad1.insert(index=0, header=adsci.header, data=adsci.data, auto_number=True)
    print "ad1.insert(index=0,header=adsci.header, data=adsci.data, auto_number=True)"
    print "\n             >>>>>>>  AD MODIFIED <<<<<<<<"
    ad1.info()
    eq_(ad1[0].extname(), "VAR")
    eq_(ad1[0].extver(), 4)
