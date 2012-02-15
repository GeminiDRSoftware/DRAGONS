from astrodata.adutils.testutil \
    import AstroData, sci123, scivardq123, eq_  

def test11():
    """
    ASTRODATA-insert TEST 11: AUTO NUMBER, XVER param allows duplicate XNAM XVER pairs  
    """
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    print "\n             >>>>>>>     AD     <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.insert(index=1, extver=2, header=adsci.header, data=adsci.data, \
        auto_number=True)
    print("ad1.insert(index=1, extver=2, header=adsci.header, \
        data=adsci.data, auto_number=True)")
    print "\n             >>>>>>>  AD MODIFIED <<<<<<<<"
    ad1.info()
    eq_(ad1[1].extname(), "SCI")
    eq_(ad1[1].extver(), 2)

