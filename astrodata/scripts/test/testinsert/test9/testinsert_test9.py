from astrodata.adutils.testutil \
    import AstroData, sci123, scivardq123, eq_ 

def test9():
    """ASTRODATA-insert TEST 9: AUTO NUMBER, XVER param inserts high XVER """
    ad1 = AstroData(sci123) 
    ad4 = AstroData(scivardq123) 
    print "\n             >>>>>>>     AD    <<<<<<<<"
    ad1.hdulist.__delitem__(3)
    print("ad1.hdulist.__delitem__(3)")
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['VAR', 3]
    print("adsci = ad4['VAR', 3]")
    print("ad1.insert(index=2, header=adsci.header, data=adsci.data,"
        "auto_number=True)")
    ad1.insert(index=2, header=adsci.header, data=adsci.data, \
        auto_number=True, extver=5)
    print "\n             >>>>>>>  AD MODIFIED <<<<<<<<"
    ad1.info()
    eq_(ad1[2].extname(), "VAR")
    eq_(ad1[2].extver(), 5)
