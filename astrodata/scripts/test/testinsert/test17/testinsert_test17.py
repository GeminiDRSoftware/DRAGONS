from astrodata.adutils.testutil \
    import AstroData, mdfscivardq1, sci123, eq_ 

def test17():
    """ASTRODATA-insert TEST 17: AUTO NUMBER, MDF into MEF
    """
    ad3 = AstroData(mdfscivardq1) 
    ad1 = AstroData(sci123) 
    print "\n             >>>>>>>     AD     <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    ad1.insert(index=0, header=ad3[0].header, data=ad3[0].data) 
    mystr = "ad1.insert(index=0, header=ad3[1].header, data=ad3[1].data,"
    mystr += " auto_number=True)" 
    print mystr
    print "\n             >>>>>>>  AD (MODIFIED) <<<<<<<<"
    ad1.info()
    eq_(ad1[0].extname(), "MDF")
    eq_(ad1[0].extver(), None)








