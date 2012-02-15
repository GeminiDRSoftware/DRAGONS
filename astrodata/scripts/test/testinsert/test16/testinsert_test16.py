from astrodata.adutils.testutil \
    import AstroData, mdfscivardq1, eq_ 

def test16():
    """ASTRODATA-insert TEST 16: AUTO NUMBER, MDF to empty AD
    """
    ad3 = AstroData(mdfscivardq1) 
    print "\n             >>>>>>>     AD     <<<<<<<<"
    ad_new = AstroData(phu=ad3.phu)
    ad_new.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    ad_new.insert(index=0, header=ad3[0].header, data=ad3[0].data,\
        auto_number=True) 
    mystr = "ad_new.insert(index=0, header=ad4[1].header, data=ad4[1].data,"
    mystr += " auto_number=True)" 
    print mystr
    print "\n             >>>>>>>  AD MODIFIED <<<<<<<<"
    ad_new.info()
    eq_(ad_new[0].extname(), "MDF")
    eq_(ad_new[0].extver(), None)
