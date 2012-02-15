from astrodata.adutils.testutil import AstroData, eq_, mdfscivardq1

def test16():
    """ASTRODATA-append TEST 16: AUTO NUMBER MDF
    """
    ad3 = AstroData(mdfscivardq1) 
    print "\n             >>>>>>>     AD NEW    <<<<<<<<"
    ad_new = AstroData(phu=ad3.phu)
    ad_new.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    ad_new.append(header=ad3[0].header, data=ad3[0].data,\
        auto_number=True) 
    mystr = "ad_new.append(header=ad4[1].header, data=ad4[1].data,"
    mystr += " auto_number=True)" 
    print mystr
    print "\n             >>>>>>>  AD NEW <<<<<<<<"
    ad_new.info()
    eq_(ad_new[0].extname(), "MDF")
    eq_(ad_new[0].extver(), None)

