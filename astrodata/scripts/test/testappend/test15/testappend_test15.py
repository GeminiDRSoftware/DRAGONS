from astrodata.adutils.testutil import AstroData, eq_, scivardq123

def test15():
    """ASTRODATA-append TEST 15: AUTO NUMBER, extver param override (dq2-dq1)
    """
    ad4 = AstroData(scivardq123) 
    print "\n             >>>>>>>     AD NEW    <<<<<<<<"
    ad_new = AstroData(phu=ad4.phu)
    ad_new.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    print ad4.info()
    ad_new.append(header=ad4[1].header, data=ad4[1].data,  extver=1, \
        auto_number=True) 
    ad_new.append(header=ad4[4].header, data=ad4[4].data,  extver=1, \
        auto_number=True) 
    mystr = "ad_new.append(header=ad4[1].header, data=ad4[1].data,  extver=1,"
    mystr += " auto_number=True)" 
    mystr += "\nad_new.append(header=ad4[4].header, data=ad4[4].data,  extver=1,"
    mystr += " auto_number=True)"
    print mystr
    print "\n             >>>>>>>  AD NEW <<<<<<<<"
    ad_new.info()
    eq_(ad_new[0].extname(), "SCI")
    eq_(ad_new[0].extver(), 1)
    eq_(ad_new[1].extname(), "DQ")
    eq_(ad_new[1].extver(), 1)

