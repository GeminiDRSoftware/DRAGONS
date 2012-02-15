from astrodata.adutils.testutil import sci123, AstroData, eq_

def test14():
    """ASTRODATA-append TEST 14: AUTO NUMBER, given phu, construct sci123"""
    ad1 = AstroData(sci123) 
    print "\n             >>>>>>>     AD NEW    <<<<<<<<"
    ad_new = AstroData(phu=ad1.phu)
    ad_new.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    for i in range(1,4):
        adsci = ad1['SCI', i]
        print("adsci = ad1['SCI', %d]" % i)
        ad_new.append(header=adsci.header, data=adsci.data, auto_number=True)
        mystr = "ad_new.append(header=adsci.header, data=adsci.data,"
        mystr += " auto_number=True)"
        print mystr
    print "\n             >>>>>>>  AD NEW <<<<<<<<"
    ad_new.info()
    eq_(ad_new[0].extname(), "SCI")
    eq_(ad_new[0].extver(), 1)
    eq_(ad_new[1].extname(), "SCI")
    eq_(ad_new[1].extver(), 2)
    eq_(ad_new[2].extname(), "SCI")
    eq_(ad_new[2].extver(), 3)


