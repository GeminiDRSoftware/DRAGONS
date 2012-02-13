from nose.tools import eq_, raises

from file_urls import *
from astrodata import AstroData
from astrodata import Errors

def runappend(f1=None, f2=None, auto=False):
    ad = AstroData(f1)
    md = AstroData(f2)
    pstr = "\n\n             >>>>>>>     AD     <<<<<<<<\n"
    pstr += str(ad.infostr())
    pstr += "\n\n             >>>>>>>    AD APPEND   <<<<<<<<\n"
    pstr += str(md.infostr())
    ad.append(moredata=md, auto_number=auto)
    pstr +="\n\n             >>>>>>>  NEW AD <<<<<<<<\n"
    pstr += str(ad.infostr())
    print(pstr)
    return ad

def test1():
    """ASTRODATA-append TEST 1: AUTO NUMBER, mdfscivardq1 to sci123"""
    ad = runappend(f1=sci123, f2=mdfscivardq1, auto=True) 
    eq_(ad[3].extname(), "MDF")
    eq_(ad[4].extname(), "SCI")
    eq_(ad[5].extname(), "VAR")
    eq_(ad[6].extname(), "DQ")
    eq_(ad[4].extver(), 4)
    eq_(ad[5].extver(), 4)
    eq_(ad[6].extver(), 4)

def test2():
    """ASTRODATA-append TEST 2: AUTO NUMBER, sci123 to mdfscivardq1"""
    ad = runappend(f1=mdfscivardq1, f2=sci123, auto=True) 
    eq_(ad[4].extname(), "SCI")
    eq_(ad[5].extname(), "SCI")
    eq_(ad[6].extname(), "SCI")
    eq_(ad[4].extver(), 2)
    eq_(ad[5].extver(), 3)
    eq_(ad[6].extver(), 4)

def test3():
    """ASTRODATA-append TEST 3: AUTO NUMBER, sci123 to scivardq123"""
    ad = runappend(f1=scivardq123, f2=sci123, auto=True) 
    eq_(ad[9].extname(), "SCI")
    eq_(ad[10].extname(), "SCI")
    eq_(ad[11].extname(), "SCI")
    eq_(ad[9].extver(), 4)
    eq_(ad[10].extver(), 5)
    eq_(ad[11].extver(), 6)

def test4():
    """ASTRODATA-append TEST 4: AUTO NUMBER, scivardq123 to sci123"""
    ad = runappend(f1=sci123, f2=scivardq123, auto=True) 
    eq_(ad[3].extname(), "SCI")
    eq_(ad[4].extname(), "SCI")
    eq_(ad[5].extname(), "SCI")
    eq_(ad[6].extname(), "DQ")
    eq_(ad[7].extname(), "DQ")
    eq_(ad[8].extname(), "DQ")
    eq_(ad[9].extname(), "VAR")
    eq_(ad[10].extname(), "VAR")
    eq_(ad[11].extname(), "VAR")
    eq_(ad[3].extver(), 4)
    eq_(ad[4].extver(), 5)
    eq_(ad[5].extver(), 6)
    eq_(ad[6].extver(), 4)
    eq_(ad[7].extver(), 5)
    eq_(ad[8].extver(), 6)
    eq_(ad[9].extver(), 4)
    eq_(ad[10].extver(), 5)
    eq_(ad[11].extver(), 6)

def test5():
    """ASTRODATA-append TEST 5: AUTO NUMBER, scivardq123 to mdfscivardq1"""
    ad = runappend(f1=mdfscivardq1, f2=scivardq123, auto=True) 
    eq_(ad[4].extname(), "SCI")
    eq_(ad[5].extname(), "SCI")
    eq_(ad[6].extname(), "SCI")
    eq_(ad[7].extname(), "DQ")
    eq_(ad[8].extname(), "DQ")
    eq_(ad[9].extname(), "DQ")
    eq_(ad[10].extname(), "VAR")
    eq_(ad[11].extname(), "VAR")
    eq_(ad[12].extname(), "VAR")
    eq_(ad[4].extver(), 2)
    eq_(ad[5].extver(), 3)
    eq_(ad[6].extver(), 4)
    eq_(ad[7].extver(), 2)
    eq_(ad[8].extver(), 3)
    eq_(ad[9].extver(), 4)
    eq_(ad[10].extver(), 2)
    eq_(ad[11].extver(), 3)
    eq_(ad[12].extver(), 4)

def test6():
    """ASTRODATA-append TEST 6: AUTO NUMBER, mdfscivardq1 to scivardq123"""
    ad = runappend(f1=scivardq123, f2=mdfscivardq1, auto=True) 
    eq_(ad[9].extname(), "MDF")
    eq_(ad[10].extname(), "SCI")
    eq_(ad[11].extname(), "VAR")
    eq_(ad[12].extname(), "DQ")
    eq_(ad[10].extver(), 4)
    eq_(ad[11].extver(), 4)
    eq_(ad[12].extver(), 4)

def test7():
    """ASTRODATA-append TEST 7: AUTO NUMBER, sci1 to sci123"""
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.append(header=adsci.header, data=adsci.data, auto_number=True)
    print "ad1.append(header=adsci.header, data=adsci.data, auto_number=True)"
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[3].hdulist[1].name, "SCI")
    eq_(ad1[3].extver(), 4)

def test8():
    """ASTRODATA-append TEST 8: AUTO NUMBER, var2 to sci123"""
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    adsci = ad4['VAR', 2]
    print("adsci = ad4['VAR', 2]")
    ad1.append(header=adsci.header, data=adsci.data, auto_number=True)
    print "ad1.append(header=adsci.header, data=adsci.data, auto_number=True)"
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[3].extname(), "VAR")
    eq_(ad1[3].extver(), 4)


def test9():
    """ASTRODATA-append TEST 9: AUTO NUMBER, var3 to sci12"""
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
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

@ raises(Errors.AstroDataError)
def test10():
    """ASTRODATA-append TEST 10: AUTO NUMBER, Fail when data with no header"""
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
    adsci = ad4['SCI', 1]
    ad1.append(data=adsci.data, auto_number=True)

def test11():
    """ASTRODATA-append TEST 11: AUTO NUMBER, Do not alter extver if > existing"""
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.append(extver=10, header=adsci.header, data=adsci.data, auto_number=True)
    print("ad1.append(extver=10, header=adsci.header, data=adsci.data,"
          " auto_number=True)")
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[3].extname(), "SCI")
    eq_(ad1[3].extver(), 10)

@ raises(Errors.AstroDataError)
def test12():
    """ASTRODATA-append TEST 12: Fail scivardq123 to sci123 (NO AUTO NUMBER)"""
    ad = runappend(f1=sci123, f2=scivardq123) 

@ raises(Errors.AstroDataError)
def test13():
    """ASTRODATA-append TEST 13: Fail sci1 to sci123 (NO AUTO NUMBER)"""
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.append(header=adsci.header, data=adsci.data)


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


def test16():
    """ASTRODATA-append TEST 16: AUTO NUMBER MDF
    """
    ad3 = AstroData(mdfscivardq1) 
    print "\n             >>>>>>>     AD NEW    <<<<<<<<"
    ad_new = AstroData(phu=ad3.phu)
    ad_new.info()
    print "\n             >>>>>>>    AD APPEND   <<<<<<<<"
    #print ad3[0].data.__class__
    ad_new.append(header=ad3[0].header, data=ad3[0].data,\
        auto_number=True) 
    mystr = "ad_new.append(header=ad4[1].header, data=ad4[1].data,"
    mystr += " auto_number=True)" 
    print mystr
    print "\n             >>>>>>>  AD NEW <<<<<<<<"
    ad_new.info()
    eq_(ad_new[0].extname(), "MDF")
    eq_(ad_new[0].extver(), None)




