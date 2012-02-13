from nose.tools import eq_, raises

from file_urls import * 
from astrodata import AstroData
from astrodata import Errors


def runinsert(index=None, f1=None, f2=None, auto=False):
    ad = AstroData(f1)
    md = AstroData(f2)
    pstr = "\n\n             >>>>>>>     AD     <<<<<<<<\n"
    pstr += str(ad.infostr())
    pstr += "\n\n             >>>>>>>    AD APPEND   <<<<<<<<\n"
    pstr += str(md.infostr())
    ad.insert(index=index, moredata=md, auto_number=auto)
    pstr +="\n\n             >>>>>>>  NEW AD <<<<<<<<\n"
    pstr += str(ad.infostr())
    print(pstr)
    return ad

def test1():
    """ASTRODATA-insert TEST 1: AUTO NUMBER, mdfscivardq1 to sci123"""
    ad = runinsert(index=1, f1=sci123, f2=mdfscivardq1, auto=True) 
    eq_(ad[4].extname(), "MDF")
    eq_(ad[3].extname(), "SCI")
    eq_(ad[2].extname(), "VAR")
    eq_(ad[1].extname(), "DQ")
    eq_(ad[3].extver(), 4)
    eq_(ad[2].extver(), 4)
    eq_(ad[1].extver(), 4)

def test2():
    """ASTRODATA-insert TEST 2: AUTO NUMBER, sci123 to mdfscivardq1"""
    ad = runinsert(index=0, f1=mdfscivardq1, f2=sci123, auto=True) 
    eq_(ad[2].extname(), "SCI")
    eq_(ad[1].extname(), "SCI")
    eq_(ad[0].extname(), "SCI")
    eq_(ad[2].extver(), 2)
    eq_(ad[1].extver(), 3)
    eq_(ad[0].extver(), 4)

def test3():
    """ASTRODATA-insert TEST 3: AUTO NUMBER, sci123 to scivardq123"""
    ad = runinsert(index=4, f1=scivardq123, f2=sci123, auto=True) 
    eq_(ad[6].extname(), "SCI")
    eq_(ad[5].extname(), "SCI")
    eq_(ad[4].extname(), "SCI")
    eq_(ad[6].extver(), 4)
    eq_(ad[5].extver(), 5)
    eq_(ad[4].extver(), 6)


def test4():
    """ASTRODATA-insert TEST 4: AUTO NUMBER, scivardq123 to sci123"""
    ad = runinsert(index=1, f1=sci123, f2=scivardq123, auto=True) 
    eq_(ad[9].extname(), "SCI")
    eq_(ad[8].extname(), "SCI")
    eq_(ad[7].extname(), "SCI")
    eq_(ad[6].extname(), "DQ")
    eq_(ad[5].extname(), "DQ")
    eq_(ad[4].extname(), "DQ")
    eq_(ad[3].extname(), "VAR")
    eq_(ad[2].extname(), "VAR")
    eq_(ad[1].extname(), "VAR")
    eq_(ad[9].extver(), 4)
    eq_(ad[8].extver(), 5)
    eq_(ad[7].extver(), 6)
    eq_(ad[6].extver(), 4)
    eq_(ad[5].extver(), 5)
    eq_(ad[4].extver(), 6)
    eq_(ad[3].extver(), 4)
    eq_(ad[2].extver(), 5)
    eq_(ad[1].extver(), 6)

def test5():
    """ASTRODATA-insert TEST 5: AUTO NUMBER, scivardq123 to mdfscivardq1"""
    ad = runinsert(index=1, f1=mdfscivardq1, f2=scivardq123, auto=True) 
    eq_(ad[9].extname(), "SCI")
    eq_(ad[8].extname(), "SCI")
    eq_(ad[7].extname(), "SCI")
    eq_(ad[6].extname(), "DQ")
    eq_(ad[5].extname(), "DQ")
    eq_(ad[4].extname(), "DQ")
    eq_(ad[3].extname(), "VAR")
    eq_(ad[2].extname(), "VAR")
    eq_(ad[1].extname(), "VAR")
    eq_(ad[9].extver(), 2)
    eq_(ad[8].extver(), 3)
    eq_(ad[7].extver(), 4)
    eq_(ad[6].extver(), 2)
    eq_(ad[5].extver(), 3)
    eq_(ad[4].extver(), 4)
    eq_(ad[3].extver(), 2)
    eq_(ad[2].extver(), 3)
    eq_(ad[1].extver(), 4)

@raises(Errors.AstroDataError)
def test6():
    """ASTRODATA-insert TEST 6: Fail when index out of range"""
    ad = runinsert(index=11, f1=scivardq123, f2=mdfscivardq1, auto=True) 

def test7():
    """
    ASTRODATA-insert TEST 7: AUTO NUMBER, Incremnt XVER & latter indx if XNAME exists
    """
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
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

def test8():
    """ASTRODATA-insert TEST 8: Increment XVER if XNAME unknown
    """
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
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

def test9():
    """ASTRODATA-insert TEST 9: AUTO NUMBER, XVER param inserts high XVER """
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
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

@raises(Errors.AstroDataError)
def test10():
    """ASTRODATA-insert TEST 10: AUTO NUMBER, Fail when header not provided with data"""
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
    adsci = ad4['SCI', 1]
    ad1.insert(index=0, data=adsci.data, auto_number=True)

def test11():
    """
    ASTRODATA-insert TEST 11: AUTO NUMBER, XVER param allows duplicate XNAM XVER pairs  
    """
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
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

@ raises(TypeError)
def test12():
    """ASTRODATA-insert TEST 12: Fail when no index found
    """
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
    ad1.insert(moredata=ad4)

@ raises(Errors.AstroDataError)
def test13():
    """ASTRODATA-insert TEST 13: Fail when matching XNAM XVER found"""
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
    ad1.info()
    adsci = ad4['SCI', 1]
    
    # Inserting sci 1 when sci 1 already exists
    ad1.insert(index=0, header=adsci.header, data=adsci.data)

@raises(Errors.AstroDataError)
def test14():
    """ASTRODATA-insert TEST 14: Fail when XNAM XVER conflict"""
    ad1 = AstroData(sci123) #sci 3
    ad4 = AstroData(scivardq123) #sci3 var3 dq3
    ad4.insert(index=4, moredata=ad1)

def test15():
    """ASTRODATA-insert TEST 15: Build AD from scratch
    """
    ad3 = AstroData(mdfscivardq1) #mdf sci var dq
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad_new = AstroData()
    ad_new.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    ad_new.insert(index=0, moredata=ad3)
    ad3.info()
    print("ad_new.insert(index=0, moredata=ad3, auto_number=True)")
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad_new.info()
    eq_(ad_new[3].extname(), "MDF")
    eq_(ad_new[2].extname(), "SCI")
    eq_(ad_new[1].extname(), "VAR")
    eq_(ad_new[0].extname(), "DQ")
    eq_(ad_new[2].extver(), 1)
    eq_(ad_new[1].extver(), 1)
    eq_(ad_new[0].extver(), 1)

def test16():
    """ASTRODATA-insert TEST 16: AUTO NUMBER, MDF to empty AD
    """
    ad3 = AstroData(mdfscivardq1) #mdf sci var dq
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

def test17():
    """ASTRODATA-insert TEST 17: AUTO NUMBER, MDF into MEF
    """
    ad3 = AstroData(mdfscivardq1) #mdf sci var dq
    ad1 = AstroData(sci123) #sci 3
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








