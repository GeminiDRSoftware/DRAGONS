from nose.tools import *

import file_urls 
from astrodata import AstroData
from astrodata import Errors


def ad_insert_test1():
    """ad_insert_test1 -moredata=AD, with auto_number
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad3 = AstroData(file_urls.testdatafile_3) #mdf sci var dq
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    ad3.info()
    ad3.hdulist.info()
    ad1.insert(index=1, moredata=ad3, auto_number=True)
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.hdulist.info()
    ad1.info()
    eq_(ad1[4].extname(), "MDF")
    eq_(ad1[3].extname(), "SCI")
    eq_(ad1[2].extname(), "VAR")
    eq_(ad1[1].extname(), "DQ")
    eq_(ad1[3].extver(), 4)
    eq_(ad1[2].extver(), 4)
    eq_(ad1[1].extver(), 4)

def ad_insert_test2():
    """ad_insert_test2 -moredata=AD, with auto_number
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad3 = AstroData(file_urls.testdatafile_3) #mdf sci var dq
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad3.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    ad1.info()
    ad3.insert(index=0, moredata=ad1, auto_number=True)
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad3.info()
    eq_(ad3[2].extname(), "SCI")
    eq_(ad3[1].extname(), "SCI")
    eq_(ad3[0].extname(), "SCI")
    eq_(ad3[2].extver(), 2)
    eq_(ad3[1].extver(), 3)
    eq_(ad3[0].extver(), 4)

def ad_insert_test3():
    """ad_insert_test3 -moredata=AD, with auto_number
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad4.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    ad1.info()
    ad4.insert(index=4, moredata=ad1, auto_number=True)
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad4.info()
    eq_(ad4[6].extname(), "SCI")
    eq_(ad4[5].extname(), "SCI")
    eq_(ad4[4].extname(), "SCI")
    eq_(ad4[6].extver(), 4)
    eq_(ad4[5].extver(), 5)
    eq_(ad4[4].extver(), 6)


def ad_insert_test4():
    """ad_insert_test4 -moredata=AD, with auto_number
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    ad4.info()
    ad1.insert(index=1, moredata=ad4, auto_number=True)
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[9].extname(), "SCI")
    eq_(ad1[8].extname(), "SCI")
    eq_(ad1[7].extname(), "SCI")
    eq_(ad1[6].extname(), "DQ")
    eq_(ad1[5].extname(), "DQ")
    eq_(ad1[4].extname(), "DQ")
    eq_(ad1[3].extname(), "VAR")
    eq_(ad1[2].extname(), "VAR")
    eq_(ad1[1].extname(), "VAR")
    eq_(ad1[9].extver(), 4)
    eq_(ad1[8].extver(), 5)
    eq_(ad1[7].extver(), 6)
    eq_(ad1[6].extver(), 4)
    eq_(ad1[5].extver(), 5)
    eq_(ad1[4].extver(), 6)
    eq_(ad1[3].extver(), 4)
    eq_(ad1[2].extver(), 5)
    eq_(ad1[1].extver(), 6)

def ad_insert_test5():
    """ad_insert_test5 -moredata=AD, with auto_number
    """
    ad3 = AstroData(file_urls.testdatafile_3) #mdf sci var dq
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad3.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    ad4.info()
    ad3.insert(index=1, moredata=ad4, auto_number=True)
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad3.info()
    eq_(ad3[9].extname(), "SCI")
    eq_(ad3[8].extname(), "SCI")
    eq_(ad3[7].extname(), "SCI")
    eq_(ad3[6].extname(), "DQ")
    eq_(ad3[5].extname(), "DQ")
    eq_(ad3[4].extname(), "DQ")
    eq_(ad3[3].extname(), "VAR")
    eq_(ad3[2].extname(), "VAR")
    eq_(ad3[1].extname(), "VAR")
    eq_(ad3[9].extver(), 2)
    eq_(ad3[8].extver(), 3)
    eq_(ad3[7].extver(), 4)
    eq_(ad3[6].extver(), 2)
    eq_(ad3[5].extver(), 3)
    eq_(ad3[4].extver(), 4)
    eq_(ad3[3].extver(), 2)
    eq_(ad3[2].extver(), 3)
    eq_(ad3[1].extver(), 4)

@raises(Errors.AstroDataError)
def ad_insert_test6():
    """ad_insert_test6 -moredata, index too high, raise
    """
    ad3 = AstroData(file_urls.testdatafile_3) #mdf sci var dq
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<dd<<"
    ad4.info()
    ad4.insert(index=11, moredata=ad3, auto_number=True)
    print("ad4.insert(index=11, moredata=ad3, auto_number=True)")

def ad_insert_test7():
    """ad_insert_test7 -auto_number, add sci1 to existing sci1
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.insert(index=1, header=adsci.header, data=adsci.data, auto_number=True)
    print "ad1.insert(index=1, header=adsci.header, data=adsci.data, auto_number=True)"
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[1].hdulist[1].name, "SCI")
    eq_(ad1[1].extver(), 4)

def ad_insert_test8():
    """ad_insert_test8 -auto_number, insert var
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['VAR', 2]
    print("adsci = ad4['VAR', 2]")
    ad1.insert(index=0, header=adsci.header, data=adsci.data, auto_number=True)
    print "ad1.insert(index=0,header=adsci.header, data=adsci.data, auto_number=True)"
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[0].extname(), "VAR")
    eq_(ad1[0].extver(), 4)

def ad_insert_test9():
    """ad_insert_test9 -auto_number, insert high var
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.hdulist.__delitem__(3)
    print("ad1.hdulist.__delitem__(3)")
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['VAR', 3]
    print("adsci = ad4['VAR', 3]")
    print "ad1.insert(index=2, header=adsci.header, data=adsci.data, auto_number=True)"
    ad1.insert(index=2, header=adsci.header, data=adsci.data, auto_number=True, extver=5)
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[2].extname(), "VAR")
    eq_(ad1[2].extver(), 5)

@ raises(Errors.AstroDataError)
def ad_insert_test10():
    """ad_insert_test10 -auto_number, raise when no header
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    print "ad1.insert(index=0, data=adsci.data, auto_number=True)"
    ad1.insert(index=0, data=adsci.data, auto_number=True)

def ad_insert_test11():
    """ad_insert_test11 -auto_number, insert with high sci extver
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.insert(index=1, extver=10, header=adsci.header, data=adsci.data, \
        auto_number=True)
    print("ad1.insert(index=1, extver=10, header=adsci.header, \
        data=adsci.data, auto_number=True)")
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()
    eq_(ad1[1].extname(), "SCI")
    eq_(ad1[1].extver(), 10)

@ raises(TypeError)
def ad_insert_test12():
    """ad_insert_test12 -no index, raise TypeError
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    ad1.insert(moredata=ad4)

@ raises(Errors.AstroDataError)
def ad_insert_test13():
    """ad_insert_test13 -no auto_number, raise when conflict
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    adsci = ad4['SCI', 1]
    print("adsci = ad4['SCI', 1]")
    ad1.insert(index=0, header=adsci.header, data=adsci.data)
    print("ad1.insert(index-o, extver=10, header=adsci.header, data=adsci.data,\
          auto_number=True)")
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad1.info()

@ raises(Errors.AstroDataError)
def ad_insert_test14():
    """ad_insert_test14 -moredata No auto_number, throw exception
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
    ad4.insert(index=4, moredata=ad1)

def ad_insert_test15():
    """ad_insert_test15 -moredata,  insert into blank ad
    """
    ad3 = AstroData(file_urls.testdatafile_3) #mdf sci var dq
    ad4 = AstroData(file_urls.testdatafile_4) #sci3 var3 dq3
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
