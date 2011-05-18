import sys
import os

import pyfits
import astrodata
from astrodata import AstroData
import adtest_utils

testfile = adtest_utils.testdatafile_1

def checkad(ad):
    assert ad.mode != None
    print("\tad.mode is set to %s" % ad.mode)
    assert type(ad.phu) is pyfits.core.PrimaryHDU
    print("\tad.phu type is pyfits.core.PrimaryHDU")
    for ext in ad:
        assert type(ext) is astrodata.AstroData
        print("\tad[%s,%s] type is astrodata.AstroData" % (ext.extname(), ext.extver()))
    ad.close()

def constructor_test1():
    """constructor: test1 -Create AstroData instance using filename.
    """
    print("\n\tTest input file: %s" % testfile)
    ad = AstroData(testfile)
    print("\tad = AstroData(%s)" % ad.filename)
    checkad(ad)

def constructor_test2():
    """constructor: test2 -Create AstroData instance using hdulist.
    """
    print("\n\tTest input file: %s" % testfile)
    hdulist = pyfits.open(testfile)
    ad = AstroData(hdulist)
    print("\tad = AstroData(hdulist)")
    checkad(ad)
    hdulist.close()

def constructor_test3():
    """constructor: test3 -Create AstroData instance using phu and data.
    """
    print("\n\tTest input file: %s" % testfile)
    hdulist = pyfits.open(testfile)
    ad = AstroData(header=hdulist[0], data=[hdulist[1],hdulist[2],hdulist[3]])
    print("\tad = AstroData(hdulist)")
    checkad(ad)
    hdulist.close()




     
