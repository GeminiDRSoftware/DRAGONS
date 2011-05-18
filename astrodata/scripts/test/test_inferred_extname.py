import sys
import os
import pyfits

from nose.tools import *

from astrodata import AstroData
import adtest_utils

testfile = adtest_utils.testdatafile_1

def inferred_extname_test1():
    """inferred_extname: test1 -Check extension names are inferred 'SCI'.
    """
    alist = []
    blist = []
    testhdulist = pyfits.open(testfile)
    for hdu in testhdulist[1:]:
        alist.append(hdu.name)
    print "\n\tList of ext. names of input:", repr(alist)
    ad = AstroData(testhdulist) 
    for ahdu in ad.hdulist[1:]:
        blist.append(ahdu.name)
    print "\tList of ext. names after AD instance:", repr(blist)
    for a in alist:
        eq_(a, "", msg="Input hdu names have been assigned")
    for b in blist:
        eq_(str(b), "SCI", msg="Inferred name is not 'SCI'")
    ad.close()
    testhdulist.close()
    alist = []
    blist = []

def inferred_extname_test2():
    """inferred_extname: test2 -Inferred naming off when input ext named.
    """
    testhdulist = pyfits.open(testfile)
    alist = []
    blist = []
    if testhdulist[1].name == "SCI":
        for hdu2 in testhdulist[1:]:
            hdu2.name = ""
    testhdulist[1].name = "SCI"
    for hdu in testhdulist[1:]:
        alist.append(hdu.name)
    print "\n\tList of ext. names of input:", repr(alist)
    ad = AstroData(testhdulist) 
    for ahdu in ad.hdulist[1:]:
        blist.append(ahdu.name)
    print "\tList of ext. names after AD instance:", repr(blist)
    eq_(alist, blist, msg="Inferred naming is on")


