import pyfits

from nose.tools import *

from astrodata import AstroData
import file_urls

mef_file = file_urls.testdatafile_1
sef_file = file_urls.testdatafile_1

def inferredsci(testhdulist):
    alist = []
    blist = []
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

def turnoff(testhdulist):
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

def inferred_extname_test1():
    """inferred_extname_test1 -raw ad MEF extname is inferred 'SCI'
    """
    testhdulist = pyfits.open(mef_file)
    inferredsci(testhdulist)

def inferred_extname_test2():
    """inferred_extname_test2 -ad MEF inferred off when input ext named
    """
    testhdulist = pyfits.open(mef_file)
    turnoff(testhdulist)

def inferred_extname_test3():
    """inferred_extname_test3 -raw ad single ext. extname is inferred 'SCI'
    """
    testhdulist = pyfits.open(sef_file)
    inferredsci(testhdulist)

def inferred_extname_test4():
    """inferred_extname_test4 -ad single ext. inferred off when input ext named
    """
    testhdulist = pyfits.open(sef_file)
    turnoff(testhdulist)
