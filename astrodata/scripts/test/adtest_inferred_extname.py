import sys, os
import pyfits
from astrodata import AstroData
from optparse import OptionParser

parser = OptionParser()
parser.set_description(
"""AstroData Test: Inferred Extension Naming.\n
Created by C.Allen, K.Dement, 01May2011.""" )

parser.add_option('--test1', action='store_true', dest='test1', default=False,
                   help='run test1:')
parser.add_option('--test2', action='store_true', dest='test2', default=False,
                   help='run test2:')
(options,  args) = parser.parse_args()

if not options.test1 and not options.test2:
    options.test1 = True
    options.test2 = True

print "\n", "_"*57,"\n\n\tAstroData TEST: Inferred Extension Naming\n"
if len(args) is 1:
    print "** Testdata: " + args[0]
    testhdulist = pyfits.open(args[0])
else:
    testdatafile = "../../../../test_data/recipedata/N20090703S0163.fits"
    print "** Testdata: " + testdatafile
    testhdulist = pyfits.open(testdatafile)
#print testhdulist.info()

if options.test1:
    print "TEST 1: Check that all extension names are inferred 'SCI'"
    alist = []
    blist = []
    for hdu in testhdulist[1:]:
        alist.append(hdu.name)
    print "\tList of ext. names of input:", repr(alist)
    ad = AstroData(testhdulist) 
    for ahdu in ad.hdulist[1:]:
        blist.append(ahdu.name)
    print "\tList of ext. names after AD instance:", repr(blist)
    ad.close()
    try:
        for a in alist:
            assert a  == ""
        for b in blist:
            assert str(b) == "SCI"
    except:
        print "\t>> FAILED: Inferred naming error.\n"
        raise AssertionError
    print "\t>> PASSED: All extension names are inferred.\n"

if options.test2:
    print "TEST 2: Check that extension names are NOT inferred 'SCI'"
    print "        when at least one input extension is named."
    alist = []
    blist = []
    if testhdulist[1].name == "SCI":
        for hdu2 in testhdulist[1:]:
            hdu2.name = ""
    # if test1 is run first and you do not rename it will not infer
    # even though it is all blank.
    testhdulist[1].name = "MDF"
    for hdu in testhdulist[1:]:
        alist.append(hdu.name)
    print "\tList of ext. names of input:", repr(alist)
    ad = AstroData(testhdulist) 
    for ahdu in ad.hdulist[1:]:
        blist.append(ahdu.name)
    print "\tList of ext. names after AD instance:", repr(blist)
    try:
        assert alist == blist
    except:
        print "\t>> FAILED: Extension names have changed.\n"
        raise AssertionError
    print "\t>> PASSED: All extension names are the same.\n"


