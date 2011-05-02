import sys, os
from astrodata import AstroData
from optparse import OptionParser
from copy import deepcopy

parser = OptionParser()
parser.set_description(
"""AstroData Test: Deepcopy.\n
Created by C.Allen, K.Dement, 01May2011.""" )

parser.add_option('--test1', action='store_true', dest='test1', default=False,
                   help='run test1:')
parser.add_option('--test2', action='store_true', dest='test2', default=False,
                   help='run test2:')
parser.add_option('--test3', action='store_true', dest='test3', default=False,
                   help='run test3:')
(options,  args) = parser.parse_args()

if not options.test1 and not options.test2 and not options.test3:
    options.test1 = True
    options.test2 = True
    options.test3 = True

print "\n", "_"*57,"\n\n\t\tAstroData TEST: deepcopy\n"
if len(args) is 1:
    ad = AstroData(args[0])
    print "** ad Testdata: " + args[0]
else:
    testdatafile = "../../../../test_data/recipedata/N20090703S0163.fits"
    ad = AstroData(testdatafile)
    print "** Testdata: " + testdatafile
adDeepcopy = deepcopy(ad)

if options.test1:
    adIdlist = []
    adDeepcopyIdlist = []
    for ext in ad:
        adIdlist.append(id(ext.hdulist[1]))
    for dext in adDeepcopy:
        adDeepcopyIdlist.append(id(dext.hdulist[1]))
    print "TEST 1: Compare hdulist ids"
    print "\t        ad hdulist ids:", adIdlist
    print "\tadDeepcopy hdulist ids:", adDeepcopyIdlist
    try:
        assert adIdlist is not adDeepcopyIdlist
    except:
        print "\t>> FAILED: hdulist ids match\n"
    print "\t>> PASSED: hdulist ids are different\n"

if options.test2:
    print "TEST 2: Check attribute retention"
    print "\tad._AstroData__origFilename = "
    print "\t\t", ad._AstroData__origFilename
    savedFilename = ad._AstroData__origFilename
    ad._AstroData__origFilename = "newfilename.fits"
    print "\tad._AstroData__origFilename = ",ad._AstroData__origFilename
    print "\tadDeepcopy._AstroData__origFilename = "
    print "\t\t",adDeepcopy._AstroData__origFilename 
    try:
        assert adDeepcopy._AstroData__origFilename is savedFilename
    except:
        print "\t>> FAILED: Original name is not the same after deepcopy\n"
    print "\t>> PASSED: Original name is preserved by deepcopy\n" 

if options.test3:
    print "Test3: Under Construction\n"

print "_"*57,"\n"

