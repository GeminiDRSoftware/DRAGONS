import sys
import os
from optparse import OptionParser
from copy import deepcopy

from nose.tools import *

import adtest_utils
from astrodata import AstroData

testfile = adtest_utils.testdatafile_1

def deepcopy_test_1():
    """deepcopy: test1 -Compare hdulist object ids between ad and deepcopy(ad).
    """
    print("\n\tTest input file: %s" % testfile)
    ad = AstroData(testfile)
    adDeepcopy = deepcopy(ad)
    adIdlist = []
    adDeepcopyIdlist = []
    for ext in ad:
        adIdlist.append(id(ext.hdulist[1]))
    for dext in adDeepcopy:
        adDeepcopyIdlist.append(id(dext.hdulist[1]))
    print "\t        ad hdulist ids:", adIdlist
    print "\tadDeepcopy hdulist ids:", adDeepcopyIdlist
    assert_not_equal(adIdlist, adDeepcopyIdlist, msg="hdulist ids are equal")
    ad.close()
    adDeepcopy.close()

def deepcopy_test_2():
    """deepcopy: test2 -Check that attribute change does not affect deepcopy.
    """
    print("\n\tTest input file: %s" % testfile)
    ad = AstroData(testfile)
    adDeepcopy = deepcopy(ad)
    print "\tad._AstroData__origFilename = "
    print "\t\t", ad._AstroData__origFilename
    savedFilename = ad._AstroData__origFilename
    ad._AstroData__origFilename = "newfilename.fits"
    print "\tad._AstroData__origFilename = ",ad._AstroData__origFilename
    print "\tadDeepcopy._AstroData__origFilename = "
    print "\t\t",adDeepcopy._AstroData__origFilename 
    eq_(adDeepcopy._AstroData__origFilename, savedFilename,
            msg="The attribute __origFilename has been altered in deepcopy")
    ad.close()
    adDeepcopy.close()
