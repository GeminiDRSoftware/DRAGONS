from copy import deepcopy

from nose.tools import *

import file_urls 
from astrodata import AstroData

mef_file = file_urls.testdatafile_1
sef_file = file_urls.testdatafile_1

def deepcopy_test_1():
    """deepcopy_test1 -MEF ad and deepcopy(ad) hdulist are different
    """
    print("\n\tTest input file: %s" % mef_file)
    ad = AstroData(mef_file)
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
    """deepcopy_test2 -MEF attribute change does not affect deepcopy
    """
    print("\n\tTest input file: %s" % mef_file)
    ad = AstroData(mef_file)
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

def deepcopy_test_3():
    """deepcopy_test3 -single ext ad and deepcopy(ad) hdulist are different
    """
    print("\n\tTest input file: %s" % sef_file)
    ad = AstroData(sef_file)
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

def deepcopy_test_4():
    """deepcopy_test4 -single ext ad attribute change does not affect deepcopy
    """
    print("\n\tTest input file: %s" % sef_file)
    ad = AstroData(sef_file)
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

