from copy import deepcopy

from nose.tools import assert_not_equal, eq_

from file_urls import sci123, sci1
from astrodata import AstroData

def test1():
    """ASTRODATA-deepcopy TEST 1: Pyfits HDUList id's are not equal (MEF)"""
    ad = AstroData(sci123)
    adDeepcopy = deepcopy(ad)
    adIdlist = []
    adDeepcopyIdlist = []
    for ext in ad:
        adIdlist.append(id(ext.hdulist[1]))
    for dext in adDeepcopy:
        adDeepcopyIdlist.append(id(dext.hdulist[1]))
    assert_not_equal(adIdlist, adDeepcopyIdlist, msg="hdulist ids are equal")
    ad.close()
    adDeepcopy.close()

def test2():
    """
    ASTRODATA-deepcopy TEST 2: Orig attribute change does not affect copy (MEF)
    """
    ad = AstroData(sci123)
    adDeepcopy = deepcopy(ad)
    savedFilename = ad._AstroData__origFilename
    ad._AstroData__origFilename = "newfilename.fits"
    eq_(adDeepcopy._AstroData__origFilename, savedFilename,
            msg="The attribute _AstroData__origFilename has been altered in deepcopy")
    ad.close()
    adDeepcopy.close()

def test3():
    """ASTRODATA-deepcopy TEST 3: Pyfits HDUList id's are not equal (SEF)"""
    ad = AstroData(sci1)
    adDeepcopy = deepcopy(ad)
    adIdlist = []
    adDeepcopyIdlist = []
    for ext in ad:
        adIdlist.append(id(ext.hdulist[1]))
    for dext in adDeepcopy:
        adDeepcopyIdlist.append(id(dext.hdulist[1]))
    assert_not_equal(adIdlist, adDeepcopyIdlist, msg="hdulist ids are equal")
    ad.close()
    adDeepcopy.close()

def test4():
    """
    ASTRODATA-deepcopy TEST 4: Orig attribute change does not affect copy (SEF)
    """
    ad = AstroData(sci1)
    adDeepcopy = deepcopy(ad)
    savedFilename = ad._AstroData__origFilename
    ad.mode = 'update'
    ad._AstroData__origFilename = "newfilename.fits"
    eq_(adDeepcopy.mode, 'readonly',
            msg="Deepcopy Failure, mode is not readonly")
    ad.close()
    adDeepcopy.close()

