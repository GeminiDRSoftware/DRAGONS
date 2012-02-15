from copy import deepcopy

from astrodata.adutils.testutil import assert_not_equal, sci1, AstroData

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

