from copy import deepcopy

from astrodata.adutils.testutil \
    import sci123, AstroData, assert_not_equal

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

