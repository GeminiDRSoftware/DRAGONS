from copy import deepcopy

from astrodata.adutils.testutil import sci123, eq_, AstroData

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
