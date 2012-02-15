from copy import deepcopy

from astrodata.adutils.testutil import  eq_, sci1, AstroData

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

