import os

from astrodata.adutils.testutil import AstroData, assert_true, sci123

def test2():
    '''ASTRODATA-mode TEST 2: Overwrite readonly AD using name change'''
    ad = AstroData(sci123)
    outfile = 'testmode2.fits'
    if os.path.exists(outfile):
        os.remove(outfile)
    ad.filename = outfile
    ad.write()
    assert_true(os.path.isfile(outfile), 'ad.write() FAIL')
    os.remove(outfile)
    ad.close()

