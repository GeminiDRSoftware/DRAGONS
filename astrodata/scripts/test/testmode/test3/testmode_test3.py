import os

from astrodata.adutils.testutil import assert_true, AstroData, sci123

def test3():
    '''ASTRODATA-mode TEST 3: Clobber unchanged readonly file'''
    ad = AstroData(sci123)
    ad.write(clobber=True)
    assert_true(os.path.isfile(sci123), 'Clobber fail')
    ad.close()

