import os
import shutil

from astrodata.adutils.testutil import AstroData, ok_, sci123

def test5():
    '''ASTRODATA-mode TEST 5: Clobber changed readonly file'''
    outfile = 'testmode5.fits'
    if os.path.exists(outfile):
        os.remove(outfile)
    shutil.copy(sci123, outfile)
    ad = AstroData(outfile, mode='update')
    ad.hdulist[0].header['INSTRUME'] = 'GMOS-S'
    ad.write(clobber=True)
    ad.close()
    ad = AstroData(outfile)
    ok_(ad.hdulist[0].header['INSTRUME'] == 'GMOS-S', msg='Keyword is different')
    os.remove(outfile)
    ad.close()

