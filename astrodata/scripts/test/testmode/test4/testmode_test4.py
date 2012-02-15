from astrodata.adutils.testutil import raises, AstroData, Errors, sci123

@raises(Errors.AstroDataError)
def test4():
    '''ASTRODATA-mode TEST 4: Fail when try to update readonly file
    '''
    ad = AstroData(sci123, mode='update')
    ad.hdulist[0].header['INSTRUME'] = 'GMOS-S'
    ad.write()

