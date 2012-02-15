from astrodata.adutils.testutil \
    import raises, AstroData, Errors, sci123

@raises(Errors.AstroDataError)
def test1():
    '''ASTRODATA-mode TEST 1: Fail when try to overwrite readonly'''
    ad = AstroData(sci123)
    ad.write()

