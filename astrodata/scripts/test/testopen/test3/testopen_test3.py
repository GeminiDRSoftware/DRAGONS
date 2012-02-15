from astrodata.adutils.testutil import sci123, pyfits, AstroData, checkad

def test3():
    '''ASTRODATA-open TEST 3: Pass pyfits HDUList (MEF, SCI123)'''
    hdulist_mef = pyfits.open(sci123)
    ad = AstroData(hdulist_mef)
    checkad(ad)
    hdulist_mef.close()
