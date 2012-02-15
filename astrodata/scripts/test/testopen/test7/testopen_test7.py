from astrodata.adutils.testutil import sci1, AstroData, checkad, pyfits 

def test7():
    '''ASTRODATA-open TEST 7: Pass param phu=pyfits header'''
    hdulist_sef = pyfits.open(sci1)
    ad = AstroData(phu=hdulist_sef[0].header)
    checkad(ad)
    hdulist_sef.close()
