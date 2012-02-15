from astrodata.adutils.testutil import sci1, pyfits, AstroData, checkad

def test6():
    '''ASTRODATA-open TEST 6: Pass param phu=pyfits PHU'''
    hdulist_sef = pyfits.open(sci1)
    ad = AstroData(phu=hdulist_sef[0])
    checkad(ad)
    hdulist_sef.close()
