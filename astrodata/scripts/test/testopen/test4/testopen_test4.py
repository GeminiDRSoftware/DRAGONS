from astrodata.adutils.testutil import sci1, AstroData, pyfits, checkad

def test4():
    '''ASTRODATA-open TEST 4: Pass pyfits HDUList (SEF, SCI1)'''
    hdulist_sef = pyfits.open(sci1)
    ad = AstroData(hdulist_sef)
    checkad(ad)
    hdulist_sef.close()
