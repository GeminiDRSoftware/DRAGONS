from astrodata.adutils.testutil import pyfits, sci1, AstroData, checkad

def test8():
    '''ASTRODATA-open TEST 8: Pass params phu, header, and data'''
    hdulist_sef = pyfits.open(sci1)
    phu = hdulist_sef[0]
    header = hdulist_sef[1].header
    data = hdulist_sef[1].data
    ad = AstroData(phu=phu, header=header, data=data)
    checkad(ad)
    hdulist_sef.close()

