from astrodata.adutils.testutil import sci1, pyfits, AstroData, checkad

def test9():
    '''ASTRODATA-open TEST 9: Pass params header and data'''
    hdulist_sef = pyfits.open(sci1)
    header = hdulist_sef[1].header
    data = hdulist_sef[1].data
    ad = AstroData(header=header, data=data)
    checkad(ad)
    hdulist_sef.close()



     
