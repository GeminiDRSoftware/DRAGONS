from astrodata.adutils.testutil import AstroData, sci123, checkad

def test1():
    '''ASTRODATA-open TEST 1: Pass filename (MEF, SCI123)'''
    ad = AstroData(sci123)
    checkad(ad)
