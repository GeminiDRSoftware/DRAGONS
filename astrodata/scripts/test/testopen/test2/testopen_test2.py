from astrodata.adutils.testutil import sci1, AstroData, checkad

def test2():
    '''ASTRODATA-open TEST 2: Pass filename (SEF, SCI1)'''
    ad = AstroData(sci1)
    checkad(ad)

