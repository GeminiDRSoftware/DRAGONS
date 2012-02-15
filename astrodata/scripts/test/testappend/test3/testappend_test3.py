from astrodata.adutils.testutil import *

def test3():
    """ASTRODATA-append TEST 3: AUTO NUMBER, sci123 to scivardq123"""
    ad = runappend(f1=scivardq123, f2=sci123, auto=True) 
    eq_(ad[9].extname(), "SCI")
    eq_(ad[10].extname(), "SCI")
    eq_(ad[11].extname(), "SCI")
    eq_(ad[9].extver(), 4)
    eq_(ad[10].extver(), 5)
    eq_(ad[11].extver(), 6)

