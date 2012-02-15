from astrodata.adutils.testutil import *

def test2():
    """ASTRODATA-append TEST 2: AUTO NUMBER, sci123 to mdfscivardq1"""
    ad = runappend(f1=mdfscivardq1, f2=sci123, auto=True) 
    eq_(ad[4].extname(), "SCI")
    eq_(ad[5].extname(), "SCI")
    eq_(ad[6].extname(), "SCI")
    eq_(ad[4].extver(), 2)
    eq_(ad[5].extver(), 3)
    eq_(ad[6].extver(), 4)

