from astrodata.adutils.testutil import * 

def test1():
    """ASTRODATA-append TEST 1: AUTO NUMBER, mdfscivardq1 to sci123"""
    ad = runappend(f1=sci123, f2=mdfscivardq1, auto=True) 
    eq_(ad[3].extname(), "MDF")
    eq_(ad[4].extname(), "SCI")
    eq_(ad[5].extname(), "VAR")
    eq_(ad[6].extname(), "DQ")
    eq_(ad[4].extver(), 4)
    eq_(ad[5].extver(), 4)
    eq_(ad[6].extver(), 4)

