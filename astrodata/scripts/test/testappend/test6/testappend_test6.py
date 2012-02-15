from astrodata.adutils.testutil import runappend, scivardq123, mdfscivardq1, eq_ 

def test6():
    """ASTRODATA-append TEST 6: AUTO NUMBER, mdfscivardq1 to scivardq123"""
    ad = runappend(f1=scivardq123, f2=mdfscivardq1, auto=True) 
    eq_(ad[9].extname(), "MDF")
    eq_(ad[10].extname(), "SCI")
    eq_(ad[11].extname(), "VAR")
    eq_(ad[12].extname(), "DQ")
    eq_(ad[10].extver(), 4)
    eq_(ad[11].extver(), 4)
    eq_(ad[12].extver(), 4)

