from astrodata.adutils.testutil import runappend, mdfscivardq1, scivardq123, eq_ 

def test5():
    """ASTRODATA-append TEST 5: AUTO NUMBER, scivardq123 to mdfscivardq1"""
    ad = runappend(f1=mdfscivardq1, f2=scivardq123, auto=True) 
    eq_(ad[4].extname(), "SCI")
    eq_(ad[5].extname(), "SCI")
    eq_(ad[6].extname(), "SCI")
    eq_(ad[7].extname(), "DQ")
    eq_(ad[8].extname(), "DQ")
    eq_(ad[9].extname(), "DQ")
    eq_(ad[10].extname(), "VAR")
    eq_(ad[11].extname(), "VAR")
    eq_(ad[12].extname(), "VAR")
    eq_(ad[4].extver(), 2)
    eq_(ad[5].extver(), 3)
    eq_(ad[6].extver(), 4)
    eq_(ad[7].extver(), 2)
    eq_(ad[8].extver(), 3)
    eq_(ad[9].extver(), 4)
    eq_(ad[10].extver(), 2)
    eq_(ad[11].extver(), 3)
    eq_(ad[12].extver(), 4)

