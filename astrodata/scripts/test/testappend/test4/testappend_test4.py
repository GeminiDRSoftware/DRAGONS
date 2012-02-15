from astrodata.adutils.testutil import runappend, sci123, scivardq123, eq_ 

def test4():
    """ASTRODATA-append TEST 4: AUTO NUMBER, scivardq123 to sci123"""
    ad = runappend(f1=sci123, f2=scivardq123, auto=True) 
    eq_(ad[3].extname(), "SCI")
    eq_(ad[4].extname(), "SCI")
    eq_(ad[5].extname(), "SCI")
    eq_(ad[6].extname(), "DQ")
    eq_(ad[7].extname(), "DQ")
    eq_(ad[8].extname(), "DQ")
    eq_(ad[9].extname(), "VAR")
    eq_(ad[10].extname(), "VAR")
    eq_(ad[11].extname(), "VAR")
    eq_(ad[3].extver(), 4)
    eq_(ad[4].extver(), 5)
    eq_(ad[5].extver(), 6)
    eq_(ad[6].extver(), 4)
    eq_(ad[7].extver(), 5)
    eq_(ad[8].extver(), 6)
    eq_(ad[9].extver(), 4)
    eq_(ad[10].extver(), 5)
    eq_(ad[11].extver(), 6)

