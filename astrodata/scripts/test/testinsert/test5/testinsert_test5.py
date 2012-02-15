from astrodata.adutils.testutil \
    import eq_, mdfscivardq1, scivardq123, runinsert 

def test5():
    """ASTRODATA-insert TEST 5: AUTO NUMBER, scivardq123 to mdfscivardq1"""
    ad = runinsert(index=1, f1=mdfscivardq1, f2=scivardq123, auto=True) 
    eq_(ad[9].extname(), "SCI")
    eq_(ad[8].extname(), "SCI")
    eq_(ad[7].extname(), "SCI")
    eq_(ad[6].extname(), "DQ")
    eq_(ad[5].extname(), "DQ")
    eq_(ad[4].extname(), "DQ")
    eq_(ad[3].extname(), "VAR")
    eq_(ad[2].extname(), "VAR")
    eq_(ad[1].extname(), "VAR")
    eq_(ad[9].extver(), 2)
    eq_(ad[8].extver(), 3)
    eq_(ad[7].extver(), 4)
    eq_(ad[6].extver(), 2)
    eq_(ad[5].extver(), 3)
    eq_(ad[4].extver(), 4)
    eq_(ad[3].extver(), 2)
    eq_(ad[2].extver(), 3)
    eq_(ad[1].extver(), 4)
