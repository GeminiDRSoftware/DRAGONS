from astrodata.adutils.testutil import eq_, runinsert, sci123, mdfscivardq1 

def test1():
    """ASTRODATA-insert TEST 1: AUTO NUMBER, mdfscivardq1 to sci123"""
    ad = runinsert(index=1, f1=sci123, f2=mdfscivardq1, auto=True) 
    eq_(ad[4].extname(), "MDF")
    eq_(ad[3].extname(), "SCI")
    eq_(ad[2].extname(), "VAR")
    eq_(ad[1].extname(), "DQ")
    eq_(ad[3].extver(), 4)
    eq_(ad[2].extver(), 4)
    eq_(ad[1].extver(), 4)

