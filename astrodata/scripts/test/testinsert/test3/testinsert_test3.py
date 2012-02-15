from astrodata.adutils.testutil import runinsert, scivardq123, sci123, eq_ 

def test3():
    """ASTRODATA-insert TEST 3: AUTO NUMBER, sci123 to scivardq123"""
    ad = runinsert(index=4, f1=scivardq123, f2=sci123, auto=True) 
    eq_(ad[6].extname(), "SCI")
    eq_(ad[5].extname(), "SCI")
    eq_(ad[4].extname(), "SCI")
    eq_(ad[6].extver(), 4)
    eq_(ad[5].extver(), 5)
    eq_(ad[4].extver(), 6)

