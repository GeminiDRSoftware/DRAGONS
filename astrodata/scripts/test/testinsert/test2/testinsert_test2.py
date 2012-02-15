from astrodata.adutils.testutil \
    import runinsert, mdfscivardq1, sci123, eq_ 

def test2():
    """ASTRODATA-insert TEST 2: AUTO NUMBER, sci123 to mdfscivardq1"""
    ad = runinsert(index=0, f1=mdfscivardq1, f2=sci123, auto=True) 
    eq_(ad[2].extname(), "SCI")
    eq_(ad[1].extname(), "SCI")
    eq_(ad[0].extname(), "SCI")
    eq_(ad[2].extver(), 2)
    eq_(ad[1].extver(), 3)
    eq_(ad[0].extver(), 4)
