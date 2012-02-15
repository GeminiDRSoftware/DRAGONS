from astrodata.adutils.testutil \
    import runinsert, raises, Errors, scivardq123, mdfscivardq1

@raises(Errors.AstroDataError)
def test6():
    """ASTRODATA-insert TEST 6: Fail when index out of range"""
    ad = runinsert(index=11, f1=scivardq123, f2=mdfscivardq1, auto=True) 

