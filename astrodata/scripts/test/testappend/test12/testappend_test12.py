from astrodata.adutils.testutil \
    import Errors, raises, runappend, sci123, scivardq123 

@ raises(Errors.AstroDataError)
def test12():
    """ASTRODATA-append TEST 12: Fail scivardq123 to sci123"""
    ad = runappend(f1=sci123, f2=scivardq123) 

