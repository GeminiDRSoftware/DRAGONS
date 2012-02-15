from nose.tools import eq_

from astrodata.adutils.testutil import eq_, runinsert, sci123, scivardq123

def test4():
    """ASTRODATA-insert TEST 4: AUTO NUMBER, scivardq123 to sci123"""
    ad = runinsert(index=1, f1=sci123, f2=scivardq123, auto=True) 
    eq_(ad[9].extname(), "SCI")
    eq_(ad[8].extname(), "SCI")
    eq_(ad[7].extname(), "SCI")
    eq_(ad[6].extname(), "DQ")
    eq_(ad[5].extname(), "DQ")
    eq_(ad[4].extname(), "DQ")
    eq_(ad[3].extname(), "VAR")
    eq_(ad[2].extname(), "VAR")
    eq_(ad[1].extname(), "VAR")
    eq_(ad[9].extver(), 4)
    eq_(ad[8].extver(), 5)
    eq_(ad[7].extver(), 6)
    eq_(ad[6].extver(), 4)
    eq_(ad[5].extver(), 5)
    eq_(ad[4].extver(), 6)
    eq_(ad[3].extver(), 4)
    eq_(ad[2].extver(), 5)
    eq_(ad[1].extver(), 6)
