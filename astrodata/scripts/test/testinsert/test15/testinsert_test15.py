from astrodata.adutils.testutil \
    import mdfscivardq1, AstroData, eq_ 

def test15():
    """ASTRODATA-insert TEST 15: Build AD from scratch
    """
    ad3 = AstroData(mdfscivardq1) 
    print "\n             >>>>>>>     AD HOST    <<<<<<<<"
    ad_new = AstroData()
    ad_new.info()
    print "\n             >>>>>>>    AD insert   <<<<<<<<"
    ad_new.insert(index=0, moredata=ad3)
    ad3.info()
    print("ad_new.insert(index=0, moredata=ad3, auto_number=True)")
    print "\n             >>>>>>>  AD HOST (NEW) <<<<<<<<"
    ad_new.info()
    eq_(ad_new[3].extname(), "MDF")
    eq_(ad_new[2].extname(), "SCI")
    eq_(ad_new[1].extname(), "VAR")
    eq_(ad_new[0].extname(), "DQ")
    eq_(ad_new[2].extver(), 1)
    eq_(ad_new[1].extver(), 1)
    eq_(ad_new[0].extver(), 1)
