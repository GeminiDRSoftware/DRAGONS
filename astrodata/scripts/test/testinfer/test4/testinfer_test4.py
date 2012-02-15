import pyfits

from astrodata.adutils.testutil import eq_, sci1, AstroData

def test4():
    """ASTRODATA-inferred TEST 4: Do not infer when XNAM found (SEF)
    """
    testhdulist = pyfits.open(sci1)
    testhdulist[1].name = "MARS"
    hlist = []
    alist = [] 
    for hdu in testhdulist[1:]:
        hlist.append(hdu.name)
    print("\n\n    HDUList EXTNAME: %s" % hlist)
    ad = AstroData(testhdulist) 
    alist.append((ad[0].extname(), ad[0].extver()))
    print("\n    AD (EXTNAME,EXTVER): %s" % alist)
    testhdulist.close()
    eq_(ad[0].extname(), 'MARS', msg="extname is not sci")
    eq_(ad[0].extver(), 1, msg="extver is incorrect")
