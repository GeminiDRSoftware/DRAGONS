import pyfits

from astrodata.adutils.testutil import eq_, sci1, AstroData

def test3():
    """ASTRODATA-inferred TEST 3: 'SCI' 1 if no XNAM found (SEF)"""
    testhdulist = pyfits.open(sci1)
    hlist = []
    alist = []
    for hdu in testhdulist[1:]:
        hlist.append(hdu.name)
    print("\n\n    HDUList EXTNAME: %s" % hlist)
    ad = AstroData(testhdulist) 
    alist.append((ad[0].extname(), ad[0].extver()))
    eq_(ad[0].extname(), 'SCI', msg="extname is not sci")
    eq_(ad[0].extver(), 1, msg="extver is incorrect")
    print("\n    AD (EXTNAME,EXTVER): %s" % alist)
    ad.close()
    testhdulist.close()
