import pyfits

from astrodata.adutils.testutil import sci123, eq_, AstroData

def test2():
    """ASTRODATA-inferred TEST 2: Do not infer when XNAM found (MEF)"""
    testhdulist = pyfits.open(sci123)
    for hdu in testhdulist[2:]:
        hdu.name = ""
        hdu.header.rename_key("extname","xname" )
    hlist = []
    alist = [] 
    for hdu in testhdulist[1:]:
        hlist.append(hdu.name)
    print("\n\n    HDUList EXTNAME: %s" % hlist)
    ad = AstroData(testhdulist) 
    for i in range(0,3):
        alist.append((ad[i].extname(), ad[i].extver()))
    print("\n    AD (EXTNAME,EXTVER): %s" % alist)
    testhdulist.close()
    eq_(ad[0].extname(), 'SCI', msg="extname is not sci")
    eq_(ad[1].extname(), None, msg="extname is not sci")
    eq_(ad[2].extname(), None, msg="extname is not sci")
    eq_(ad[0].extver(), 1, msg="extver is incorrect")
    eq_(ad[1].extver(), 2, msg="extver is incorrect")
    eq_(ad[2].extver(), 3, msg="extver is incorrect")

