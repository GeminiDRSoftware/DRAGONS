import pyfits

from astrodata.adutils.testutil import sci123, eq_, AstroData

def test1():
    """ASTRODATA-inferred TEST 1: 'SCI',<VER> for all exts if no XNAMs found (MEF)"""
    testhdulist = pyfits.open(sci123)
    
    # make it so no XNAM found
    for hdu in testhdulist[1:]:
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
        eq_(ad[i].extname(), 'SCI', msg="extname is not sci")
        eq_(ad[i].extver(), i + 1, msg="extver is incorrect")
    print("\n    AD (EXTNAME,EXTVER): %s" % alist)
    ad.close()
    testhdulist.close()

