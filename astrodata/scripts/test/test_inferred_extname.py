import pyfits

from nose.tools import eq_

from astrodata import AstroData
from file_urls import sci123, sci1



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
