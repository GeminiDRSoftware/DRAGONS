import sys
import astrodata
import pyfits
import numpy
from astrodata import AstroData, Errors
from nose.tools import assert_true, eq_, raises, ok_, assert_not_equal

prepath = ''
for path in sys.path:
    if 'gemini_python' in path and 'rtftests' not in path:
        splist = path.split('gemini_python')
        prepath += splist[0]
        break
sci123 = prepath + "gemini_python/test_data/infrastructure/N20090703S0163.fits" 
sci1 = prepath + "gemini_python/test_data/infrastructure/S20100122S0063.fits" 
mdfscivardq1 = prepath + \
    "gemini_python/test_data/infrastructure/nS20040613S0162.fits" 
scivardq123 = prepath + \
    "gemini_python/test_data/infrastructure/N20110313S0188_varAdded.fits"


def runappend(f1=None, f2=None, auto=False):
    ad = AstroData(f1)
    md = AstroData(f2)
    pstr = "\n\n             >>>>>>>     AD     <<<<<<<<\n"
    pstr += str(ad.infostr())
    pstr += "\n\n             >>>>>>>    AD APPEND   <<<<<<<<\n"
    pstr += str(md.infostr())
    ad.append(moredata=md, auto_number=auto)
    pstr +="\n\n             >>>>>>>  NEW AD <<<<<<<<\n"
    pstr += str(ad.infostr())
    print(pstr)
    return ad

def runinsert(index=None, f1=None, f2=None, auto=False):
    ad = AstroData(f1)
    md = AstroData(f2)
    pstr = "\n\n             >>>>>>>     AD     <<<<<<<<\n"
    pstr += str(ad.infostr())
    pstr += "\n\n             >>>>>>>    AD APPEND   <<<<<<<<\n"
    pstr += str(md.infostr())
    ad.insert(index=index, moredata=md, auto_number=auto)
    pstr +="\n\n             >>>>>>>  NEW AD <<<<<<<<\n"
    pstr += str(ad.infostr())
    print(pstr)
    return ad

def checkad(ad):
    #check mode
    ok_(ad.mode != None, msg='mode is None')
    
    #check type subdata
    for i in range(len(ad)):
        eq_(type(ad[i]),astrodata.AstroData)
    
    #check AstroData subdata 
    for i in range(len(ad)):
        exn = ad[i].extname()
        exv = ad[i].extver()
        eq_(id(ad[i]), id(ad[exn, exv]), msg='object ids are different')
    
    #check phu type
    eq_(type(ad.phu),pyfits.core.PrimaryHDU)
    
    #check phu propagation
    checkid = id(ad.phu)
    eq_(id(ad.phu),id(ad.hdulist[0]), msg='objects ids are different')
    for i in range(len(ad)):
        eq_(checkid, id(ad[i].hdulist[0]), msg='object ids are different')

    #check phu.header propagation
    checkid = id(ad.phu.header)
    eq_(id(ad.phu.header),id(ad.hdulist[0].header),\
        msg='objects ids are different')
    for i in range(len(ad)):
        eq_(checkid, id(ad[i].hdulist[0].header), \
            msg='object ids are different')
    
    #check imageHdu propagation
    for i in range(len(ad)):
        idhdu1 = id(ad.hdulist[i+1].data)
        idhdu2 = id(ad[i].hdulist[1].data)
        eq_(idhdu1, idhdu2, msg='object ids are different')
    
    ad.close()
