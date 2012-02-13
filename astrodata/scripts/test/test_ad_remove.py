from nose.tools import *

from file_urls import sci123  
from astrodata import AstroData
from astrodata import Errors

def test1():
    """ASTRODATA-remove TEST 1: Use index"""
    ad1 = AstroData(sci123) 
    print "\n             >>>>>>>     AD       <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD remove   <<<<<<<<"
    ad1.remove(1)
    print("ad1.remove(1)")
    print "\n             >>>>>>>  AD MODIFIED   <<<<<<<<"
    ad1.info()
    eq_(ad1[0].extname(), "SCI")
    eq_(ad1[0].extver(), 2)
    eq_(ad1[1].extname(), "SCI")
    eq_(ad1[1].extver(), 3)

def test2():
    """ASTRODATA-remove TEST 2: Use 'XNAM', XVER"""
    ad1 = AstroData(sci123) 
    print "\n             >>>>>>>     AD       <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD remove   <<<<<<<<"
    ad1.remove(("SCI",1))
    print("ad1.remove(('SCI',1))")
    print "\n             >>>>>>>  AD  MODIFIED   <<<<<<<<"
    ad1.info()
    eq_(ad1[0].extname(), "SCI")
    eq_(ad1[0].extver(), 2)
    eq_(ad1[1].extname(), "SCI")

@raises(Errors.AstroDataError)
def test3():
    """ASTRODATA-remove TEST 3: Fail when given index out of range"""
    ad1 = AstroData(sci123) 
    print "\n             >>>>>>>     AD       <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD remove   <<<<<<<<"
    ad1.remove(3)
    print("ad1.remove(3)")
