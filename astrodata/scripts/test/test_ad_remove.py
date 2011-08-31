from nose.tools import *

import file_urls 
from astrodata import AstroData
from astrodata import Errors


def ad_remove_test1():
    """ad_remove_test1 -given ad index, remove ext
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    print "\n             >>>>>>>     AD       <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD remove   <<<<<<<<"
    ad1.remove(1)
    print("ad1.remove(1)")
    print "\n             >>>>>>>  AD  (NEW)     <<<<<<<<"
    ad1.info()
    eq_(ad1[0].extname(), "SCI")
    eq_(ad1[0].extver(), 2)
    eq_(ad1[1].extname(), "SCI")
    eq_(ad1[1].extver(), 3)

def ad_remove_test2():
    """ad_remove_test2 -given (EXTNAME,EXTVER), remove ext
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    print "\n             >>>>>>>     AD       <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD remove   <<<<<<<<"
    ad1.remove(("SCI",1))
    print("ad1.remove(('SCI',1))")
    print "\n             >>>>>>>  AD  (NEW)     <<<<<<<<"
    ad1.info()
    eq_(ad1[0].extname(), "SCI")
    eq_(ad1[0].extver(), 2)
    eq_(ad1[1].extname(), "SCI")

@raises(Errors.AstroDataError)
def ad_remove_test3():
    """ad_remove_test1 -given index out of range, raise
    """
    ad1 = AstroData(file_urls.testdatafile_1) #sci 3
    print "\n             >>>>>>>     AD       <<<<<<<<"
    ad1.info()
    print "\n             >>>>>>>    AD remove   <<<<<<<<"
    ad1.remove(3)
    print("ad1.remove(3)")
