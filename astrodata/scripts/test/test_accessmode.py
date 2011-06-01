from nose.tools import *

import file_urls 
from astrodata import AstroData
from astrodata import Errors

testfile = file_urls.testdatafile_1

def accessmode_test1():
    '''ACCESSMODE: test1 -load ad readonly and try to write  
    '''
    print('\n\t* testfile: %s' % testfile)
    ad = AstroData(testfile)
    print('\tad = AstroData(testfile)')
    print('\tad.write()')
    throwexcept = False
    try:
        ad.write()
    except Errors.AstroDataError,e:
        throwexcept = True
        print("\t%s" % e)
    assert_true(throwexcept)

def accessmode_test2():
    '''ACCESSMODE: test2 -  
    '''
    
