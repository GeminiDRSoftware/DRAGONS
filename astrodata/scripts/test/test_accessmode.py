from nose.tools import *

import file_urls 
from astrodata import AstroData

testfile = file_urls.testdatafile_1

@raises(Exception)
def accessmode_test1():
    '''ACCESSMODE: test1 -load ad readonly and try to write  
    '''
    print('\n\t* testfile: %s' % testfile)
    ad = AstroData(testfile)
    print('\tad = AstroData(testfile)')
    print('\tad.write()')
    ad.write()

@raises(Exception)
def accessmode_test2():
    '''ACCESSMODE: test2 -subdata will not close the original hdulist  
    '''
    raise typeError("pass?")
    
