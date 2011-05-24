import sys
import os
from optparse import OptionParser

from nose.tools import *

import file_urls 
from astrodata import AstroData

testfile = file_urls.testdatafile_1

def ad_close_test1():
    """AD_CLOSE: test1 -
    """
    print("\n\tTest input file: %s" % testfile)
    ad = AstroData(testfile)

