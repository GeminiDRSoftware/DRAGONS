import sys, os
import subprocess
from optparse import OptionParser
from astrodata import Errors
import time
#import test_astrodata
import inspect

parser = OptionParser()
parser.set_description(
"""AstroData Self-Test.\n
Created by C.Allen, K.Dement, 01May2011.""" )

parser.add_option('-r','--runall', action='store_true', dest='runall',\
                    default=False, help='run all AstroData class tests')
parser.add_option('-v','--verbose', action='store_true', dest='verbose',\
                    default=False, help='see standard out')
(options,  args) = parser.parse_args()
verbose = options.verbose

def astrodata_trunner(ad_test):
    '''Helper function for python nose
    '''
    #verbose = False
    inlist = []
    inlist.append('python')
    inlist.append(ad_test)
    if verbose:
        retcode = subprocess.call(inlist)
        if retcode > 0:
            raise Errors.DescriptorValueTypeError
    else:
        retcode = subprocess.call(inlist, stdout=open('/dev/null','w'))
        if retcode > 0:
            raise Errors.DescriptorValueTypeError

def test_deepcopy():
    astrodata_trunner('adtest_deepcopy.py')

def test_inferred_extname():
    astrodata_trunner('adtest_inferred_extname.py')

def test_dv_operators():
    astrodata_trunner('adtest_dv_operators.py')



# Run tests using -r option in interpretor
if options.runall:
    adtests = os.listdir('.') 
    testlist = []
    for adtest in adtests:
        if adtest[:7] == 'adtest_':
            temp = adtest[7:]
            testlist.append(temp[:-3])
    times = []
    numpass = 0
    numfail = 0
    print('\nrunning AstroData class self-tests...\n')
    for tes in testlist:
        timerStart = time.time() 
        try:
            exec('test_%s()' % tes)
            print '.',
            numpass += 1
        except:
            print 'F',
            numfail += 1
        times.append(time.time() - timerStart)
    total = 0
    for tim in times:
        total += tim
    print("\n")
    if verbose:
        for i in range(len(testlist)):
            print("%s (%.5fs)" % (testlist[i],times[i]))
    print('-'*60)
    print("Ran %i tests in %.2fs" % (len(testlist), total))
    #print "\nResults: Pass = ", numpass, " Fail = ", numfail
    if numfail == 0:
        print('\nOK')
    else:
        print('\nFAILED (%i)' % numfail)
