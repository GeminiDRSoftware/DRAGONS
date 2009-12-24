#!/usr/bin/env python
import time
ost = time.time()
import os
import sys
from optparse import OptionParser
from iq import getiq

'''
@author: Jen Holt
@author: River Allen
'''

if __name__ == '__main__':
    # Parse input arguments
    usage = 'usage: %prog [options] datasets'
    VERSION = '1.1'
    p = OptionParser(usage=usage, version='v'+VERSION)
    p.add_option('--debug', action='store_true', help='toggle debug messages')
    p.add_option('--verbose', '-v', action='store_true', help='toggle on verbose mode')
    p.add_option('--list', '-l', action='store', type='string', dest='inlist', help='list of input datasets')
    p.add_option('--function', action='store', type='string', dest='function', default='both', help='function to fit [gauss|moffat|both](default: %default)')
    p.add_option('--residuals', action='store_true', help='keep residual images')
    p.add_option('--display', action='store_true', help='toggle on display')
    p.add_option('--inter', action='store_true', help='toggle on interactive mode')
    p.add_option('--rawpath', action='store', type='string', dest='rawpath', default='.', help='location of input images')
    p.add_option('--prefix', action='store', type='string', default='auto', dest='prefix', help='prefixes to use for intermediate data')
    p.add_option('--observatory', action='store', type='string', dest='observatory', default='gemini-north', help='observatory, [gemini-north|gemini-south] (default=%default)')
    p.add_option('--noclip', action='store_true', default=False, help='toggle off sigma clipping of FWHM and ellipticity measurements')
    p.add_option('--sigma', action='store', type='float',  dest='sigma', default=2.3, help='threshold for sigma clipping (default=%default)')
    p.add_option('--pymark', action='store_true', help='mark daofind outputs')
    p.add_option('--niters', action='store', type='int', dest='niters', default=4, help='iteration for sigma clipping (default=%default)')
    p.add_option('--boxsize', action='store', type='int', dest='boxsize', default=8, help='size of thumbnail to fit in arc seconds (default=%default ")')
    p.add_option('--outfile', action='store', type='string', dest='outfile', default='default', help='name of the output data file, if default the name will be input image name with ".dat"')
    p.add_option('--garbageStat', action='store_true', dest='garbageStat', default=False, help='turn on garbage statistics for less than four objects')
    p.add_option('--qa', action='store_true', dest='qa', default=False, help='QA mode. Faster, but potentially worse.')
    (options, args) = p.parse_args()
    
    if options.display!=True:
        options.pymark = False

    # Set default
    doclip=True
    if options.noclip: doclip=False
    
    if options.debug:
        options.verbose = True
        print 'options: ', options
        print 'args: ', args

    # Set defaults
    datasuffix = '.dat'
    
    # Set list of input dataset
    inimages = []
    if options.inlist != None:
        f = open(options.inlist, mode='r')
        lines = f.readlines()
        f.close()
        inimages = map((lambda i: matchnewline.sub('',lines[i])),range(len(lines)))
    else:
        inimages = args
    
    
    # Call iq routine
    for image in inimages:
        st = time.time()
        #outfile=matchfits.sub(datasuffix, image)
        outfile= os.path.basename( image ).split( os.path.extsep )[0]
        getiq.gemiq(image, outFile=options.outfile, function=options.function, 
            verbose=options.verbose, residuals=options.residuals,
            display=options.display, interactive=options.inter,
            rawpath=options.rawpath, prefix=options.prefix,
            observatory=options.observatory, clip=doclip,
            sigma=options.sigma, pymark=options.pymark, niters=options.niters,
            boxSize=options.boxsize, debug=options.debug, garbageStat=options.garbageStat,
            qa=options.qa)
        et = time.time()
        print '%s time: %04d (seconds)' %( image, et-st)
    
    
    
oet = time.time()
print '-' * 40
print 'Overall time: %04d (seconds)' %(oet-ost)
