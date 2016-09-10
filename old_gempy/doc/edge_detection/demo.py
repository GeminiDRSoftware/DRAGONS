#! /usr/bin/env python

import sys
import optparse
import time
from matplotlib import pylab
from astrodata.adutils import logutils


def demo(test_data_path):
    """
       Edge Detection demo script. 
       The input parameter is the pathname to the 'test_data' directory
       of the gemini_python SVN repository.
       It will display images in DS9 and plot in your terminal. Please
       make sure ds9 is running already.

       Using Python shell:
       >>> import demo
       >>> demo.demo('/data1/gemini_python/test_data')

       Using Unix shell:
       demo.py --tdir=<test_data_path> run '

    """

    import os
       
    #from gempy.science import extract as extr
    import extract as extr
    from astrodata import AstroData

    # Set test_dir path to the test files; they are available in the
    # test_data directory in the SVN repository:
    # http://chara.hi.gemini.edu/svn/DRSoftware/gemini_python/test_data/edge_detection/

    if  not os.access(test_data_path,os.F_OK):
        print "\n >>>> ERROR in test_data_path",test_data_path
        return

    test_data_path = os.path.join(test_data_path, 'edge_detection')

    if test_data_path == None:
       print "..... ERROR: Please edit this script and set the 'test_data_path'"

    gnirs_file = os.path.join(test_data_path, 'nN20101215S0475_comb.fits')
    f2_file =    os.path.join(test_data_path, 'fS20100220S0035_comb.fits')
    gmos_file =  os.path.join(test_data_path, 'mgS20100113S0110.fits')


    # EDIT the for loop to include gmos. GMOS is SLOW demo (~4 mins)

    #for fname in [gnirs_file,f2_file,gmos_file]:
    print "Starting time:",time.asctime()
    pylab.interactive(True)
    for fname in [gnirs_file,f2_file]:
       ad = AstroData(fname)

       print ".... 1) Finding slit edges for",ad.filename,' (',ad.instrument(),')'
       adout = extr.trace_slits(ad,debug=True)

       print ".... 2) Cutting slits... \n"
       ad_cuts = extr.cut_slits(adout,debug=True)

       bname = os.path.basename(fname)

       # Writing to FITS file.
       ad_cuts[0].filename='cuts_'+bname
       ad_cuts[0].write(clobber=True)
  
       print ".... 3) Writing FITS file: ",ad_cuts[0].filename," .. in your working directory."
       print ".... 4) Displayinig the whole frame in frame: 1"

    print "Ending time:",time.asctime()

if __name__ == '__main__':
   
     # Parse input arguments
    usage = 'Usage: \n       demo.py --tdir=<test_data_path> run '

    p = optparse.OptionParser(usage=usage, version='autoid 0.0')
    p.add_option('--tdir', action='store', type='string',
                 help='test_data pathname',metavar=2)

    (par, args) = p.parse_args()
    iList= args
    if len(iList) == 0:
       print usage
       print "\n---- Edge detection demo:\n\
              1) It will run trace_slits and cuts_slits ULF in debug mode.\n\
              2) Make sure DS9 is running. \n\
              3) 'test_data_path' is the pathname in your SVN local copy to the \n \
                 .../gemini_python/test_data directory.\n\
              4) Make sure you specify 'test_data_path' pathname in '--tdir='\n"

       sys.exit()
    if args[0] == 'run':
        print "running:",par.tdir
    demo(par.tdir)
