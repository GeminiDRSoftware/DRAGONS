#!/usr/bin/env python

# import pdb

import astrodata 
from astrodata.DataSpider import *
from astrodata.AstroData import *
from optparse import OptionParser

from utils import terminal
from utils.terminal import TerminalController

        
############################################################
# this script was developed to exercise the GeminiDataType class
# but now serves a general purpose in addition to that and as
# a demo for GeminiData... see options documentation.

REASLSTDOUT = sys.stdout
REALSTDERR = sys.stderr
fstdout = terminal.FilteredStdout()
fstdout.addFilter( terminal.ColorFilter())
sys.stdout = fstdout   
termsize = terminal.getTerminalSize()

# parsing the command line
parser = OptionParser()
parser.add_option("-t", "--typewalk", dest="twdir", default =".",
        help="Recursively walk given directory and put type information to stdout.")
parser.add_option("-o", "--only", dest="only", default="all",
        help= "Choose only certain types to find, list multiple types separated by commas with NO WHITESPACE.")
parser.add_option("-v", "--showdescriptors", dest="showdescriptors", default=None,
        help = "Show descriptors listed, separate with commas and NO WHITESPACE" )
parser.add_option("-l", "--listdescriptors", dest="listdescriptors", action="store_true",
        help = "Lists available descriptors")
parser.add_option("-p", "--printheader", metavar="PHEADLIST", dest="pheads", default=None,
        help= "Headers to print out for found files, list keywords separated by comma, no whitespace.")
parser.add_option("-d", "--htmldoc", dest="htmldoc", action="store_true",
        help="Show Classification Dictionary as html page")
parser.add_option("-i", "--info", dest="showinfo", action="store_true",
        help="Show file info as in 'pyfits.HDUList.info()'")     
parser.add_option("-s", "--status", dest="onlyStatus", action="store_true",
        help="Compare only to the processing status dictionary of classifications")
parser.add_option("-y", "--typology", dest="onlyTypology", action="store_true",
        help="Compare only to the processing status dictionary of classifications")
parser.add_option("-f", "--filemask", dest="filemask", default = None)

# REVIEW THIS, -v option above is available
if (False):# Now the descriptors, in alphabetical order
    parser.add_option("-A", "--airmass", dest="showairmass", action="store_true",
            help="Show airmass using appropriate descriptor")     
    parser.add_option("-C", "--camera", dest="showcamera", action="store_true",
            help="Show camera using appropriate descriptor")     
    parser.add_option("-V", "--cwave", dest="showcwave", action="store_true",
            help="Show cwave using appropriate descriptor")     
    parser.add_option("-B", "--datasec", dest="showdatasec", action="store_true",
            help="Show datasec using appropriate descriptor")     
    parser.add_option("-H", "--detsec", dest="showdetsec", action="store_true",
            help="Show detsec using appropriate descriptor")     
    parser.add_option("-U", "--disperser", dest="showdisperser", action="store_true",
            help="Show disperser using appropriate descriptor")     
    parser.add_option("-E", "--exptime", dest="showexptime", action="store_true",
            help="Show exptime using appropriate descriptor")     
    parser.add_option("-F", "--filtername", dest="showfiltername", action="store_true",
            help="Show filtername using appropriate descriptor")     
    parser.add_option("-Q", "--filterid", dest="showfilterid", action="store_true",
            help="Show filterid using appropriate descriptor")     
    parser.add_option("-K", "--fpmask", dest="showfpmask", action="store_true",
            help="Show fpmask using appropriate descriptor")     
    parser.add_option("-G", "--gain", dest="showgain", action="store_true",
            help="Show gain using appropriate descriptor")     
    parser.add_option("-I", "--instrument", dest="showinstrument", action="store_true",
            help="Show instrument using appropriate descriptor")     
    parser.add_option("-M", "--mdfrow", dest="showmdfrow", action="store_true",
            help="Show mdfrow using appropriate descriptor")     
    parser.add_option("-L", "--nonlinear", dest="shownonlinear", action="store_true",
            help="Show nonlinear using appropriate descriptor")     
    parser.add_option("-Z", "--nsciext", dest="shownsciext", action="store_true",
            help="Show nsciext using appropriate descriptor")     
    parser.add_option("-J", "--object", dest="showobject", action="store_true",
            help="Show object using appropriate descriptor")     
    parser.add_option("-O", "--obsmode", dest="showobsmode", action="store_true",
            help="Show obsmode using appropriate descriptor")     
    parser.add_option("-P", "--pixscale", dest="showpixscale", action="store_true",
            help="Show pixscale using appropriate descriptor")     
    parser.add_option("-N", "--rdnoise", dest="showrdnoise", action="store_true",
            help="Show rdnoise using appropriate descriptor")     
    parser.add_option("-S", "--satlevel", dest="showsatlevel", action="store_true",
            help="Show satlevel using appropriate descriptor")     
    parser.add_option("-D", "--utdate", dest="showutdate", action="store_true",
            help="Show utdate using appropriate descriptor")     
    parser.add_option("-T", "--uttime", dest="showuttime", action="store_true",
            help="Show uttime using appropriate descriptor")     
    parser.add_option("-W", "--wdelta", dest="showwdelta", action="store_true",
            help="Show wdelta using appropriate descriptor")     
    parser.add_option("-R", "--wrefpix", dest="showwrefpix", action="store_true",
            help="Show wrefpix using appropriate descriptor")     
    parser.add_option("-X", "--xbin", dest="showxbin", action="store_true",
            help="Load descriptor for given data type and retrieve 'xbin'.")
    parser.add_option("-Y", "--ybin", dest="showybin", action="store_true",
            help="Show non-linear limit using appropriate descriptor")

(options, args) = parser.parse_args()
        
# start the Gemini Specific class code

dt = DataSpider()
cl = dt.getClassificationLibrary()

if (options.listdescriptors):
    from astrodata import Descriptors
    import CalculatorInterface
    funs = dir(CalculatorInterface.CalculatorInterface)
    descs = []
    for fun in funs:
        if "_" not in fun:
            descs.append(fun)
    print ", ".join(descs)
    
elif (options.htmldoc):
    print cl.htmlDoc()
else:
    try:
        dt.typewalk(options.twdir, only=options.only,
                    pheads = options.pheads, showinfo = options.showinfo,
                    onlyStatus = options.onlyStatus,
                    onlyTypology = options.onlyTypology,
                    # generic descriptor interface,
                    showDescriptors = options.showdescriptors,
                    filemask = options.filemask
                    )
    except KeyboardInterrupt:
        print "Interrupted by Control-C"
