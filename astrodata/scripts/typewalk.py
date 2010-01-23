#!/usr/bin/env python

# import pdb
try:
    import astrodata 
    from astrodata.DataSpider import *
    from astrodata.AstroData import *
    from optparse import OptionParser

    from utils import terminal
    from utils.terminal import TerminalController

    import traceback as tb
except:
    raise
if False:
    from astrodata.LocalCalibrationService import CalibrationService

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
parser.add_option("-f", "--filemask", dest="filemask", default = None,
        help="Only files matching the given regular expression will be displayed")
parser.add_option("-c", "--showcalibrations", dest="showCals", action="store_true",
        help="When set, show any locally available calibrations")
parser.add_option("-x", "--dontrecurse", dest="stayTop", action="store_true",
        help="When set, don't recurse subdirs.")
parser.add_option("--force-width", dest = "forceWidth", default=None,
                  help="Use to force width of terminal for output purposes instead of using actual temrinal width.")
parser.add_option("--force-height", dest = "forceHeight", default=None,
                  help="Use to force height of terminal for output purposes instead of using actual temrinal height.")
parser.add_option("-r", "--raiseexception", dest="raiseExcept", action="store_true",
        help="Throw exceptions on some failures, e.g. failed descriptor calls to allow debugging of the problem.")
parser.add_option("-k", "--showstack", dest="showStack", action="store_true",
        help="When a high level KeyboardInterrupt is caught, show the stack.")
parser.add_option("-w", "--where", dest="where", default = None,
        help="Allows a condition to test, should use descriptors") 

(options, args) = parser.parse_args()

#set up terminal
terminal.forceWidth = options.forceWidth
terminal.forceHeight = options.forceHeight


# start the Gemini Specific class code

dt = DataSpider()
cl = dt.getClassificationLibrary()

showStack = True
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
                    filemask = options.filemask,
                    showCals = options.showCals,
                    stayTop = options.stayTop,
                    raiseExcept = options.raiseExcept,
                    where = options.where,
                    )
    except KeyboardInterrupt:
    
        print "Interrupted by Control-C"
        if (showStack):
            st = tb.extract_stack()
            print repr(st)
