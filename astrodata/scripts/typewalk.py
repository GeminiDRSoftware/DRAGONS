#!/usr/bin/env python
import sys
import os
# remove current working directory from PYTHONPATH to speed up import in
# gigantic data directories
# print repr(sys.path)

import cProfile

curpath = os.getcwd()
# @@REVIEW Note: This is here because it's very confusing when someone runs a script IN the
# package itself.  This helps a little... perhaps a warning would be better.
if (curpath in sys.path):
    sys.path.remove(curpath)
# print repr(sys.path)

opti = False
if opti:
    print "Starting Main Imports"
# import pdb
try:
    import inspect
    import astrodata 
    from astrodata.DataSpider import *
    from astrodata.AstroData import *
    from optparse import OptionParser


    import traceback as tb
except:
    raise
    
if False:
    from astrodata.LocalCalibrationService import CalibrationService

if opti:
    print "Finished Top Imports"
############################################################
# this script was developed to exercise the GeminiDataType class
# but now serves a general purpose in addition to that and as
# a demo for GeminiData... see options documentation.

# parsing the command line
parser = OptionParser(usage = "usage: %prog [options] <path>")
parser.add_option("-t", "--typewalk", dest="twdir", default =".",
        help="Recursively walk given directory and put type information to stdout.")
parser.add_option("-o", "--only", dest="only", default="all",
        help= "Choose only certain types to find, list multiple types separated by commas with NO WHITESPACE.")
parser.add_option("-v", "--showdescriptors", dest="showdescriptors",
        default=None,
        help = """Show descriptors listed. User should separate descriptor names
with commas and NO WHITESPACE. The "-l" option can be used to show available
descriptor names. Special values "all" and "err" can be used instead of
descriptor names.  The "all" argument shows all descriptors and "err" argument
shows all
descriptors that are failing (for descriptor debugging).
""" )
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
parser.add_option("--showcalibrations", dest="showCals", action="store_true",
        help="When set, show any locally available calibrations")
parser.add_option("-x", "--dontrecurse", dest="stayTop", action="store_true",
        help="When set, don't recurse subdirs.")
parser.add_option("-b", "--batch", dest="batchnum", default = 100,
        help="In -x shallow walk mode... how many files to process at a time in the current directory. This helps control behavior in large data directories. Default = 100.")
parser.add_option("--force-width", dest = "forceWidth", default=None,
                  help="Use to force width of terminal for output purposes instead of using actual temrinal width.")
parser.add_option("--force-height", dest = "forceHeight", default=None,
                  help="Use to force height of terminal for output purposes instead of using actual temrinal height.")
parser.add_option("--raiseexception", dest="raiseExcept", action="store_true",
        help="Throw exceptions on some failures, e.g. failed descriptor calls to allow debugging of the problem.")
parser.add_option("-k", "--showstack", dest="showStack", action="store_true",
        help="When a high level KeyboardInterrupt is caught, show the stack.")
parser.add_option("-w", "--where", dest="where", default = None,
        help="Allows a condition to test, should use descriptors") 
parser.add_option("-r", "--recipe", "--reduce", dest="recipe", default = None,
        help="Allows a condition to test, should use descriptors") 
parser.add_option("-c", "--color", dest = "usecolor", action = "store_true", 
        help="Colorizes the display")
(options, args) = parser.parse_args()

# allow paths to be given as arguments
if len(args) > 0:
    for path in args:
        if os.path.exists(path):
            os.chdir(path)

if options.usecolor == False or options.usecolor == None:
    import os
    os.environ["TERM"] = ""
from astrodata.adutils import terminal
from astrodata.adutils.terminal import TerminalController
REASLSTDOUT = sys.stdout
REALSTDERR = sys.stderr
fstdout = terminal.FilteredStdout()
fstdout.addFilter( terminal.ColorFilter())
sys.stdout = fstdout   
termsize = terminal.getTerminalSize()

#set up terminal
terminal.forceWidth = options.forceWidth
terminal.forceHeight = options.forceHeight


# start the Gemini Specific class code

dt = DataSpider()
cl = dt.get_classification_library()

showStack = True
if (options.listdescriptors):
    from astrodata import Descriptors
    CalculatorInterface = get_calculator_interface()
    funs = inspect.getmembers(CalculatorInterface, inspect.ismethod)
    descs = []
    print "${UNDERLINE}Available Descriptors${NORMAL}"

    for funtuple in funs:
        funame = funtuple[0]
        fun = funtuple[1]
        if "_" != funame[0]:
            descs.append(funame)
    if True:
        print "\t"+"\n\t".join(descs)
    else:
        #this makes something that looks like a python list with quoted desc names
        qdescs = [ '"%s"' % s for s in descs]
        print "["+", ".join(qdescs)+"]"
elif (options.htmldoc):
    print cl.html_doc()
else:
    try:
        osd = options.showdescriptors
        
        if (osd == "all" or osd == "err"):
            CalculatorInterface = get_calculator_interface()
            descs = []
            if osd == "err":
                descs.append("err")
            
            funs = inspect.getmembers(CalculatorInterface, inspect.ismethod)

            for funtuple in funs:
                funame = funtuple[0]
                fun = funtuple[1]
                if "_" != funame[0]:
                    descs.append(funame)

                    options.showdescriptors = ",".join(descs) 

        if opti:
            print "Calling DataSpider.typewalk(..)"
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
                    batchnum = int(options.batchnum)-1,
                    opti = opti,
                    recipe = options.recipe
                    )
        if opti:
            print "Done DataSpider.typewalk(..)"
    except KeyboardInterrupt:
    
        print "Interrupted by Control-C"
        if False: #(showStack):
            st = tb.extract_stack()
            print repr(st)
