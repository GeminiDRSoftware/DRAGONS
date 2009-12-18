#!/bin/env python

# import pdb

from DataSpider import *
from AstroData import *
from optparse import OptionParser
        
############################################################
# this script was developed to exercise the GeminiDataType class
# but now serves a general purpose in addition to that and as
# a demo for GeminiData... see options documentation.

# parsing the command line
parser = OptionParser()
parser.add_option("-t", "--typewalk", dest="twdir", default =".",
        help="Recursively walk given directory and put type information to stdout.")
parser.add_option("-o", "--only", dest="only", default="all",
        help= "Choose only certain types to find, list multiple types separated by commas with NO WHITESPACE.")
parser.add_option("-v", "--showdescriptors", dest="showdescriptors", default=None,
        help = "Show descriptors listed, separate with commas and NO WHITESPACE" )
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

(options, args) = parser.parse_args()
        
# start the Gemini Specific class code

dt = DataSpider()
cl = dt.getClassificationLibrary()

if (options.htmldoc):
    print cl.htmlDoc()
else:
    try:
        dt.typewalk(options.twdir, only=options.only,
                    pheads = options.pheads, showinfo = options.showinfo,
                    onlyStatus = options.onlyStatus,
                    onlyTypology = options.onlyTypology,
                    # generic descriptor interface,
                    showDescriptors = options.showdescriptors)
    except KeyboardInterrupt:
        print "Interrupted by Control-C"
