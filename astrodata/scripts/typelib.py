#!/usr/bin/env python

from astrodata.AstroDataType import *
from optparse import OptionParser
import os

cl = getClassificationLibrary()
# print repr(cl.typesDict)
#FROM COMMANDLINE WHEN READY
parser = OptionParser()
(options, args) = parser.parse_args()

if len(args)>0:
    astrotype = args[0]
else:
    astrotype = None
    
import astrodata
import astrodata.RecipeManager as rm

assdict = rm.centralPrimitivesIndex
a = cl.gvizDoc(astrotype= astrotype, writeout = True, assDict = assdict)
import webbrowser
url = "file://"+os.path.join(os.path.abspath("."), "gemdtype.viz.svg")
webbrowser.open(url);
