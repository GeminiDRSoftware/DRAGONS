#!/usr/bin/env python

from astrodata.AstroDataType import *
from optparse import OptionParser

cl = getClassificationLibrary()
# print repr(cl.typesDict)
#FROM COMMANDLINE WHEN READY
parser = OptionParser()
(options, args) = parser.parse_args()

if len(args)>0:
    astrotype = args[0]
else:
    astrotype = None
a = cl.gvizDoc(astrotype= astrotype, writeout = True)
