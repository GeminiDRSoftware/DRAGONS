#!/usr/bin/env python
#
# Author: C. Allen and K. Dement
# Date: April 2010
# Desciption: Tool to inspect Primitives

import os, sys
from astrodata.PrimInspect import PrimInspect, primsetcmp
from optparse import OptionParser

#Instantiate OptionParser
parser = OptionParser()
parser.set_description( 
"""The Gemini Primitive Inspection Tool.\n Created by C.Allen, K.Dement, Apr2010.""" )

parser.add_option('-c', '--useColor', action='store_true', dest='useColor', default=False,
                  help='Apply color output scheme')
parser.add_option('-f', '--makeOutputFile', action='store_true', dest='makeOutputFile', default=False,
                  help='Make output textfile (primitives_List.txt).')
parser.add_option('-i', '--showInfo', action='store_true', dest='showInfo', default=False,
                  help='Show information on Primitive class name and location')
parser.add_option('-p', '--showParams', action='store_true', dest='showParams', default=False,
                  help='include parameter information in output')
parser.add_option('-s', '--showSetsOnly', action='store_true', dest='showSetsOnly', default=False,
                  help='Show only Primitives sets')
parser.add_option('-u', '--showUsage', action='store_true', dest='showUsage', default=False,
                  help='Include usage information in output')
parser.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False,
                  help='Include ALL information about Primitives')
(options,  args) = parser.parse_args()
options.datasets = []
options.astrotypes = []
options.args = args
for arg in args:
    if os.path.exists(arg):
        options.datasets.append( arg )
    else:
        options.astrotypes.append( arg )

#Instantiate PrimInspectObject
pin = PrimInspect( options )

#Proceed to show Primitives
pin.datasets = options.datasets
pin.astrotypes = options.astrotypes
pin.buildDictionaries()
primsets = pin.primsdict.keys()
print "lP48:", repr(primsets)

primsets.sort( primsetcmp )
print "lP51:", repr(primsets)

names = []
if options.showSetsOnly:
    pin.show( "\n\n\t\t\t\t${BOLD}PRIMITIVE SETS${NORMAL}" )
    pin.show( "-"*80 ) 
    count = 1
for primset in primsets:
    nam = primset.__class__.__name__
    if nam in names:
        continue
    else:
        names.append( nam )    
    if options.showSetsOnly:
        cl = pin.name2class[ nam ]
        pin.show( "\n\n%2d. ${BOLD}%s${NORMAL}\n" %( count,cl.astrotype ) )
        pin.showSetInfo( nam,cl ) 
        count+=1
    else:
        pin.showPrims( nam )
if options.showSetsOnly:
    pin.show( "\n\n" )
    pin.show( "-"*80 )
else:
    pin.show( "_"*80 )
pin.close_fhandler()


#------------------------------- eof
