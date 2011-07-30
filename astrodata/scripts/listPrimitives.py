#!/usr/bin/env python
import os

from optparse import OptionParser

from astrodata.PrimInspect import PrimInspect

# set up commandline args and options
parser = OptionParser()
parser.set_description( "Gemini Observatory Primitive Inspection Tool "
                       "(v_1.0 2011)")
parser.add_option('-c', '--use-color', action='store_true', dest='use_color',
                  default=False, help='apply color output scheme')
parser.add_option('-f', '--make-file', action='store_true', 
                  dest='make_file', default=False,
                  help='write report to file (primitives_list.txt).')
parser.add_option('-i', '--show-info', action='store_true', dest='show_info',
                  default=False,
                  help='show information on Primitive class name and location')
parser.add_option('-p', '--show-param', action='store_true', dest='show_param',
                  default=False,
                  help='show parameters')
parser.add_option('-s', '--show-set', action='store_true', 
                  dest='show_set', default=False,
                  help='show only primitve sets')
parser.add_option('-u', '--show-usage', action='store_true', dest='show_usage',
                  default=False, help='show usage')
parser.add_option('-v', '--verbose', action='store_true', dest='verbose',
                  default=False, help='set verbose mode')
(options,  args) = parser.parse_args()
optc = options.use_color
optp = options.show_param
optu = options.show_usage
opti = options.show_info
optv = options.verbose
optf = options.make_file

# distinguish between data and astrotype in args
datasets = []
astrotypes = []
for arg in args:
    if os.path.exists(arg) and not os.path.isdir(arg):
        datasets.append(arg)
    else:
        astrotypes.append(arg)

# create primitive inspect object
pin = PrimInspect(use_color=optc, show_param=optp, show_usage=optu,
                  show_info=opti, make_file=optf, verbose=optv)
pin.datasets = datasets
pin.astrotypes = astrotypes
pin.build_dictionaries()
primsets = pin.primsdict.keys()

primsets.sort(pin.primsetcmp)

names = []
if options.show_set:
    pin.show("\n" + "="*79) 
    pin.show("${BOLD}PRIMITIVE SETS${NORMAL}")
    pin.show("="*79) 
    count = 1
for primset in primsets:
    nam = primset.__class__.__name__
    if nam in names:
        continue
    else:
        names.append(nam)    
    if options.show_set:
        cl = pin.name2class[ nam ]
        if len(primsets) == 1:
            pin.show("\n  ${BOLD}%s${NORMAL}\n" % cl.astrotype)
        else:
            pin.show("\n%2d. ${BOLD}%s${NORMAL}\n" %(count,cl.astrotype))
        primlist = pin.primsdict_kbn[nam]
        pin.show_set_info(nam, cl, primlist) 
        count += 1
    else:
        pin.showPrims(nam)
pin.show("\n" + "="*79) 

