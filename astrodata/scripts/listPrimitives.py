#!/usr/bin/env python
import os

from optparse import OptionParser

from astrodata.PrimInspect import PrimInspect

# set up commandline args and options
parser = OptionParser()
parser.set_description( "Gemini Observatory Primitive Inspection Tool "
                       "(v_1.0 2011)")
parser.add_option("-c", "--use-color", action="store_true", dest="use_color",
                  default=False, help="apply color output scheme")
parser.add_option("-e", "--engineering", action="store_true", dest="engineering",
                  default=False, help="include engineering recipes")
parser.add_option("-f", "--copy-tofile", action="store_true", 
                  dest="copy_tofile", default=False,
                  help="write to file (primitives_list.txt).")
parser.add_option("-i", "--info", action="store_true", dest="info",
                  default=False,
                  help="show primitive set information")
parser.add_option("-p", "--parameters", action="store_true", dest="parameters",
                  default=False,
                  help="show parameters")
parser.add_option("-r", "--recipes", action="store_true", dest="recipes",
                  default=False, help="list top recipes")
parser.add_option("-s", "--primitive-set", action="store_true", 
                  dest="primitive_set", default=False,
                  help="show primitive by set")
parser.add_option("-u", "--usage", action="store_true", dest="usage",
                  default=False, help="show usage")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose",
                  default=False, help="set verbose mode")
parser.add_option("--primitives", action="store_true", dest="primitives",
                  default=False, help="show all primitives for set")
parser.add_option("--view-recipe", dest="view_recipe", 
                  default=None, help="show all primitives for set")

(options,  args) = parser.parse_args()
optc = options.use_color
optp = options.parameters
optu = options.usage
opti = options.info
optv = options.verbose
optf = options.copy_tofile
opte = options.engineering
oview = options.view_recipe
options.primitives = True

# distinguish between data and astrotype in args
datasets = []
astrotypes = []

if options.recipes or oview:
    pin = PrimInspect(use_color=optc)
    pin.list_recipes(pkg="Gemini",eng=opte, view=oview)
else:
    for arg in args:
        if os.path.exists(arg) and not os.path.isdir(arg):
            datasets.append(arg)
        else:
            astrotypes.append(arg)
    pin = PrimInspect(use_color=optc, show_param=optp, show_usage=optu,
                      show_info=opti, make_file=optf, verbose=optv,
                      datasets=datasets, astrotypes=astrotypes)

    # execution
    if options.primitive_set:
        pin.show_primitive_sets()
    elif options.primitives:
        pin.show_primitive_sets(prims=True)
