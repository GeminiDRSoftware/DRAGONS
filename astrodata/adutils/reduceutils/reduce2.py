#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                              astrodata/scripts
#                                                                     reduce2.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Rev$'[11:-3]
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
# reduce2.py -- refactored reduce, cl parsing exported, functionalized.
#               see parseUtils.py
# ------------------------------------------------------------------------------
__version__ = '2.0'
# ------------------------------------------------------------------------------
"""
This module (reduce2) provides a functionalized interface to running the core 
processing of the reduce pipeline. This core functionality is provided by the 
imported coreReduce module. 

It provides both a nominal 'reduce' command line interface and a 'main'
function that can be called with an 'args' parameter.

Eg.,

Get "args' from the defined reduce parser:

>>> args = parseUtils.buildParser(version).parse_args()
>>> import reduce2
>>> reduce2.main(args)

In the above example, 'args' is

-- argparse Namespace instance

Use of the parseUtils function buildParser will get the caller a fully defined 
reduce Namespace instance, values for which can be then be adjusted as desired.

Eg.,

parseUtils.buildParser:
----------------------
>>> args = parseUtils.buildParser(version).parse_args()
>>> args.logfile
'reduce.log'
>>> args.files
[]
>>> args.files.append('some_fits_file.fits')

Once 'args' attributes have been appropriately set, the caller then simply 
calls main():

>>> reduce2.main(args)
"""
import sys

from signal import SIGTERM
# ---------------------------- Package Import ----------------------------------
from astrodata.adutils import logutils

import parseUtils
from coreReduce import Reduce
# ------------------------------------------------------------------------------
def main(args):
    """
    See the module docstring on how to call main.

    parameters: <inst>, 'args' object
    return:     <int>,   exit code
    """
    global log
    # --------------------------------------------------------------------------
    # exit status
    estat    = 0

    logutils.config(mode=args.logmode,
                    console_lvl=args.loglevel,
                    file_name=args.logfile)

    log = logutils.get_logger(__name__)
    log.stdinfo("\t\t\t--- reduce, v%s ---" % __version__)
    r_reduce = Reduce(args)
    try:
        estat = r_reduce.runr()
    except Exception as err:
        estat = SIGTERM
        log.error("main() caught unhandled exception:")
        log.error(type(err))
        log.error(str(err))

    log.stdinfo("reduce exited on status %d" % estat)
    return (estat)

# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = parseUtils.buildParser(__version__)
    args   = parser.parse_args()

    if args.displayflags:
        parseUtils.show_parser_options(parser, args)
        for item in ["Input fits file(s):\t%s" % inf for inf in args.files]:
            print item
        sys.exit()

    # Deal with argparse structures, which are different than optparse 
    # implementation. astrotype, recipename, etc. are now lists.
    args   = parseUtils.normalize_args(args)

    sys.exit(main(args))
